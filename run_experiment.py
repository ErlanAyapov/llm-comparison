#!/usr/bin/env python3
"""Прогон эксперимента по замороженным задачам (tasks/*.jsonl), протокол п. 6.

Ответы НЕ оцениваются — пишутся как есть вместе со всеми метриками.
Возобновляемый: пары (модель, qid) с успешной записью пропускаются; записи с error
повторяются при следующем запуске (в файле остаются обе, оценка берёт последнюю).

Примеры:
  python run_experiment.py                         # все модели, E1+E2+E3
  python run_experiment.py --exp e2 --limit 10     # пилот
  python run_experiment.py --models qwen3:8b sherkala-chat --run-id pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import dotenv
import yaml

import prompts
from utils.doc_store import HashMismatchError, verify_hash
from utils.ollama_client import OllamaClient

log = logging.getLogger("run")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def prompt_sha(messages) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(p: Path, rec: dict):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ps_snapshot(client: OllamaClient, model: str) -> dict | None:
    for m in client.ps():
        if m.get("name") == model or m.get("model") == model:
            return {k: m.get(k) for k in ("size", "size_vram", "context_length", "expires_at", "digest")}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    # e2 / e2_ru — тест «вырезка» (build_e2_cut.py); e2_offtopic — прежний E2 (вопросы не по теме, архив)
    ap.add_argument("--exp", nargs="+", default=["e1", "e2", "e2_ru", "e3"],
                    choices=["e1", "e2", "e2_ru", "e3", "e2_offtopic"])
    ap.add_argument("--models", nargs="*", help="подмножество моделей (по умолчанию все из config)")
    ap.add_argument("--limit", type=int, help="не больше N задач на эксперимент (пилот)")
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-unload", action="store_true",
                    help="не выгружать модель перед прогоном (по умолчанию выгружаем, чтобы начать с пустым KV-кэшем)")
    ap.add_argument("--allow-digest-change", action="store_true",
                    help="не останавливаться, если digest модели отличается от зафиксированного в manifest")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dotenv.load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    host = os.getenv("HOST") or cfg["host"]
    gen = cfg["generation"]
    tasks_dir = Path(cfg["paths"]["tasks_dir"])
    manifest = json.loads((tasks_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_version"] != prompts.PROMPT_VERSION:
        sys.exit("prompts.PROMPT_VERSION не совпадает с manifest — пересоберите задачи (build_tasks.py)")
    for k in ("num_ctx", "num_predict", "temperature", "seed"):
        if manifest["generation"][k] != gen[k]:
            sys.exit(f"generation.{k} в config отличается от manifest — пересоберите задачи")

    models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
    client = OllamaClient(host, timeout=gen["timeout_s"])
    tags = client.tags()
    server_version = client.version()

    out_root = Path(cfg["paths"]["results_dir"]) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    models_meta_path = out_root / "models.json"
    models_meta = json.loads(models_meta_path.read_text(encoding="utf-8")) if models_meta_path.exists() else {}

    tasks = {e: [json.loads(l) for l in open(tasks_dir / f"{e}.jsonl", encoding="utf-8")] for e in args.exp}
    if args.limit:
        tasks = {e: t[:args.limit] for e, t in tasks.items()}

    options = {"num_ctx": gen["num_ctx"], "num_predict": gen["num_predict"],
               "temperature": gen["temperature"], "seed": gen["seed"]}
    session_id = uuid.uuid4().hex[:12]

    for m in models:
        name, think = m["name"], m["think"]
        if name not in tags:
            log.error("Модель %s не найдена на сервере — пропуск", name)
            continue
        digest = tags[name]["digest"]
        expected = manifest["models"].get(name, {}).get("digest")
        if expected and expected != digest and not args.allow_digest_change:
            sys.exit(f"{name}: digest {digest[:12]} != manifest {expected[:12]} (модель изменилась)")
        if name not in manifest["models"]:
            sys.exit(f"{name}: нет в manifest — задачи собраны без этой модели, пересоберите")

        show = client.show(name)
        meta = models_meta.setdefault(name, {"sessions": []})
        meta.update({"digest": digest, "size": tags[name].get("size"), "details": show.get("details"),
                     "capabilities": show.get("capabilities"), "modelfile_parameters": show.get("parameters"),
                     "think_sent": think, "ollama_version": server_version})
        sess = {"session_id": session_id, "started_at": datetime.now().isoformat(timespec="seconds")}
        if not args.no_unload:
            client.unload(name)                                   # сброс кэша префиксов (KV) Ollama
            sess["unloaded_before_run"] = True
        seen_prompts: list[str] = []                              # для оценки доли закэшированного префикса

        if not args.no_warmup:
            log.info("[%s] прогрев...", name)
            w = client.chat(name, prompts.build_messages("Тест.", "Жауап бер: иә."), options, think, gen["keep_alive"])
            sess["warmup"] = {k: w[k] for k in ("load_duration", "total_duration", "prompt_eval_count", "eval_count")}
        sess["ps_after_warmup"] = ps_snapshot(client, name)
        ps = sess["ps_after_warmup"] or {}
        if ps.get("size") and ps.get("size_vram") is not None and ps["size_vram"] < ps["size"]:
            log.warning("[%s] модель не целиком в VRAM: %.1f из %.1f ГБ — скорость будет занижена",
                        name, ps["size_vram"] / 1e9, ps["size"] / 1e9)
        meta["sessions"].append(sess)
        models_meta_path.write_text(json.dumps(models_meta, ensure_ascii=False, indent=1), encoding="utf-8")

        for exp, rows in tasks.items():
            out = out_root / exp / f"{slug(name)}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = {r["qid"] for r in read_jsonl(out) if not r.get("error")}
            todo = [t for t in rows if t["qid"] not in done]
            log.info("[%s] %s: выполнено %d, осталось %d", name, exp, len(rows) - len(todo), len(todo))
            t_exp = time.perf_counter()
            for i, t in enumerate(todo, 1):
                rec = {
                    "run_id": args.run_id, "session_id": session_id, "exp": exp, "qid": t["qid"],
                    "source_id": t.get("source_id"), "lang": t.get("lang"),
                    "model": name, "model_digest": digest, "ollama_version": server_version,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "file_path": t["file_path"], "file_hash": t["file_hash"], "hash_verified": False,
                    "doc_whole": t["window"]["doc_whole"], "multi_segment": t["window"]["multi_segment"],
                    "window_segments": t["window"]["segments"], "answer_offset": t["window"]["answer_offset"],
                    "context_chars": t["context_chars"],
                    "prompt_version": t["prompt_version"], "prompt_sha256": t["prompt_sha256"],
                    "prompt_tokens_expected": t["prompt_tokens"].get(name),
                    "question": t["question"], "gold_answer": t["gold_answer"], "gold_line_ids": t["gold_line_ids"],
                    **{k: options[k] for k in ("num_ctx", "num_predict", "temperature", "seed")},
                    "think_sent": think, "error": None,
                }
                if exp.startswith("e2"):
                    rec.update({k: t.get(k) for k in ("answerable", "category", "orig_gold_answer",
                                                      "removed_line_ids", "cut_margin")})
                try:
                    verify_hash(t["file_path"], t["file_hash"])        # защита от изменённого документа
                    rec["hash_verified"] = True
                    messages = prompts.build_messages(t["context"], t["question"])
                    if prompt_sha(messages) != t["prompt_sha256"]:
                        raise RuntimeError("prompt_sha256 mismatch: промпт отличается от замороженного")
                    flat = messages[0]["content"] + "\x00" + messages[1]["content"]
                    shared = max((len(os.path.commonprefix([flat, p])) for p in seen_prompts), default=0)
                    seen_prompts.append(flat)
                    # Ollama переиспользует KV-кэш общего префикса: prefill_tps/ttft для таких запросов
                    # занижены по времени; cache_prefix_frac ~ 0 -> «холодный» запрос
                    rec["cache_prefix_chars"] = shared
                    rec["cache_prefix_frac"] = round(shared / len(flat), 4)
                    res = client.chat(name, messages, options, think, gen["keep_alive"])
                    rec.update(res)
                    rec["ctx_overflow"] = (res["prompt_eval_count"] + res["eval_count"]) >= options["num_ctx"]
                    rec["hit_num_predict"] = res["done_reason"] == "length" or res["eval_count"] >= options["num_predict"]
                    rec["thinking_leak"] = bool(res["thinking_raw"])
                except HashMismatchError as e:
                    rec["error"] = f"hash_mismatch: {e}"
                except KeyboardInterrupt:
                    log.warning("Прервано пользователем; последняя задача не записана")
                    return
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                append_jsonl(out, rec)

                if rec["error"]:
                    log.error("[%s] %s %s ОШИБКА: %s", name, exp, t["qid"], rec["error"][:200])
                else:
                    flags = "".join(f" !{f}" for f in ("ctx_overflow", "hit_num_predict", "thinking_leak") if rec[f])
                    log.info("[%s] %s %d/%d %s | in %d out %d | gen %s t/s | %.1fs | %s%s",
                             name[-24:], exp, i, len(todo), t["qid"], rec["prompt_eval_count"], rec["eval_count"],
                             rec["gen_tps"], rec["wall_s"], rec["answer_raw"][:60].replace("\n", " "), flags)
            log.info("[%s] %s завершён за %.1f мин", name, exp, (time.perf_counter() - t_exp) / 60)

        sess["ps_after_run"] = ps_snapshot(client, name)
        sess["finished_at"] = datetime.now().isoformat(timespec="seconds")
        models_meta_path.write_text(json.dumps(models_meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
