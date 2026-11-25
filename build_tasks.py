#!/usr/bin/env python3
"""Сборка замороженных задач E1/E2/E3 (протокол, п. 4).

Шаги:
  1. Проверка хешей всех документов, построчное чтение.
  2. Калибровка: символов на токен для каждого документа у каждой модели (точный подсчёт
     через Ollama, truncate=false). Берётся худшая (наименьшая) — один текст для всех моделей.
  3. Окна контекста вокруг line_ids (seed, qid). E3 использует окна E1 (по source_id).
  4. Точная проверка: каждый итоговый промпт у каждой модели <= num_ctx - num_predict - safety.
     Не влезло -> окно ужимается, повторная проверка у всех моделей.
  5. Выбор документа для E2 (если не задан в config.yaml).
  6. tasks/e1.jsonl, e2_offtopic.jsonl (прежний E2, архив), e3.jsonl, manifest, calibration. Новый E2 — build_e2_cut.py.

Запуск:  python build_tasks.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dotenv
import yaml

import prompts
from utils.context_builder import build_window
from utils.doc_store import join_lines, load_lines
from utils.ollama_client import OllamaClient
from utils.tools import file_hash

log = logging.getLogger("build_tasks")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def prompt_sha(messages: list[dict]) -> str:
    return sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
def calibrate(client: OllamaClient, models: list[dict], docs: dict[str, list[str]], cache_path: Path) -> dict:
    """{file_path: {model: chars_per_token}}, + overhead токенов пустого промпта по моделям."""
    cache = load_json(cache_path) if cache_path.exists() else {"ratios": {}, "overhead": {}}
    for m in models:
        name, think = m["name"], m["think"]
        if name not in cache["overhead"]:
            cache["overhead"][name] = client.count_tokens(name, prompts.build_messages("", ""), think)
        ov = cache["overhead"][name]
        for fp, lines in docs.items():
            r = cache["ratios"].setdefault(fp, {})
            if name in r:
                continue
            text = join_lines(lines)
            n = client.count_tokens(name, prompts.build_messages(text, ""), think) - ov
            r[name] = round(len(text) / max(n, 1), 4)
            log.info("calib %-45s %-40s %6d chars %6d tok  %.3f c/t", name[-45:], Path(fp).name[-40:], len(text), n, r[name])
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache


def verify_tokens(client, models, items: dict[str, dict], limit: int) -> set[str]:
    """Считает токены у всех моделей для items[key]['messages_list']; возвращает ключи, не влезшие хоть у одной."""
    failed = set()
    for m in models:
        name, think = m["name"], m["think"]
        for key, it in items.items():
            toks = [client.count_tokens(name, msgs, think) for msgs in it["messages_list"]]
            it["prompt_tokens"][name] = max(toks)
            if max(toks) > limit:
                failed.add(key)
        log.info("verify %-50s: %d/%d не влезли", name[-50:], sum(1 for k in items if items[k]["prompt_tokens"].get(name, 0) > limit), len(items))
    return failed


def update_config_e2(cfg_path: Path, fp: str, fh: str):
    txt = cfg_path.read_text(encoding="utf-8")
    txt = re.sub(r"(e2_document:\s*\n\s*file_path:)\s*.*\n(\s*file_hash:)\s*.*\n",
                 lambda mm: f"{mm.group(1)} \"{fp}\"\n{mm.group(2)} {fh}\n", txt, count=1)
    cfg_path.write_text(txt, encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", help="подмножество моделей (по умолчанию все из config)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dotenv.load_dotenv()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    host = os.getenv("HOST") or cfg["host"]
    gen, ctx = cfg["generation"], cfg["context"]
    models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
    limit = gen["num_ctx"] - gen["num_predict"] - ctx["safety_tokens"]

    client = OllamaClient(host)
    tags = client.tags()
    for m in models:
        if m["name"] not in tags:
            raise SystemExit(f"Модель не найдена на сервере: {m['name']}")

    tasks_dir = Path(cfg["paths"]["tasks_dir"]); tasks_dir.mkdir(exist_ok=True)
    e1 = load_json(cfg["datasets"]["e1"]); e2 = load_json(cfg["datasets"]["e2"]); e3 = load_json(cfg["datasets"]["e3"])

    # 1. документы
    doc_hash = {}
    for q in e1:
        doc_hash.setdefault(q["file_path"], q["file_hash"])
        if doc_hash[q["file_path"]] != q["file_hash"]:
            raise SystemExit(f"Разные file_hash для одного документа: {q['file_path']}")
    for q in e3:
        if doc_hash.get(q["file_path"]) != q["file_hash"]:
            raise SystemExit(f"E3 {q['id']}: file_hash не совпадает с E1")
    docs = {fp: load_lines(fp, h) for fp, h in doc_hash.items()}
    log.info("Документов: %d, хеши совпадают", len(docs))

    # 2. калибровка
    calib = calibrate(client, models, docs, tasks_dir / "calibration.json")
    names = [m["name"] for m in models]
    ratio = {fp: min(calib["ratios"][fp][n] for n in names) for fp in docs}
    overhead = max(calib["overhead"][n] for n in names)
    min_ratio_global = min(ratio.values())

    def q_tokens_est(*questions):
        return max(int(len(q) / min_ratio_global) + 8 for q in questions)

    # 3. окна E1/E3
    e3_by_src = {q["source_id"]: q for q in e3}
    if set(e3_by_src) != {q["id"] for q in e1}:
        raise SystemExit("E3 source_id не совпадают с id E1")

    items = {}
    for q in e1:
        ru = e3_by_src[q["id"]]
        budget_tok = limit - overhead - q_tokens_est(q["question"], ru["question"])
        items[q["id"]] = {"q": q, "ru": ru, "budget": int(budget_tok * ratio[q["file_path"]]),
                          "prompt_tokens": {}, "shrinks": 0}

    def build(key):
        it = items[key]; q = it["q"]
        w = build_window(docs[q["file_path"]], q["line_ids"], it["budget"], ctx["seed"], q["id"])
        it["window"] = w
        it["messages_list"] = [prompts.build_messages(w.text, q["question"]),
                               prompts.build_messages(w.text, it["ru"]["question"])]
        it["prompt_tokens"] = {}

    for k in items:
        build(k)

    pending = set(items)
    rnd = 0
    while pending:
        rnd += 1
        log.info("Проверка токенов E1/E3, раунд %d: %d задач", rnd, len(pending))
        failed = verify_tokens(client, models, {k: items[k] for k in pending}, limit)
        for k in failed:
            it = items[k]
            if it["window"].span_exceeds_budget or it["shrinks"] > 30:
                raise SystemExit(f"{k}: фрагмент с ответом не влезает в контекст")
            if it["window"].doc_whole:
                it["budget"] = min(it["budget"], len(it["window"].text))
            it["budget"] = int(it["budget"] * ctx["shrink_factor"])
            it["shrinks"] += 1
            build(k)
        pending = failed

    # 4. документ для E2
    e2_fp, e2_fh = cfg["e2_document"]["file_path"], cfg["e2_document"]["file_hash"]
    if not e2_fp:
        kw = {k.lower() for q in e2 for k in q.get("absent_keywords", [])}
        longest_q = max((q["question"] for q in e2), key=len)
        cands = []
        for fp, lines in docs.items():
            text = join_lines(lines)
            low = text.lower()
            hits = [k for k in kw if k in low]
            fits = len(text) <= (limit - overhead - q_tokens_est(longest_q)) * ratio[fp]
            log.info("E2 кандидат %-45s chars=%6d fits=%s hits=%s", Path(fp).name[-45:], len(text), fits, hits[:5])
            if fits and not hits:
                cands.append((len(text), fp))
        if not cands:
            raise SystemExit("Нет подходящего документа для E2")
        e2_fp = max(cands)[1]; e2_fh = doc_hash[e2_fp]
        update_config_e2(cfg_path, e2_fp, e2_fh)
        log.info("E2: выбран %s (записан в %s)", e2_fp, cfg_path)
    e2_lines = load_lines(e2_fp, e2_fh)
    e2_text = join_lines(e2_lines)
    e2_items = {q["id"]: {"q": q, "messages_list": [prompts.build_messages(e2_text, q["question"])],
                          "prompt_tokens": {}} for q in e2}
    failed = verify_tokens(client, models, e2_items, limit)
    if failed:
        raise SystemExit(f"E2: документ не влезает для {sorted(failed)[:5]}...")

    # 5. запись
    def base(exp, q, fp, fh, w, msgs, ptoks):
        return {
            "exp": exp, "qid": q["id"], "source_id": q.get("source_id"), "lang": q.get("lang"),
            "question": q["question"], "gold_answer": q["answer"], "gold_line_ids": q.get("line_ids", []),
            "file_path": fp, "file_hash": fh,
            "window": {"segments": w["segments"], "doc_whole": w["doc_whole"],
                       "multi_segment": w["multi_segment"], "answer_offset": w.get("answer_offset")},
            "context": w["text"],
            "context_chars": len(w["text"]),
            "prompt_version": prompts.PROMPT_VERSION, "prompt_sha256": prompt_sha(msgs),
            "prompt_tokens": ptoks,
        }

    rows1, rows3 = [], []
    for k, it in items.items():
        w = it["window"]
        wd = {"segments": w.segments, "doc_whole": w.doc_whole, "multi_segment": w.multi_segment,
              "answer_offset": w.answer_offset, "text": w.text}
        q, ru = it["q"], it["ru"]
        # prompt_tokens у items — максимум по kk/ru; пересчитывать отдельно не нужно для контроля лимита
        rows1.append(base("e1", q, q["file_path"], q["file_hash"], wd, it["messages_list"][0], it["prompt_tokens"]))
        r3 = base("e3", ru, ru["file_path"], ru["file_hash"], wd, it["messages_list"][1], it["prompt_tokens"])
        r3["doc_lang"] = ru.get("doc_lang")
        rows3.append(r3)
    rows2 = []
    for k, it in e2_items.items():
        q = it["q"]
        wd = {"segments": [(1, len(e2_lines))], "doc_whole": True, "multi_segment": False,
              "answer_offset": None, "text": e2_text}
        r = base("e2", q, e2_fp, e2_fh, wd, it["messages_list"][0], it["prompt_tokens"])
        r.update({"answerable": q.get("answerable", False), "category": q.get("category"),
                  "absent_keywords": q.get("absent_keywords", [])})
        rows2.append(r)

    write_jsonl(tasks_dir / "e1.jsonl", rows1)
    write_jsonl(tasks_dir / "e2_offtopic.jsonl", rows2)
    write_jsonl(tasks_dir / "e3.jsonl", rows3)

    def stats(rows):
        mx = defaultdict(int)
        for r in rows:
            for n, t in r["prompt_tokens"].items():
                mx[n] = max(mx[n], t)
        return {"n": len(rows), "doc_whole": sum(r["window"]["doc_whole"] for r in rows),
                "multi_segment": sum(r["window"]["multi_segment"] for r in rows),
                "max_prompt_tokens": dict(mx)}

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ollama_host": host, "ollama_version": client.version(),
        "models": {m["name"]: {"think": m["think"], "digest": tags[m["name"]]["digest"]} for m in models},
        "generation": gen, "context": ctx, "prompt_token_limit": limit,
        "prompt_version": prompts.PROMPT_VERSION, "system_prompt": prompts.SYSTEM_PROMPT,
        "user_template": prompts.USER_TEMPLATE, "refusal_phrase": prompts.REFUSAL_PHRASE,
        "datasets": {k: {"path": v, "sha256": file_hash(v)} for k, v in cfg["datasets"].items()},
        "e2_offtopic_document": {"file_path": e2_fp, "file_hash": e2_fh, "chars": len(e2_text)},
        "chars_per_token_min_by_doc": ratio, "overhead_tokens": calib["overhead"],
        "verify_rounds_e1e3": rnd,
        "stats": {"e1": stats(rows1), "e2_offtopic": stats(rows2), "e3": stats(rows3)},
    }
    (tasks_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Готово: %s", json.dumps(manifest["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
