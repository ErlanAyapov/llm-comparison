#!/usr/bin/env python3
"""Новый E2 — тест на галлюцинации «вырезка» (протокол п. 1, 4.4).

Для вопроса E1 берётся ТО ЖЕ замороженное окно контекста, из которого удалены строки ответа
(line_ids) ± MARGIN строк. Получается пара «E1 (ответ есть) / E2 (ответ вырезан)» на одном тексте.
E2_ru — те же окна с русской формулировкой вопроса (qa_ru, по source_id).

Фильтры (кандидат отбрасывается, если ответ мог «просочиться»):
  * нормализованный эталонный ответ встречается в оставшемся тексте;
  * в тексте есть хотя бы один ключевой токен ответа — число или слово (по основе из 6 символов),
    которого нет в вопросе и которое встречается менее чем в DF_MAX доле документов корпуса.

Отбор: N вопросов, равномерно по документам (round-robin), порядок задаётся seed.

  python build_e2_cut.py --n 200
-> tasks/e2.jsonl, tasks/e2_ru.jsonl, tasks/e2_cut_rejected.jsonl, обновлённый tasks/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import dotenv
import yaml

import prompts
from build_tasks import prompt_sha, write_jsonl
from utils.context_builder import GAP_MARKER
from utils.doc_store import join_lines, load_lines
from utils.ollama_client import OllamaClient

log = logging.getLogger("e2cut")
MARGIN = 2
DF_MAX = 0.5
RELAX_FRAC = 1 / 3
STEM = 6


def toks(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", s.lower().replace("№", " ")).split()


def stem(t: str) -> str:
    return t if any(c.isdigit() for c in t) else t[:STEM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="только статистика и примеры, без сервера и записи")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dotenv.load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen, ctxc = cfg["generation"], cfg["context"]
    limit = gen["num_ctx"] - gen["num_predict"] - ctxc["safety_tokens"]
    tdir = Path(cfg["paths"]["tasks_dir"])
    e1 = [json.loads(l) for l in open(tdir / "e1.jsonl", encoding="utf-8")]
    e3 = {t["source_id"]: t for t in (json.loads(l) for l in open(tdir / "e3.jsonl", encoding="utf-8"))}

    docs = {}
    for t in e1:
        if t["file_path"] not in docs:
            docs[t["file_path"]] = load_lines(t["file_path"], t["file_hash"])
    # документная частота основ
    df = defaultdict(int)
    for lines in docs.values():
        for s in {stem(x) for x in toks(join_lines(lines))}:
            df[s] += 1
    ndocs = len(docs)

    cands, cands_relaxed, rejected = defaultdict(list), defaultdict(list), []
    for t in e1:
        lines = docs[t["file_path"]]
        cut = set()
        for i in t["gold_line_ids"]:
            cut.update(range(i - MARGIN, i + MARGIN + 1))
        segs = []
        for s, e in t["window"]["segments"]:
            cur = None
            for ln in range(s, e + 1):
                if ln in cut:
                    continue
                if cur and ln == cur[1] + 1:
                    cur[1] = ln
                else:
                    cur = [ln, ln]
                    segs.append(cur)
        # между исходными сегментами окна — маркер, внутри (на месте вырезки) — бесшовно
        parts = []
        for s, e in t["window"]["segments"]:
            parts.append(join_lines([lines[ln - 1] for ln in range(s, e + 1) if ln not in cut]))
        text = f"\n{GAP_MARKER}\n".join(p for p in parts if p)

        q_stems = {stem(x) for x in toks(t["question"])}
        key = {stem(x) for x in toks(t["gold_answer"])
               if (any(c.isdigit() for c in x) or len(x) >= 5)}
        key = {k for k in key if k not in q_stems and df[k] / ndocs < DF_MAX}
        text_stems = {stem(x) for x in toks(text)}
        leak = sorted(key & text_stems)
        gold_in = " ".join(toks(t["gold_answer"])) in " ".join(toks(text))
        reason, tier = None, "strict"
        if not key:
            reason = "no_key_tokens"
        elif gold_in:
            reason = "gold_in_text"
        elif leak:
            digit_leak = any(any(c.isdigit() for c in k) for k in leak)
            if not digit_leak and len(key) >= 3 and len(leak) / len(key) <= RELAX_FRAC:
                tier = "relaxed"      # совпали только отдельные общие слова ответа, без чисел
            else:
                reason = "key_token_leak"
        rec = {"source_id": t["qid"], "segments": segs, "text": text, "key_tokens": sorted(key),
               "leak_tokens": leak, "tier": tier}
        if reason:
            rejected.append({"source_id": t["qid"], "reason": reason, "leak": leak[:10], "key": sorted(key)[:10]})
        else:
            (cands if tier == "strict" else cands_relaxed)[t["file_path"]].append(rec)

    rng = random.Random(args.seed)
    chosen = []
    for pool_src in (cands, cands_relaxed):            # сначала строгие, затем — добор из ослабленных
        pools = []
        for fp in sorted(pool_src):
            lst = pool_src[fp][:]
            rng.shuffle(lst)
            pools.append(lst)
        rng.shuffle(pools)
        while len(chosen) < args.n and any(pools):
            for p in pools:
                if p and len(chosen) < args.n:
                    chosen.append(p.pop())
    log.info("Кандидатов: strict %d, relaxed %d из %d; отброшено: %s; выбрано: %d (strict %d)",
             sum(map(len, cands.values())), sum(map(len, cands_relaxed.values())), len(e1),
             {r: sum(1 for x in rejected if x["reason"] == r) for r in sorted({x["reason"] for x in rejected})},
             len(chosen), sum(c["tier"] == "strict" for c in chosen))

    e1_by = {t["qid"]: t for t in e1}
    rows_kk, rows_ru = [], []
    for c in sorted(chosen, key=lambda c: c["source_id"]):
        for lang, src in (("kk", e1_by[c["source_id"]]), ("ru", e3[c["source_id"]])):
            msgs = prompts.build_messages(c["text"], src["question"])
            row = {
                "exp": "e2" if lang == "kk" else "e2_ru",
                "qid": ("cut_" if lang == "kk" else "cut_ru_") + c["source_id"],
                "source_id": c["source_id"], "lang": lang,
                "question": src["question"], "gold_answer": prompts.REFUSAL_PHRASE,
                "orig_gold_answer": src["gold_answer"], "answerable": False,
                "gold_line_ids": [], "removed_line_ids": src["gold_line_ids"], "cut_margin": MARGIN,
                "key_tokens": c["key_tokens"], "filter_tier": c["tier"], "leak_tokens": c["leak_tokens"],
                "file_path": src["file_path"], "file_hash": src["file_hash"],
                "window": {"segments": c["segments"], "doc_whole": False,
                           "multi_segment": len(src["window"]["segments"]) > 1,
                           "answer_offset": src["window"]["answer_offset"],
                           "source_doc_whole": src["window"]["doc_whole"]},
                "context": c["text"], "context_chars": len(c["text"]),
                "prompt_version": prompts.PROMPT_VERSION, "prompt_sha256": prompt_sha(msgs),
                "prompt_tokens": {}, "_msgs": msgs,
            }
            (rows_kk if lang == "kk" else rows_ru).append(row)

    if args.dry_run:
        rng2 = random.Random(1)
        for r in rng2.sample(rows_kk, 6):
            print("\n---", r["qid"], "| Q:", r["question"][:110])
            print("    gold (вырезан):", r["orig_gold_answer"][:110], "| key:", r["key_tokens"][:8])
            print("    tier:", r["filter_tier"], "leak:", r["leak_tokens"])
            print("    окно:", r["window"]["segments"], "chars", r["context_chars"], "removed", r["removed_line_ids"])
        return
    client = OllamaClient(os.getenv("HOST") or cfg["host"])
    for m in cfg["models"]:
        for r in rows_kk + rows_ru:
            n = client.count_tokens(m["name"], r["_msgs"], m["think"])
            if n > limit:
                raise SystemExit(f"{r['qid']} не влезает у {m['name']}: {n}")
            r["prompt_tokens"][m["name"]] = n
        log.info("Токены проверены: %s", m["name"])
    for r in rows_kk + rows_ru:
        del r["_msgs"]

    # архив старого E2 (вопросы не по теме)
    old = tdir / "e2.jsonl"
    if old.exists() and not (tdir / "e2_offtopic.jsonl").exists():
        old.rename(tdir / "e2_offtopic.jsonl")
    write_jsonl(tdir / "e2.jsonl", rows_kk)
    write_jsonl(tdir / "e2_ru.jsonl", rows_ru)
    write_jsonl(tdir / "e2_cut_rejected.jsonl", rejected)

    man_p = tdir / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if "e2" in man["stats"] and "e2_offtopic" not in man["stats"]:
        man["stats"]["e2_offtopic"] = man["stats"].pop("e2")
        man["e2_offtopic_document"] = man.pop("e2_document", None)
    man["e2_cut"] = {"n": len(rows_kk), "margin_lines": MARGIN, "df_max": DF_MAX, "stem_len": STEM,
                     "relax_frac": RELAX_FRAC, "strict": sum(r["filter_tier"] == "strict" for r in rows_kk),
                     "seed": args.seed, "candidates": sum(map(len, cands.values())),
                     "rejected": len(rejected), "documents": len({r["file_path"] for r in rows_kk})}
    for name, rows in (("e2", rows_kk), ("e2_ru", rows_ru)):
        man["stats"][name] = {"n": len(rows), "max_prompt_tokens": {
            m["name"]: max(r["prompt_tokens"][m["name"]] for r in rows) for m in cfg["models"]}}
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Готово: e2=%d, e2_ru=%d, документов %d", len(rows_kk), len(rows_ru), man["e2_cut"]["documents"])


if __name__ == "__main__":
    main()
