#!/usr/bin/env python3
"""Выборка ответов для ручной разметки — проверка судьи-LLM (протокол п. 10.2).

Стратифицированная выборка (seed 42), только ответы, которые реально оценивает судья
(точная фраза отказа размечается правилом и в выборку не входит):
  * E1/E3: N_ANS ответов поровну по 8 моделям основного сравнения; внутри — поровну
    «лексически совпало с эталоном» / «не совпало» / «эвристика: отказ».
  * E2/E2_ru: N_NEG ответов поровну по моделям; внутри — «эвристика: отказ» / «не отказ».
Модель в листе для разметчика скрыта (ключ — в validation_key.jsonl).

  python make_validation_sample.py            # выборка (нужен pymupdf)
  python make_validation_sample.py --xlsx     # лист для разметчика (нужен openpyxl)
-> reports/validation_sample.jsonl, reports/validation_key.jsonl, reports/validation_sheet.xlsx
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

N_ANS, N_NEG = 120, 80
OUT = Path("reports")


def slug(s):
    return re.sub(r"[^\w.-]+", "_", s)


def build(cfg, run_id):
    import prompts
    from interim_report import contains, is_refusal
    from utils.doc_store import load_lines
    rng = random.Random(42)
    root = Path(cfg["paths"]["results_dir"]) / run_id
    tdir = Path(cfg["paths"]["tasks_dir"])
    models = [m["name"] for m in cfg["models"] if not m.get("reference")]
    e1 = {t["qid"]: t for t in map(json.loads, open(tdir / "e1.jsonl", encoding="utf-8"))}
    picked = []
    for group, exps, n in (("ans", ("e1", "e3"), N_ANS), ("neg", ("e2", "e2_ru"), N_NEG)):
        per_model = n // len(models)
        for model in models:
            strata = defaultdict(list)
            for exp in exps:
                tasks = {t["qid"]: t for t in map(json.loads, open(tdir / f"{exp}.jsonl", encoding="utf-8"))}
                for r in map(json.loads, open(root / exp / f"{slug(model)}.jsonl", encoding="utf-8")):
                    ans = (r["answer_raw"] or "").strip()
                    if r.get("error") or ans == prompts.REFUSAL_PHRASE:
                        continue
                    t = tasks[r["qid"]]
                    if group == "ans":
                        golds = [t["gold_answer"]] + ([e1[t["source_id"]]["gold_answer"]] if exp == "e3" else [])
                        s = "refusal" if is_refusal(ans) else ("lex_match" if any(contains(ans, g) for g in golds) else "no_match")
                    else:
                        s = "refusal" if is_refusal(ans) else "answer"
                    strata[s].append((exp, t, r))
            keys = sorted(strata)
            for k in keys:
                rng.shuffle(strata[k])
            got = 0
            while got < per_model and any(strata.values()):
                for k in keys:
                    if strata[k] and got < per_model:
                        exp, t, r = strata[k].pop()
                        lines = load_lines(t["file_path"], t["file_hash"])
                        item = {"model": model, "exp": exp, "qid": r["qid"], "stratum": k,
                                "question": t["question"], "answer": r["answer_raw"]}
                        if group == "ans":
                            item["reference"] = t["gold_answer"] + (
                                f"  |  kk: {e1[t['source_id']]['gold_answer']}" if exp == "e3" else "")
                            item["evidence"] = "\n".join(lines[i - 1] for i in t["gold_line_ids"])
                        else:
                            item["reference"] = "(вырезано) " + t["orig_gold_answer"]
                            item["evidence"] = t["context"]
                        picked.append(item)
                        got += 1
    rng.shuffle(picked)
    for i, p in enumerate(picked, 1):
        p["item_id"] = f"V{i:03d}"
    OUT.mkdir(exist_ok=True)
    with open(OUT / "validation_sample.jsonl", "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(OUT / "validation_key.jsonl", "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps({k: p[k] for k in ("item_id", "model", "exp", "qid", "stratum")}, ensure_ascii=False) + "\n")
    print(f"выборка: {len(picked)} (ans {sum(p['exp'] in ('e1','e3') for p in picked)}, "
          f"neg {sum(p['exp'] in ('e2','e2_ru') for p in picked)})")


def xlsx():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    items = [json.loads(l) for l in open(OUT / "validation_sample.jsonl", encoding="utf-8")]
    wb = Workbook()
    ins = wb.active
    ins.title = "инструкция"
    for row in [
        ["Ручная разметка ответов моделей — проверка судьи-LLM"],
        [""],
        ["Тип A — ответ в документе ЕСТЬ (тип = 'A: ответ есть'). Метки:"],
        ["correct", "передан ключевой смысл эталона; язык, перефраз и лишние верные детали не важны"],
        ["partial", "часть ключевой информации верна, но неполно или с небольшой ошибкой"],
        ["incorrect", "неверно, противоречит эталону/документу, не по вопросу или слишком расплывчато"],
        ["refusal", "модель говорит, что ответа в документе нет, или отказывается"],
        [""],
        ["Тип B — фрагмент с ответом ВЫРЕЗАН (тип = 'B: ответ вырезан'). В колонке «документ» — оставшийся текст. Метки:"],
        ["refusal", "модель сообщает, что ответа нет (оговорки без новых фактов допустимы)"],
        ["supported", "ответ действительно написан в ОСТАВШЕМСЯ тексте (впишите цитату в комментарий)"],
        ["hallucination", "модель утверждает ответ, которого в оставшемся тексте нет (догадка, внешние знания, даже с оговорками)"],
        [""],
        ["Модель, давшая ответ, скрыта. Размечайте независимо, не советуясь. Колонка «метка» — выпадающий список."],
    ]:
        ins.append(row)
    ins.column_dimensions["A"].width = 16
    ins.column_dimensions["B"].width = 110
    ws = wb.create_sheet("разметка")
    head = ["item_id", "тип", "вопрос", "эталон", "документ (фрагмент / оставшийся текст)", "ответ модели", "метка", "комментарий"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
    for it in items:
        ws.append([it["item_id"], "A: ответ есть" if it["exp"] in ("e1", "e3") else "B: ответ вырезан",
                   it["question"], it["reference"], it["evidence"][:32000], it["answer"], "", ""])
    for col, w in zip("ABCDEFGH", (8, 14, 40, 35, 70, 50, 14, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"correct,partial,incorrect,refusal,supported,hallucination"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"G2:G{len(items) + 1}")
    ws.freeze_panes = "C2"
    wb.save(OUT / "validation_sheet.xlsx")
    print("xlsx:", OUT / "validation_sheet.xlsx")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--xlsx", action="store_true")
    a = ap.parse_args()
    if a.xlsx:
        xlsx()
    else:
        build(yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")), a.run_id)
