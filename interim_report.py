#!/usr/bin/env python3
"""Промежуточный отчёт по прогону: полнота и контроль данных, скорость, ПРЕДВАРИТЕЛЬНЫЕ
лексические показатели. Это не финальная оценка (evaluate.py): лексическое совпадение
занижает верные перефразировки, особенно на казахском (морфология) и в E3.

  python interim_report.py --run-id main
-> reports/interim_<run_id>.xlsx, reports/interim_<run_id>.md
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

REFUSAL_RE = re.compile(
    r"(жауап|ақпарат|мәлімет|дерек)\w*\s+(жоқ|берілмейді|берілмеген)|көрсетілмеген|айтылмаған|"
    r"құжатта\b[^.]{0,120}\sжоқ\b|"
    r"нет\s+(ответа|информации|сведений|данных)|не\s+(содержит|указан|упоминается)|отсутству|"
    r"в\s+документе\b[^.]{0,120}\s(нет|не\s+(указан|приведен|содерж|упомина))",
    re.IGNORECASE)


def is_refusal(ans: str) -> bool:
    """Эвристика: формулировка отказа есть И (ответ короткий ИЛИ формулировка в конце ответа).
    Отсекает случаи вроде «Тізбеде көрсетілмеген адамдарға ...» внутри содержательного ответа."""
    a = ans.strip()
    last = None
    for last in REFUSAL_RE.finditer(a):
        pass
    return last is not None and (len(a) <= 150 or last.end() >= len(a) - 60)
EXPS = ("e1", "e2", "e2_ru", "e3", "e2_offtopic")
KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYR = re.compile(r"[а-яёА-ЯЁ]")


def norm_tokens(s: str) -> list[str]:
    s = s.lower().replace("№", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def contains(ans: str, gold: str) -> bool:
    a, g = " ".join(norm_tokens(ans)), " ".join(norm_tokens(gold))
    return bool(g) and g in a


def recall_f1(ans: str, gold: str) -> tuple[float, float]:
    a, g = norm_tokens(ans), norm_tokens(gold)
    if not a or not g:
        return 0.0, 0.0
    ca = defaultdict(int)
    for t in a:
        ca[t] += 1
    common = 0
    for t in g:
        if ca[t] > 0:
            common += 1
            ca[t] -= 1
    if not common:
        return 0.0, 0.0
    p, r = common / len(a), common / len(g)
    return r, 2 * p * r / (p + r)


def lang_of(ans: str) -> str:
    if not CYR.search(ans):
        return "other"
    return "kk" if any(c in KK_LETTERS for c in ans) else "ru"


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["results_dir"]) / args.run_id
    models = [m["name"] for m in cfg["models"]]
    kk_gold = {q["id"]: q["answer"] for q in json.load(open(cfg["datasets"]["e1"], encoding="utf-8"))}

    recs = {}
    for f in glob.glob(str(root / "e*" / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            recs[(r["model"], r["exp"], r["qid"])] = r          # последняя запись побеждает

    expected = {e: sum(1 for _ in open(Path(cfg["paths"]["tasks_dir"]) / f"{e}.jsonl", encoding="utf-8"))
                for e in EXPS if (Path(cfg["paths"]["tasks_dir"]) / f"{e}.jsonl").exists()}

    # ---------------- по ответу
    for r in recs.values():
        ans = r.get("answer_raw") or ""
        r["_refusal"] = is_refusal(ans)
        r["_lang"] = lang_of(ans)
        if r["exp"] in ("e1", "e3"):
            golds = [r["gold_answer"]] + ([kk_gold[r["source_id"]]] if r["exp"] == "e3" else [])
            r["_contains"] = any(contains(ans, g) for g in golds)
            rf = [recall_f1(ans, g) for g in golds]
            r["_recall"] = max(x[0] for x in rf)
            r["_f1"] = max(x[1] for x in rf)

    # ---------------- сводка
    summary, control, speed = [], [], []
    for m in models:
        for e in EXPS:
            R = [r for (mm, ee, _), r in recs.items() if mm == m and ee == e]
            ok = [r for r in R if not r.get("error")]
            if not R:
                continue
            row = {"model": m, "exp": e, "n": len(ok), "expected": expected[e]}
            if e.startswith("e2"):
                row["refusal_rate_%"] = round(100 * sum(r["_refusal"] for r in ok) / len(ok), 1)
                if e in ("e2", "e2_ru"):
                    # ответ совпал с вырезанным эталоном -> утечка ответа в контекст или «знание» модели
                    row["orig_gold_in_answer_%"] = round(100 * sum(
                        contains(r["answer_raw"] or "", r.get("orig_gold_answer") or "") for r in ok) / len(ok), 1)
                row["exact_refusal_phrase_%"] = round(
                    100 * sum((r["answer_raw"] or "").strip() == "Құжатта жауап жоқ." for r in ok) / len(ok), 1)
            else:
                row["contains_gold_%"] = round(100 * sum(r["_contains"] for r in ok) / len(ok), 1)
                row["token_recall_mean"] = round(st.mean(r["_recall"] for r in ok), 3)
                row["token_f1_mean"] = round(st.mean(r["_f1"] for r in ok), 3)
                row["false_refusal_%"] = round(100 * sum(r["_refusal"] for r in ok) / len(ok), 1)
            if e in ("e3", "e2_ru"):
                row["answered_in_ru_%"] = round(100 * sum(r["_lang"] == "ru" for r in ok) / len(ok), 1)
                row["answered_in_kk_%"] = round(100 * sum(r["_lang"] == "kk" for r in ok) / len(ok), 1)
            row["median_output_tokens"] = med([r["eval_count"] for r in ok])
            summary.append(row)

            control.append({
                "model": m, "exp": e, "records": len(R), "expected": expected[e],
                "errors": sum(bool(r.get("error")) for r in R),
                "hash_verified": sum(bool(r.get("hash_verified")) for r in R),
                "ctx_overflow": sum(bool(r.get("ctx_overflow")) for r in ok),
                "hit_num_predict": sum(bool(r.get("hit_num_predict")) for r in ok),
                "thinking_leak": sum(bool(r.get("thinking_leak")) for r in ok),
                "empty_answers": sum(not (r.get("answer_raw") or "").strip() for r in ok),
            })
            cold = [r for r in ok if (r.get("cache_prefix_frac") or 0) < 0.1]
            speed.append({
                "model": m, "exp": e,
                "gen_tps_median": med([r["gen_tps"] for r in ok]),
                "prompt_tokens_median": med([r["prompt_eval_count"] for r in ok]),
                "wall_s_median": med([r["wall_s"] for r in ok]),
                "wall_total_h": round(sum(r["wall_s"] for r in ok) / 3600, 2),
                "cold_requests": len(cold),
                "prefill_tps_median_cold": med([r["prefill_tps"] for r in cold]),
                "ttft_s_median_cold": med([r["ttft_s"] for r in cold]),
            })

    models_meta = json.loads((root / "models.json").read_text(encoding="utf-8"))
    run_info = []
    for m, meta in models_meta.items():
        for s in meta["sessions"]:
            ps = s.get("ps_after_warmup") or {}
            run_info.append({"model": m, "digest": meta["digest"][:12], "session": s["session_id"],
                             "started_at": s.get("started_at"), "finished_at": s.get("finished_at"),
                             "size_GB": round((ps.get("size") or 0) / 1e9, 1),
                             "size_vram_GB": round((ps.get("size_vram") or 0) / 1e9, 1),
                             "context_length": ps.get("context_length"),
                             "load_s": round((s.get("warmup") or {}).get("load_duration", 0) / 1e9, 1),
                             "think_sent": meta.get("think_sent"),
                             "quantization": (meta.get("details") or {}).get("quantization_level")})

    # ---------------- ответы (широкий формат)
    answers = {}
    for e in expected:
        tasks = [json.loads(l) for l in open(Path(cfg["paths"]["tasks_dir"]) / f"{e}.jsonl", encoding="utf-8")]
        rows = []
        for t in tasks:
            row = {"qid": t["qid"], "question": t["question"], "gold_answer": t["gold_answer"],
                   "doc_whole": t["window"]["doc_whole"]}
            for m in models:
                r = recs.get((m, e, t["qid"]))
                row[m] = r["answer_raw"] if r else None
            rows.append(row)
        answers[e] = rows

    out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
    import pandas as pd
    xlsx = out_dir / f"interim_{args.run_id}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame(control).to_excel(w, sheet_name="control", index=False)
        pd.DataFrame(speed).to_excel(w, sheet_name="speed", index=False)
        pd.DataFrame(run_info).to_excel(w, sheet_name="run_info", index=False)
        for e, rows in answers.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=f"answers_{e}", index=False)
    (out_dir / f"interim_{args.run_id}.json").write_text(
        json.dumps({"summary": summary, "control": control, "speed": speed, "run_info": run_info},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"summary": summary, "speed": speed}, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
