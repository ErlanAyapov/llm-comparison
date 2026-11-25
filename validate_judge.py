"""Сверка судьи-LLM с ручной разметкой.

Вход:  reports/validation_human.jsonl  (item_id, human — исходная метка разметчика,
                                        adjudicated — метка после разбора расхождений)
       reports/validation_key.jsonl    (item_id -> модель, эксперимент, qid, страта)
       results/main/judge/<судья>/<эксп>/<модель>.jsonl
Выход: reports/judge_validation.json и сводка в консоль.

Пример: python validate_judge.py --judge-dir results/main/judge/gpt-oss_120b_j3
"""
import argparse
import collections
import json
from pathlib import Path


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def matrix(pairs):
    labs = sorted({x for p in pairs for x in p})
    return {"labels": labs,
            "rows_human_cols_judge": [[sum(1 for h, j in pairs if h == a and j == b) for b in labs] for a in labs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", default="results/main/judge/gpt-oss_120b_j3")
    ap.add_argument("--human", default="reports/validation_human.jsonl")
    ap.add_argument("--key", default="reports/validation_key.jsonl")
    ap.add_argument("--out", default="reports/judge_validation.json")
    a = ap.parse_args()

    key = {r["item_id"]: r for r in load_jsonl(a.key)}
    judge = {}
    for f in Path(a.judge_dir).glob("*/*.jsonl"):
        for r in load_jsonl(f):
            judge[(r["exp"], r["model"], r["qid"])] = r.get("label")

    rows = []
    for h in load_jsonl(a.human):
        k = key[h["item_id"]]
        j = judge.get((k["exp"], k["model"], k["qid"]))
        if j is None:
            continue
        typ = "A" if k["exp"] in ("e1", "e3") else "B"
        rows.append({**h, "type": typ, "judge": j})

    out = {"judge_dir": a.judge_dir, "n": len(rows), "by_type": {}}
    for typ in ("A", "B"):
        sub = [r for r in rows if r["type"] == typ]
        res = {"n": len(sub)}
        for ref in ("human", "adjudicated"):
            pairs = [(r[ref], r["judge"]) for r in sub]
            po, k = kappa(pairs)
            res[ref] = {"agreement": round(po, 4), "kappa": round(k, 3), **matrix(pairs)}
        res["changed_by_adjudication"] = sum(r["human"] != r["adjudicated"] for r in sub)
        out["by_type"][typ] = res
        print(f"Тип {typ}: n={len(sub)}  исходная разметка: согласие {res['human']['agreement']:.1%}, "
              f"kappa {res['human']['kappa']:.2f}  |  после разбора: согласие {res['adjudicated']['agreement']:.1%}, "
              f"kappa {res['adjudicated']['kappa']:.2f}  (исправлено меток: {res['changed_by_adjudication']})")
        m = res["adjudicated"]
        print("   после разбора, строки — человек, столбцы — судья:", m["labels"])
        for lab, row in zip(m["labels"], m["rows_human_cols_judge"]):
            print(f"   {lab:14}{row}")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
