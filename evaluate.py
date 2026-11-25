#!/usr/bin/env python3
"""Финальная оценка ответов судьёй-LLM (протокол п. 10).

E1 / E3 (ответ в документе есть) — метки:
  correct    — передан ключевой смысл эталона (любой язык, перефраз допустим; лишние верные детали не штрафуются)
  partial    — часть ключевой информации верна, но неполно или с небольшой ошибкой
  incorrect  — неверно, противоречит эталону/документу, или не по вопросу
  refusal    — модель утверждает, что ответа в документе нет, или отказывается
E2 / E2_ru (фрагмент с ответом вырезан) — метки:
  refusal        — модель сообщает, что ответа нет (допускаются оговорки без новых фактов)
  supported      — ответ действительно следует из ОСТАВШЕГОСЯ текста (судья приводит цитату)
  hallucination  — ответ утверждает факты, которых в оставшемся тексте нет (в т.ч. с оговорками)
Детерминированное правило (без судьи): ответ, совпадающий с фразой отказа, -> refusal.

Судья не видит, какая модель ответила. Каждая запись хранит сырой ответ судьи, рассуждение,
sha промпта судьи и JUDGE_VERSION. Возобновляемо: готовые записи пропускаются.

  python evaluate.py --judge gpt-oss:120b --think low                # всё
  python evaluate.py --exp e2 --models qwen3:8b --limit 20          # частично
  python evaluate.py --sample reports/validation_sample.jsonl       # только выборка для валидации
-> results/<run>/judge/<judge>/<exp>/<model>.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import dotenv
import yaml

import prompts
from utils.doc_store import load_lines
from utils.ollama_client import OllamaClient

log = logging.getLogger("evaluate")
JUDGE_VERSION = "j3"   # j1->j2 (18.11): обоснование до метки, уточнено supported (B); j2->j3: в A отказ — отдельное поле claims_absent
EXPS = ("e1", "e3", "e2", "e2_ru")

SYSTEM = ("You are a meticulous evaluator of question answering over Kazakh legal documents. "
          "Documents are in Kazakh; questions are in Kazakh or Russian; answers may be in Kazakh, Russian or mixed. "
          "Judge meaning, not wording or language. Reply with a single JSON object only.")

ANSWERABLE = """Question: {question}

Reference answer: {gold}
{gold_alt}
Supporting excerpt from the document (where the reference answer comes from):
<<<
{evidence}
>>>

Model answer:
<<<
{answer}
>>>

Answer two separate questions about the model answer.
1) "claims_absent": does the model answer SAY that the document does not contain the requested information (fully, or the key part asked), or decline to answer? Report only what the model says, regardless of whether that claim is true. true / false.
2) "label" — how well the model answer conveys the key information of the reference answer:
- "correct": conveys the key information (paraphrase, other language or extra details that do not contradict the document are fine).
- "partial": contains part of the key information but is incomplete, or mixes it with a minor error.
- "incorrect": the key information is missing, wrong, contradicts the reference/document, answers a different question, or is too vague to be useful.
Return JSON: {{"reason": "<one short sentence in English>", "claims_absent": true/false, "label": "..."}}"""

UNANSWERABLE = """The fragment of the document that answered the question below was REMOVED. The remaining document text is:
<<<
{context}
>>>

Question: {question}

For your information, the removed answer was: {orig_gold}

Model answer:
<<<
{answer}
>>>

First decide: does the model assert a specific answer to the question, or does it say the information is absent?
- "refusal": the model says the answer (or its key part) is not in the document and asserts no specific answer. Remarks or related quotes from the document do not change this. A correct statement that the information is absent is ALWAYS "refusal".
- "supported": the model asserts a specific answer AND that answer is explicitly stated in the REMAINING text above. Quote the exact supporting sentence.
- "hallucination": the model asserts an answer (even hedged) that is NOT stated in the remaining text — invented facts, outside knowledge, or a guess, including a guess that matches the removed answer.
Return JSON: {{"reason": "<one short sentence in English>", "evidence": "<exact quote from the remaining text supporting the asserted answer, or empty>", "label": "..."}}"""

LABELS = {"ans": ["correct", "partial", "incorrect", "refusal"],   # итоговые метки
          "ans_judge": ["correct", "partial", "incorrect"],             # что выбирает судья в A
          "neg": ["refusal", "supported", "hallucination"]}


def schema(kind):
    # порядок полей = порядок генерации: сначала обоснование, потом метка (j2)
    props = {"reason": {"type": "string"}}
    if kind == "neg":
        props["evidence"] = {"type": "string"}
        props["label"] = {"type": "string", "enum": LABELS["neg"]}
    else:
        props["claims_absent"] = {"type": "boolean"}
        props["label"] = {"type": "string", "enum": LABELS["ans_judge"]}
    return {"type": "object", "properties": props, "required": list(props)}


def slug(s):
    return re.sub(r"[^\w.-]+", "_", s)


def sha(msgs):
    return hashlib.sha256(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build(exp, task, answer, kk_task=None):
    if exp in ("e1", "e3"):
        lines = load_lines(task["file_path"], task["file_hash"])
        ev = "\n".join(lines[i - 1] for i in task["gold_line_ids"] if 0 < i <= len(lines))
        alt = f"Reference answer (Kazakh original): {kk_task['gold_answer']}\n" if kk_task else ""
        user = ANSWERABLE.format(question=task["question"], gold=task["gold_answer"], gold_alt=alt,
                                 evidence=ev, answer=answer)
        kind = "ans"
    else:
        user = UNANSWERABLE.format(context=task["context"], question=task["question"],
                                   orig_gold=task["orig_gold_answer"], answer=answer)
        kind = "neg"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], kind


def parse(raw, kind):
    try:
        d = json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else {}
    lab = d.get("label")
    if kind == "ans":
        # детерминированно (j3): модель заявила «в документе нет» и ключевой информации не дала -> refusal
        if lab not in LABELS["ans_judge"] or not isinstance(d.get("claims_absent"), bool):
            return None, d
        return ("refusal" if d["claims_absent"] and lab == "incorrect" else lab), d
    return (lab if lab in LABELS[kind] else None), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--judge", default="gpt-oss:120b")
    ap.add_argument("--think", default="low")
    ap.add_argument("--exp", nargs="*", default=list(EXPS))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", help="jsonl с полями model, exp, qid — оценивать только их")
    ap.add_argument("--tag", default="", help="суффикс каталога (напр. для повторной оценки устойчивости)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dotenv.load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    client = OllamaClient(os.getenv("HOST") or cfg["host"], timeout=cfg["generation"]["timeout_s"])
    root = Path(cfg["paths"]["results_dir"]) / args.run_id
    tdir = Path(cfg["paths"]["tasks_dir"])
    jdir = root / "judge" / (slug(args.judge) + (f"_{args.tag}" if args.tag else ""))
    models = [m["name"] for m in cfg["models"] if not args.models or m["name"] in args.models]
    want = None
    if args.sample:
        want = {(r["model"], r["exp"], r["qid"]) for r in map(json.loads, open(args.sample, encoding="utf-8"))}
    opts = {"num_ctx": 16384, "num_predict": 4096, "temperature": 0, "seed": 42}
    e1 = {t["qid"]: t for t in map(json.loads, open(tdir / "e1.jsonl", encoding="utf-8"))}
    jdigest = client.tags()[args.judge]["digest"]

    for exp in args.exp:
        tasks = {t["qid"]: t for t in map(json.loads, open(tdir / f"{exp}.jsonl", encoding="utf-8"))}
        for model in models:
            src = root / exp / f"{slug(model)}.jsonl"
            if not src.exists():
                continue
            out = jdir / exp / f"{slug(model)}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = set()
            if out.exists():
                done = {r["qid"] for r in map(json.loads, open(out, encoding="utf-8")) if r.get("label")}
            recs = [r for r in map(json.loads, open(src, encoding="utf-8")) if not r.get("error")]
            todo = [r for r in recs if r["qid"] not in done
                    and (want is None or (model, exp, r["qid"]) in want)]
            if args.limit:
                todo = todo[:args.limit]
            if not todo:
                continue
            log.info("%s / %s: к оценке %d", exp, model, len(todo))
            t0 = time.time()
            with open(out, "a", encoding="utf-8") as f:
                for i, r in enumerate(todo, 1):
                    task, ans = tasks[r["qid"]], (r["answer_raw"] or "").strip()
                    rec = {"run_id": args.run_id, "exp": exp, "model": model, "qid": r["qid"],
                           "judge": args.judge, "judge_digest": jdigest, "judge_think": args.think,
                           "judge_version": JUDGE_VERSION, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    if ans == prompts.REFUSAL_PHRASE:
                        rec.update({"label": "refusal", "rule": "exact_refusal_phrase"})
                    else:
                        msgs, kind = build(exp, task, ans, e1.get(task.get("source_id")) if exp == "e3" else None)
                        rec["judge_prompt_sha256"] = sha(msgs)
                        # повторы: другой seed (при temperature 0 сбой иначе воспроизводится), затем
                        # более высокий уровень рассуждения (редкое зацикливание рассуждения на low)
                        plan = [(args.think, 0), (args.think, 1), (args.think, 2), ("medium", 0), ("high", 0)]
                        for attempt, (th, ds) in enumerate(plan):
                            try:
                                res = client.chat(args.judge, msgs, {**opts, "seed": opts["seed"] + ds}, th, "30m", fmt=schema(kind))
                                label, parsed = parse(res["answer_raw"], kind)
                                rec.update({"label": label, "judge_json": parsed, "judge_raw": res["answer_raw"],
                                            "judge_thinking": res["thinking_raw"],
                                            "judge_done_reason": res["done_reason"],
                                            "judge_eval_count": res["eval_count"], "judge_wall_s": res["wall_s"],
                                            "attempt": attempt, "judge_seed": opts["seed"] + ds, "judge_think": th})
                                if label:
                                    break
                            except Exception as ex:          # noqa: BLE001
                                rec.update({"label": None, "error": str(ex)[:300], "attempt": attempt})
                                time.sleep(5)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    if i % 50 == 0 or i == len(todo):
                        log.info("  %s / %s %d/%d (%.1f с/ответ)", exp, model, i, len(todo), (time.time() - t0) / i)


if __name__ == "__main__":
    main()
