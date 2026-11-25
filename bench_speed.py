#!/usr/bin/env python3
"""Замер скорости на «холодном» кэше (протокол п. 6.1).

Ollama переиспользует KV-кэш общего префикса промпта, поэтому в основном прогоне
prefill_tps и ttft занижены по времени для вопросов к уже встречавшемуся документу.
Здесь каждый запрос начинается с уникальной метки -> общего префикса нет -> полный prefill.
Промпты — реальные задачи E1, равномерно по квантилям длины промпта.

  python bench_speed.py --run-id main --n 12 --reps 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import dotenv
import yaml

import prompts
from run_experiment import append_jsonl, ps_snapshot, slug
from utils.ollama_client import OllamaClient

log = logging.getLogger("bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--n", type=int, default=12, help="промптов на модель (по квантилям длины)")
    ap.add_argument("--reps", type=int, default=2, help="повторов каждого промпта (каждый с новой меткой)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dotenv.load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen = cfg["generation"]
    client = OllamaClient(os.getenv("HOST") or cfg["host"], timeout=gen["timeout_s"])
    tags = client.tags()
    tasks = [json.loads(l) for l in open(Path(cfg["paths"]["tasks_dir"]) / "e1.jsonl", encoding="utf-8")]
    options = {k: gen[k] for k in ("num_ctx", "num_predict", "temperature", "seed")}
    out_dir = Path(cfg["paths"]["results_dir"]) / args.run_id / "speed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in cfg["models"]:
        name, think = m["name"], m["think"]
        if args.models and name not in args.models:
            continue
        ranked = sorted(tasks, key=lambda t: t["prompt_tokens"][name])
        picks = [ranked[round(i * (len(ranked) - 1) / max(args.n - 1, 1))] for i in range(args.n)]

        client.unload(name)
        client.chat(name, prompts.build_messages("Тест.", "Жауап бер: иә."), options, think, gen["keep_alive"])
        ps = ps_snapshot(client, name)
        out = out_dir / f"{slug(name)}.jsonl"
        for t in picks:
            for rep in range(args.reps):
                msgs = prompts.build_messages(t["context"], t["question"])
                nonce = f"[{uuid.uuid4().hex[:8]}] "
                msgs[0]["content"] = nonce + msgs[0]["content"]
                rec = {"run_id": args.run_id, "model": name, "model_digest": tags[name]["digest"],
                       "qid": t["qid"], "rep": rep, "nonce": nonce,
                       "timestamp": datetime.now().isoformat(timespec="seconds"),
                       "think_sent": think, **options, "ps": ps, "error": None}
                try:
                    rec.update(client.chat(name, msgs, options, think, gen["keep_alive"]))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                append_jsonl(out, rec)
                log.info("[%s] %s rep%d | in %s | prefill %s t/s | ttft %s s | gen %s t/s",
                         name[-24:], t["qid"], rep, rec.get("prompt_eval_count"), rec.get("prefill_tps"),
                         rec.get("ttft_s"), rec.get("gen_tps"))


if __name__ == "__main__":
    main()
