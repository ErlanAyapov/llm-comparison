#!/usr/bin/env python3
"""Добавить модели в уже собранные задачи, НЕ меняя окна контекста и промпты.

Для каждой новой модели: точный подсчёт токенов каждого замороженного промпта (e1, e3, e2, e2_ru),
проверка лимита num_ctx - num_predict - safety, запись prompt_tokens[модель] в задачи,
digest и think — в manifest. Если хоть один промпт не влезает — ничего не записывается.

  python add_models.py --models llama3.1:8b-instruct-q4_K_M llama3.1:8b-instruct-fp16
Модели должны быть уже прописаны в config.yaml.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import dotenv
import yaml

import prompts
from build_tasks import prompt_sha, write_jsonl
from utils.ollama_client import OllamaClient

log = logging.getLogger("add_models")
EXPS = ("e1", "e3", "e2", "e2_ru")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dotenv.load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen, ctxc = cfg["generation"], cfg["context"]
    limit = gen["num_ctx"] - gen["num_predict"] - ctxc["safety_tokens"]
    tdir = Path(cfg["paths"]["tasks_dir"])
    man_p = tdir / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if man["prompt_token_limit"] != limit:
        sys.exit(f"лимит в manifest {man['prompt_token_limit']} != {limit} из config")

    by_name = {m["name"]: m for m in cfg["models"]}
    missing = [m for m in args.models if m not in by_name]
    if missing:
        sys.exit(f"нет в config.yaml: {missing}")

    client = OllamaClient(os.getenv("HOST") or cfg["host"])
    tags = client.tags()
    tasks = {e: [json.loads(l) for l in open(tdir / f"{e}.jsonl", encoding="utf-8")] for e in EXPS}

    for name in args.models:
        if name not in tags:
            sys.exit(f"{name}: модели нет на сервере")
        think = by_name[name]["think"]
        worst = {}
        for e, rows in tasks.items():
            mx = 0
            for i, t in enumerate(rows):
                msgs = prompts.build_messages(t["context"], t["question"])
                if prompt_sha(msgs) != t["prompt_sha256"]:
                    sys.exit(f"{t['qid']}: prompt_sha256 не совпадает — задачи изменены")
                n = client.count_tokens(name, msgs, think)
                if n > limit:
                    sys.exit(f"{e}/{t['qid']} не влезает у {name}: {n} > {limit}. Ничего не записано.")
                t["prompt_tokens"][name] = n
                mx = max(mx, n)
                if (i + 1) % 200 == 0:
                    log.info("%s %s %d/%d", name, e, i + 1, len(rows))
            worst[e] = mx
        log.info("%s: все промпты влезают, максимум %s (лимит %d)", name, worst, limit)
        man["models"][name] = {"think": think, "digest": tags[name]["digest"], "added_later": True}
        for e in EXPS:
            man["stats"][e].setdefault("max_prompt_tokens", {})[name] = worst[e]

    # запись — только если все модели прошли проверку
    backup = tdir / "manifest.before_add_models.json"
    if not backup.exists():
        shutil.copy(man_p, backup)
    for e, rows in tasks.items():
        write_jsonl(tdir / f"{e}.jsonl", rows)
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Готово: добавлены %s", args.models)


if __name__ == "__main__":
    main()
