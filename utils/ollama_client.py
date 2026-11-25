"""Клиент Ollama для эксперимента: стрим /api/chat с метриками, точный подсчёт токенов, сведения о сервере."""
from __future__ import annotations

import json
import re
import time

import requests

NS = 1e9
TOKENIZE_CTX = 64  # крошечное окно: любой реальный промпт его превышает -> сервер сообщает n_prompt_tokens без prefill


class OllamaClient:
    def __init__(self, host: str, timeout: int = 1800):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------ сведения о сервере
    def version(self) -> str:
        return requests.get(f"{self.host}/api/version", timeout=30).json().get("version")

    def tags(self) -> dict[str, dict]:
        models = requests.get(f"{self.host}/api/tags", timeout=30).json().get("models", [])
        return {m["name"]: m for m in models}

    def show(self, model: str) -> dict:
        r = requests.post(f"{self.host}/api/show", json={"model": model}, timeout=60).json()
        return {k: r.get(k) for k in ("parameters", "template", "capabilities", "details")}

    def ps(self) -> list[dict]:
        return requests.get(f"{self.host}/api/ps", timeout=30).json().get("models", [])

    def unload(self, model: str):
        """Выгружает модель (keep_alive=0) — вместе с ней сбрасывается кэш префиксов."""
        requests.post(f"{self.host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=120)
        for _ in range(60):
            if not any(m.get("name") == model or m.get("model") == model for m in self.ps()):
                return
            time.sleep(1)

    # ------------------------------------------------------------ подсчёт токенов
    def count_tokens(self, model: str, messages: list[dict], think) -> int:
        """Точное число токенов промпта (с чат-шаблоном модели), без генерации.

        Используется truncate=false: при превышении num_ctx сервер возвращает 400
        exceed_context_size_error с n_prompt_tokens.
        """
        payload = {"model": model, "messages": messages, "stream": False, "truncate": False,
                   "options": {"num_ctx": TOKENIZE_CTX, "num_predict": 1, "temperature": 0}}
        if think is not None:
            payload["think"] = think
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        body = r.json()
        err = body.get("error")
        if err:
            if isinstance(err, str):
                try:
                    err = json.loads(err).get("error", err)
                except (ValueError, AttributeError):
                    pass
            if isinstance(err, dict) and "n_prompt_tokens" in err:
                return int(err["n_prompt_tokens"])
            m = re.search(r"request \((\d+) tokens\)", str(err))
            if m:
                return int(m.group(1))
            raise RuntimeError(f"count_tokens: {err}")
        return int(body.get("prompt_eval_count") or 0)

    # ------------------------------------------------------------ генерация
    def chat(self, model: str, messages: list[dict], options: dict, think,
             keep_alive: str = "30m", fmt: dict | str | None = None) -> dict:
        """Стрим /api/chat. Возвращает ответ как есть + все метрики сервера и клиента.
        fmt — JSON-схема ответа (structured output); в прогонах участников не используется."""
        payload = {"model": model, "messages": messages, "stream": True,
                   "keep_alive": keep_alive, "truncate": False, "options": options}
        if think is not None:
            payload["think"] = think
        if fmt is not None:
            payload["format"] = fmt

        t0 = time.perf_counter()
        ttft_any = ttft_content = None
        thinking, content = [], []
        final: dict = {}

        with requests.post(f"{self.host}/api/chat", json=payload, stream=True,
                           timeout=self.timeout) as r:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(str(chunk["error"])[:500])
                msg = chunk.get("message", {})
                now = time.perf_counter() - t0
                if msg.get("thinking"):
                    ttft_any = ttft_any if ttft_any is not None else now
                    thinking.append(msg["thinking"])
                if msg.get("content"):
                    ttft_any = ttft_any if ttft_any is not None else now
                    ttft_content = ttft_content if ttft_content is not None else now
                    content.append(msg["content"])
                if chunk.get("done"):
                    final = chunk
                    break
        wall = time.perf_counter() - t0

        pe_n = final.get("prompt_eval_count", 0) or 0
        pe_d = final.get("prompt_eval_duration", 0) or 0
        ev_n = final.get("eval_count", 0) or 0
        ev_d = final.get("eval_duration", 0) or 0
        return {
            "answer_raw": "".join(content),
            "thinking_raw": "".join(thinking),
            "done_reason": final.get("done_reason"),
            "prompt_eval_count": pe_n,
            "eval_count": ev_n,
            "total_duration": final.get("total_duration"),
            "load_duration": final.get("load_duration"),
            "prompt_eval_duration": pe_d,
            "eval_duration": ev_d,
            "prefill_tps": round(pe_n / pe_d * NS, 2) if pe_d else None,
            "gen_tps": round(ev_n / ev_d * NS, 2) if ev_d else None,
            "ttft_s": round(ttft_any, 4) if ttft_any is not None else None,
            "ttft_content_s": round(ttft_content, 4) if ttft_content is not None else None,
            "wall_s": round(wall, 4),
        }
