"""Доступ к документам эксперимента: проверка хеша и построчное чтение PDF.

Номера строк (line_ids в датасетах) — 1-based индексы в
"\n".join(page.get_text("text") for page in doc).splitlines().
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pymupdf

from utils.tools import file_hash


class HashMismatchError(RuntimeError):
    pass


def verify_hash(file_path: str, expected_hash: str) -> str:
    """Считает SHA-256 файла и сверяет с ожидаемым. Возвращает фактический хеш."""
    if not expected_hash:
        raise HashMismatchError(f"Нет эталонного хеша для {file_path}")
    actual = file_hash(file_path, algorithm="sha256")
    if actual != expected_hash:
        raise HashMismatchError(
            f"Хеш не совпадает: {file_path}\n  ожидался {expected_hash}\n  фактический {actual}")
    return actual


@lru_cache(maxsize=None)
def _read_lines_cached(file_path: str, hash_: str) -> tuple[str, ...]:
    with pymupdf.open(file_path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)
    return tuple(line.strip() for line in text.splitlines())


def load_lines(file_path: str, expected_hash: str) -> list[str]:
    """Проверяет хеш и возвращает строки документа (strip). Индекс i соответствует line_id i+1."""
    h = verify_hash(file_path, expected_hash)
    return list(_read_lines_cached(str(Path(file_path)), h))


def join_lines(lines: list[str]) -> str:
    """Текст для модели: строки через \\n, пустые строки опускаются."""
    return "\n".join(l for l in lines if l)
