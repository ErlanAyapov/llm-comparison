"""Хеш файла по содержимому (проверка целостности документов)."""
import hashlib
import os


def file_hash(path, algorithm: str = "sha256", chunk_size: int = 1 << 16) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
