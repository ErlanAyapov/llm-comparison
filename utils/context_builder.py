"""Окно контекста вокруг фрагмента с ответом.

* Документ помещается в бюджет символов -> подаётся целиком (doc_whole).
* Иначе берётся непрерывное окно из целых строк, гарантированно содержащее все line_ids;
  доля свободного места ДО ответа (answer_offset) случайна (не по центру) и задаётся
  генератором random.Random(f"{seed}:{qid}") — окно воспроизводимо и не зависит от порядка.
* Если строки ответа разнесены так далеко, что непрерывный диапазон не влезает
  (6 вопросов из 705), окно собирается из нескольких фрагментов вокруг групп строк,
  разделённых маркером GAP_MARKER (multi_segment=True).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from utils.doc_store import join_lines

GAP_MARKER = "[...]"
CLUSTER_GAP = 30  # строки ответа, разнесённые больше чем на столько строк, — разные группы


@dataclass
class Window:
    segments: list[tuple[int, int]]   # [(start, end)] 1-based, включительно
    text: str
    doc_whole: bool
    answer_offset: float | None
    multi_segment: bool = False
    span_exceeds_budget: bool = False

    @property
    def start(self):
        return self.segments[0][0]

    @property
    def end(self):
        return self.segments[-1][1]


def _len(lines: list[str]) -> int:
    return len(join_lines(lines))


def _expand(lines, a, b, budget, frac, lo, hi):
    """Расширяет [a, b] (0-based) в пределах [lo, hi] до budget символов; frac — доля свободного места сверху."""
    span_len = _len(lines[a:b + 1])
    before_budget = int(max(budget - span_len, 0) * frac)
    s, e = a, b
    while s > lo and _len(lines[s - 1:b + 1]) - span_len <= before_budget:
        s -= 1
    while e < hi and _len(lines[s:e + 2]) <= budget:
        e += 1
    while s > lo and _len(lines[s - 1:e + 1]) <= budget:
        s -= 1
    return s, e


def build_window(lines: list[str], line_ids: list[int], budget_chars: int,
                 seed: int, qid: str) -> Window:
    n = len(lines)
    if _len(lines) <= budget_chars:
        return Window([(1, n)], join_lines(lines), True, None)

    rng = random.Random(f"{seed}:{qid}")
    frac = rng.random()
    ids = sorted(i - 1 for i in line_ids)
    a, b = ids[0], ids[-1]

    if _len(lines[a:b + 1]) <= budget_chars:
        s, e = _expand(lines, a, b, budget_chars, frac, 0, n - 1)
        return Window([(s + 1, e + 1)], join_lines(lines[s:e + 1]), False, round(frac, 4))

    # --- несколько групп строк
    clusters = [[ids[0], ids[0]]]
    for i in ids[1:]:
        if i - clusters[-1][1] > CLUSTER_GAP:
            clusters.append([i, i])
        else:
            clusters[-1][1] = i
    sep = len(f"\n{GAP_MARKER}\n") * (len(clusters) - 1)
    core = sum(_len(lines[x:y + 1]) for x, y in clusters) + sep
    if core > budget_chars:
        segs = [(x + 1, y + 1) for x, y in clusters]
        text = f"\n{GAP_MARKER}\n".join(join_lines(lines[x:y + 1]) for x, y in clusters)
        return Window(segs, text, False, None, True, True)

    extra = (budget_chars - core) // len(clusters)
    segs = []
    prev_end = -1
    for k, (x, y) in enumerate(clusters):
        hi = clusters[k + 1][0] - 1 if k + 1 < len(clusters) else n - 1
        s, e = _expand(lines, x, y, _len(lines[x:y + 1]) + extra, frac, prev_end + 1, hi)
        segs.append((s, e))
        prev_end = e
    # склеиваем соседние сегменты, если они сомкнулись
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    text = f"\n{GAP_MARKER}\n".join(join_lines(lines[s:e + 1]) for s, e in merged)
    return Window([(s + 1, e + 1) for s, e in merged], text, False, round(frac, 4), len(merged) > 1)
