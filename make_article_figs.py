#!/usr/bin/env python3
"""Рисунки и сводные цифры для статьи (судья gpt-oss:120b, версия j3).
-> reports/article/fig*.png, reports/article/numbers.json
  python make_article_figs.py
"""
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/article"); OUT.mkdir(parents=True, exist_ok=True)
J = Path("results/main/judge/gpt-oss_120b_j3")
MODELS = [  # (имя в Ollama, короткое имя, группа)
    ("gemma4:latest", "gemma4 (8B)", "univ"),
    ("qwen3:8b", "qwen3-8B", "univ"),
    ("huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B-GGUF4:latest", "KazLLM-1.0-8B", "kaz"),
    ("huggingface.co/issai/LLama-3.1-KazLLM-1.0-70B-GGUF4:latest", "KazLLM-1.0-70B", "kaz"),
    ("sherkala-chat:latest", "Sherkala-8B-Chat", "kaz"),
    ("llama3.1:8b-instruct-q4_K_M", "Llama-3.1-8B Q4", "base"),
    ("llama3.1:8b-instruct-fp16", "Llama-3.1-8B F16", "base"),
    ("llama3.1:70b-instruct-q4_K_M", "Llama-3.1-70B Q4", "base"),
    ("gpt-oss:120b", "gpt-oss-120b*", "ref"),
]
SHORT = {m: s for m, s, _ in MODELS}
GROUP = {s: g for _, s, g in MODELS}

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
C = {"correct": "#2a78d6", "partial": "#1baf7a", "incorrect": "#eb6834", "refusal": "#a9a8a2",
     "supported": "#1baf7a", "hallucination": "#eb6834"}
C_REF_B = "#2a78d6"  # честный отказ в E2 — «правильное» поведение, тот же синий, что и correct
GC = {"univ": "#2a78d6", "kaz": "#eb6834", "base": "#1baf7a", "ref": "#8a8984"}
GNAME = {"univ": "универсальные", "kaz": "адаптированные к казахскому", "base": "базовые Llama 3.1", "ref": "справочная"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def slug(s):
    return re.sub(r"[^\w.-]+", "_", s)


def labels(exp, model):
    rs = {}
    for l in open(J / exp / f"{slug(model)}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("label") or r["qid"] not in rs:
            rs[r["qid"]] = r
    return Counter(r["label"] for r in rs.values()), len(rs)


num = {}
for m, s, _ in MODELS:
    num[s] = {}
    for exp in ("e1", "e3", "e2", "e2_ru"):
        c, n = labels(exp, m)
        num[s][exp] = {k: round(100 * v / n, 1) for k, v in c.items() if k}
        num[s][exp]["n"] = n
        num[s][exp]["unjudged"] = c.get(None, 0)

# лексическая метрика (до судьи) — из interim_report
from interim_report import contains  # noqa: E402
tasks = {t["qid"]: t for t in map(json.loads, open("tasks/e1.jsonl", encoding="utf-8"))}
for m, s, _ in MODELS:
    rs = [json.loads(l) for l in open(f"results/main/e1/{slug(m)}.jsonl", encoding="utf-8")]
    num[s]["e1_lex"] = round(100 * sum(contains(r["answer_raw"] or "", tasks[r["qid"]]["gold_answer"]) for r in rs) / len(rs), 1)
# токены промпта на одном и том же тексте (E1, точный подсчёт с чат-шаблоном)
for m, s, _ in MODELS:
    num[s]["e1_prompt_tokens_median"] = st.median(t["prompt_tokens"][m] for t in tasks.values())
json.dump(num, open(OUT / "numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def stacked(ax, rows, keys, exp, title, colors):
    y = list(range(len(rows)))[::-1]
    for yi, s in zip(y, rows):
        left = 0
        for k in keys:
            v = num[s][exp].get(k, 0)
            ax.barh(yi, v, left=left, height=0.62, color=colors[k], edgecolor=SURF, linewidth=1.5)
            if v >= 6:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color=INK if colors[k] == "#a9a8a2" else "white")
            left += v
    ax.set_yticks(y); ax.set_yticklabels(rows)
    ax.set_xlim(0, 100); ax.set_xlabel("% ответов")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


order_a = sorted(num, key=lambda s: -num[s]["e1"].get("correct", 0))
keys_a = ["correct", "partial", "incorrect", "refusal"]
names_a = {"correct": "верно", "partial": "частично", "incorrect": "неверно", "refusal": "ложный отказ"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
stacked(axs[0], order_a, keys_a, "e1", "E1: вопрос на казахском", C)
stacked(axs[1], order_a, keys_a, "e3", "E3: вопрос на русском", C)
h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in keys_a]
fig.legend(h, [names_a[k] for k in keys_a], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "fig1_answerable.png", facecolor=SURF); plt.close(fig)

order_b = sorted(num, key=lambda s: num[s]["e2"].get("hallucination", 0))
keys_b = ["refusal", "supported", "hallucination"]
cb = {"refusal": C_REF_B, "supported": C["supported"], "hallucination": C["hallucination"]}
names_b = {"refusal": "честный отказ", "supported": "ответ из оставшегося текста", "hallucination": "галлюцинация"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
stacked(axs[0], order_b, keys_b, "e2", "E2: ответ вырезан, вопрос на казахском", cb)
stacked(axs[1], order_b, keys_b, "e2_ru", "E2_ru: ответ вырезан, вопрос на русском", cb)
h = [plt.Rectangle((0, 0), 1, 1, color=cb[k]) for k in keys_b]
fig.legend(h, [names_b[k] for k in keys_b], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "fig2_unanswerable.png", facecolor=SURF); plt.close(fig)

# рис. 3: точность vs галлюцинации
fig, ax = plt.subplots(figsize=(8, 5.6))
offs = {"gemma4 (8B)": (-8, -14), "qwen3-8B": (6, -12), "KazLLM-1.0-8B": (8, -12), "KazLLM-1.0-70B": (-8, 8),
        "Sherkala-8B-Chat": (-8, 6), "Llama-3.1-8B Q4": (6, 4), "Llama-3.1-8B F16": (6, 4),
        "Llama-3.1-70B Q4": (-8, -4), "gpt-oss-120b*": (8, 6)}
for s in num:
    x = num[s]["e1"].get("correct", 0)
    yv = (num[s]["e2"].get("hallucination", 0) + num[s]["e2_ru"].get("hallucination", 0)) / 2
    ax.scatter(x, yv, s=70, color=GC[GROUP[s]], edgecolor=SURF, linewidth=2, zorder=3)
    dx, dy = offs[s]
    ax.annotate(s, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK,
                ha="right" if dx < 0 else "left")
ax.set_xlabel("E1: верных ответов, %  (больше — лучше)")
ax.set_ylabel("Галлюцинаций, среднее E2 и E2_ru, %  (меньше — лучше)")
ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
ax.set_xlim(55, 96); ax.set_ylim(0, 55)
ax.annotate("лучше", xy=(95, 2), xytext=(92, 18), color=MUTED, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=MUTED))
h = [plt.Line2D([], [], marker="o", ls="", color=GC[g], markersize=8) for g in GC]
ax.legend(h, [GNAME[g] for g in GC], frameon=False, loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig3_tradeoff.png", facecolor=SURF); plt.close(fig)

# рис. 4: основа -> адаптация (гантели)
pairs = [("Llama-3.1-8B Q4", "KazLLM-1.0-8B"), ("Llama-3.1-8B F16", "Sherkala-8B-Chat"), ("Llama-3.1-70B Q4", "KazLLM-1.0-70B")]
metrics = [("E1: ложный отказ, %", lambda s: num[s]["e1"].get("refusal", 0)),
           ("E1: верно, %", lambda s: num[s]["e1"].get("correct", 0)),
           ("E2: галлюцинации, %", lambda s: num[s]["e2"].get("hallucination", 0)),
           ("E2_ru: галлюцинации, %", lambda s: num[s]["e2_ru"].get("hallucination", 0))]
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, (title, f) in zip(axs, metrics):
    for i, (b, a) in enumerate(pairs):
        yi = 2 - i
        vb, va = f(b), f(a)
        ax.plot([vb, va], [yi, yi], color=GRID, linewidth=3, zorder=1)
        ax.scatter(vb, yi, s=60, color=GC["base"], edgecolor=SURF, linewidth=2, zorder=3)
        ax.scatter(va, yi, s=60, color=GC["kaz"], edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(va, yi + 0.22, f"{va:.0f}", ha="center", fontsize=8, color=INK)
        ax.text(vb, yi - 0.38, f"{vb:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlim(0, 100); ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_ylim(-0.6, 2.6); ax.tick_params(axis="y", length=0)
axs[0].set_yticks([2, 1, 0]); axs[0].set_yticklabels(["8B Q4: → KazLLM", "8B F16: → Sherkala", "70B Q4: → KazLLM"])
h = [plt.Line2D([], [], marker="o", ls="", color=GC[g], markersize=8) for g in ("base", "kaz")]
fig.legend(h, ["базовая Llama 3.1 Instruct", "адаптированная модель"], loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig4_adaptation.png", facecolor=SURF); plt.close(fig)

# рис. 5: скорость (две отдельные шкалы — два графика)
speed = {"gemma4 (8B)": (61.5, 1.20), "KazLLM-1.0-8B": (39.8, 2.65), "qwen3-8B": (38.1, 2.71),
         "Sherkala-8B-Chat": (16.5, 1.07), "KazLLM-1.0-70B": (5.0, 19.69), "Llama-3.1-8B Q4": (40.8, 2.69),
         "Llama-3.1-8B F16": (16.2, 2.61), "Llama-3.1-70B Q4": (5.0, 19.70)}
order_s = sorted(speed, key=lambda s: -speed[s][0])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
y = list(range(len(order_s)))[::-1]
for ax, idx, title, fmt in ((axs[0], 0, "Генерация, токенов/с (больше — лучше)", "{:.1f}"),
                            (axs[1], 1, "Время до первого токена, с (меньше — лучше)", "{:.2f}")):
    vals = [speed[s][idx] for s in order_s]
    ax.barh(y, vals, height=0.6, color=[GC[GROUP[s]] for s in order_s], edgecolor=SURF, linewidth=1.5)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, fmt.format(v).replace(".", ","), va="center", fontsize=8, color=INK)
    ax.set_title(title, loc="left", fontsize=10, color=INK); ax.set_xlim(0, max(vals) * 1.15)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
axs[0].set_yticks(y); axs[0].set_yticklabels(order_s)
h = [plt.Rectangle((0, 0), 1, 1, color=GC[g]) for g in ("univ", "kaz", "base")]
fig.legend(h, [GNAME[g] for g in ("univ", "kaz", "base")], loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig5_speed.png", facecolor=SURF); plt.close(fig)

# рис. 6: токены на один и тот же текст
order_t = sorted(num, key=lambda s: num[s]["e1_prompt_tokens_median"])
fig, ax = plt.subplots(figsize=(8, 3.9))
y = list(range(len(order_t)))[::-1]
vals = [num[s]["e1_prompt_tokens_median"] for s in order_t]
ax.barh(y, vals, height=0.6, color=[GC[GROUP[s]] for s in order_t], edgecolor=SURF, linewidth=1.5)
for yi, v in zip(y, vals):
    ax.text(v + 60, yi, f"{v:,.0f}".replace(",", " "), va="center", fontsize=8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(order_t); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Медиана токенов промпта E1 (одинаковый текст для всех моделей)")
ax.set_xlim(0, 8400); ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "fig6_tokens.png", facecolor=SURF); plt.close(fig)

# рис. 7: лексическая метрика vs судья
order_l = sorted(num, key=lambda s: -num[s]["e1"].get("correct", 0))
fig, ax = plt.subplots(figsize=(9, 4.2))
y = list(range(len(order_l)))[::-1]
for yi, s in zip(y, order_l):
    lx, jd = num[s]["e1_lex"], num[s]["e1"].get("correct", 0)
    ax.plot([lx, jd], [yi, yi], color=GRID, linewidth=3, zorder=1)
    ax.scatter(lx, yi, s=55, color="#eb6834", edgecolor=SURF, linewidth=2, zorder=3)
    ax.scatter(jd, yi, s=55, color="#2a78d6", edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(jd + 1.5, yi, f"{jd:.0f}", va="center", fontsize=8, color=INK)
    ax.text(lx - 1.5, yi, f"{lx:.0f}", va="center", ha="right", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(order_l); ax.tick_params(axis="y", length=0)
ax.set_xlim(15, 100); ax.set_xlabel("E1, % ответов")
ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
h = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8) for c in ("#eb6834", "#2a78d6")]
ax.legend(h, ["эталон дословно в ответе (лексическая метрика)", "верно по смыслу (судья)"], frameon=False,
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig7_lex_vs_judge.png", facecolor=SURF); plt.close(fig)
print("ok", sorted(p.name for p in OUT.glob("*.png")))
