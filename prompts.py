"""Единый промпт для всех экспериментов и моделей.

При любом изменении текста — увеличить PROMPT_VERSION и пересобрать задачи (build_tasks.py).
"""
PROMPT_VERSION = "v1"

# Фиксированная фраза отказа (для автоматического подсчёта в E2), одна для любого языка вопроса.
REFUSAL_PHRASE = "Құжатта жауап жоқ."

SYSTEM_PROMPT = (
    "Сіз құжат бойынша сұрақтарға жауап беретін көмекшісіз. "
    "Тек төменде берілген құжат мәтінін пайдаланыңыз, өз біліміңізді қоспаңыз. "
    "Жауапты қысқа әрі нақты, сұрақ қойылған тілде беріңіз. "
    f"Егер құжатта сұраққа жауап болмаса, дәл осылай жазыңыз: «{REFUSAL_PHRASE}»"
)

USER_TEMPLATE = "Құжат:\n<<<\n{context}\n>>>\n\nСұрақ: {question}"


def build_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
