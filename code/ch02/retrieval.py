"""Глава 2. Гибрид RAG + контекстное окно: найти немного, но нужного.

Без внешних зависимостей. Учебная реализация:
  - нарезка документов на чанки с перекрытием;
  - лексический поиск (упрощённый BM25) вместо векторной базы;
  - сборка контекста в пределах бюджета токенов с раскладкой «по краям»
    против эффекта Lost in the Middle;
  - сравнение стоимости «весь документ в контекст» и «RAG + окно».
В проекте поиск заменяют на эмбеддинги/гибридный поиск + реранкер, не меняя интерфейс.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

WORD = re.compile(r"\w+", re.U)


def tokens(text: str) -> int:
    """Грубая оценка числа токенов. В проекте — токенайзер провайдера."""
    return max(1, len(text) // 4)


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def chunk_text(source: str, text: str, size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Режет текст на куски по ~size символов с перекрытием.

    Перекрытие нужно, чтобы факт на границе двух чанков не потерялся целиком.
    Режем по границе слова, чтобы не рвать слова пополам.
    """
    if overlap >= size:
        raise ValueError("overlap должен быть меньше size")
    out, start, n = [], 0, 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        out.append(Chunk(f"{source}#{n}", source, text[start:end].strip()))
        if end >= len(text):
            break
        start, n = end - overlap, n + 1
    return out


class Index:
    """Упрощённый BM25 по чанкам."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(words(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if chunks else 1.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        q = words(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > min_score:
                scored.append((self.chunks[i], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def edge_order(items: list):
    """Раскладка «по краям»: 1-й по релевантности — в начало, 2-й — в конец, 3-й — второй и т. д.

    Модели хуже используют информацию из середины длинного контекста (Lost in the Middle),
    поэтому самое важное кладём туда, где внимание сильнее.
    """
    head, tail = [], []
    for i, x in enumerate(items):
        (head if i % 2 == 0 else tail).append(x)
    return head + tail[::-1]


def assemble(query: str, found: list[tuple[Chunk, float]], budget_tokens: int) -> tuple[str, list[str]]:
    """Собирает блок знаний для промпта в пределах бюджета. Возвращает (текст, id чанков).

    Жадно берём чанки по убыванию релевантности, пока влезают; затем раскладываем по краям.
    Каждому чанку — источник: модель может сослаться, а мы — проверить.
    """
    picked, used = [], 0
    for chunk, _ in found:
        cost = tokens(chunk.text) + 10
        if used + cost > budget_tokens:
            continue
        picked.append(chunk)
        used += cost
    ordered = edge_order(picked)
    body = "\n\n".join(f"[{c.id}]\n{c.text}" for c in ordered)
    return f"<knowledge query={query!r}>\n{body}\n</knowledge>", [c.id for c in ordered]


@dataclass
class CostCompare:
    long_context: float
    rag: float

    @property
    def ratio(self) -> float:
        return self.long_context / self.rag if self.rag else math.inf


def compare_costs(doc_tokens: int, rag_tokens: int, turns: int, price_in: float,
                  price_cache_read: float | None = None, embed_price: float = 0.02) -> CostCompare:
    """Сколько стоит диалог из `turns` ходов: весь документ в контексте против RAG.

    long context: документ читается на каждом ходу (из кэша, если он есть);
    RAG: документ один раз прогоняется через эмбеддинги + на каждом ходу найденные чанки.
    Цены — $ за 1M токенов.
    """
    if price_cache_read is None:
        long_ctx = doc_tokens * turns * price_in
    else:  # первый ход без кэша, дальше — чтение кэша
        long_ctx = doc_tokens * price_in + doc_tokens * (turns - 1) * price_cache_read
    rag = doc_tokens * embed_price + rag_tokens * turns * price_in
    return CostCompare(long_ctx / 1_000_000, rag / 1_000_000)
