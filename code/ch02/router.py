"""Глава 2. Умный роутинг: какую модель вызвать и сколько ей думать.

Без внешних зависимостей. Три слоя выбора модели:
  1. фильтр по модальностям — отсекаем модели, которые не примут вход;
  2. классификатор сложности — маленькая быстрая модель (или ML-классификатор);
  3. резервный слой — детерминированные правила по ключевым фразам, затем модель по умолчанию.
Поверх — учёт кэша промпта: переключение модели посреди диалога сбрасывает кэш,
поэтому решение «сменить модель» принимается по стоимости хода, а не по прайсу.
Отдельно — политика Thinking Mode: рассуждения дороги (оплачиваются как выход),
поэтому включаются на фазе планирования и выключаются на фазе исполнения.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


# ---------- Модели и цены ----------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: str                         # "small" | "frontier"
    price_in: float                   # $ за 1M входных токенов
    price_out: float                  # $ за 1M выходных (и thinking) токенов
    price_cache_read: float           # $ за 1M токенов, прочитанных из кэша
    price_cache_write: float          # $ за 1M токенов, записанных в кэш
    modalities: frozenset[str] = frozenset({"text"})


def turn_cost(m: ModelSpec, cached: int, uncached: int, output: int, cache_write: int = 0) -> float:
    """Стоимость одного хода в долларах.

    cached      — токены префикса, которые модель читает из кэша;
    uncached    — токены, которые модель читает без кэша;
    cache_write — сколько из uncached при этом записывается в кэш (дороже обычного входа);
    output      — выходные токены, включая thinking.
    """
    plain = uncached - cache_write
    return (cached * m.price_cache_read + plain * m.price_in
            + cache_write * m.price_cache_write + output * m.price_out) / 1_000_000


# ---------- Запрос и решение ----------

@dataclass
class Request:
    text: str
    modalities: frozenset[str] = frozenset({"text"})
    current_model: str | None = None  # на какой модели идёт диалог (None — первый ход)
    history_tokens: int = 0           # длина уже накопленного контекста
    new_tokens: int = 200             # сколько добавится на этом ходу
    expected_output: int = 300


@dataclass
class RouteDecision:
    model: ModelSpec
    reason: str                       # для трассировки: почему выбрана эта модель
    trace: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    label: str                        # "simple" | "complex"
    confidence: float                 # 0..1


class Classifier(Protocol):
    def __call__(self, text: str) -> Verdict: ...


# ---------- Слой 3: детерминированные правила ----------

COMPLEX_HINTS = re.compile(r"(сравни|проанализируй|почему|спланируй|составь план|рассчитай|объясни разницу)", re.I)
SIMPLE_HINTS = re.compile(r"(сколько стоит|цена|статус заказа|часы работы|адрес|есть ли)", re.I)


def rule_fallback(text: str) -> str | None:
    """Правила по ключевым фразам. None — правила не знают, что делать."""
    if COMPLEX_HINTS.search(text):
        return "complex"
    if SIMPLE_HINTS.search(text):
        return "simple"
    return None


# ---------- Роутер ----------

def route(
    req: Request,
    models: list[ModelSpec],
    classifier: Classifier,
    default_tier: str = "frontier",
    min_confidence: float = 0.7,
) -> RouteDecision:
    trace: list[str] = []

    # Слой 1: модальности. Дешевле всего — не отправлять картинку в текстовую модель.
    capable = [m for m in models if req.modalities <= m.modalities]
    if not capable:
        raise ValueError(f"Нет модели для модальностей {sorted(req.modalities)}")
    trace.append("modality:" + ",".join(m.name for m in capable))

    # Слой 2: классификатор. Ошибка или низкая уверенность → слой 3.
    tier: str | None = None
    try:
        v = classifier(req.text)
        trace.append(f"classifier:{v.label}:{v.confidence:.2f}")
        if v.confidence >= min_confidence:
            tier = "small" if v.label == "simple" else "frontier"
    except Exception as e:  # классификатор упал — запрос не должен упасть вместе с ним
        trace.append(f"classifier_error:{type(e).__name__}")

    reason = "classifier"
    if tier is None:
        label = rule_fallback(req.text)
        if label is not None:
            tier, reason = ("small" if label == "simple" else "frontier"), "rules"
        else:
            tier, reason = default_tier, "default"
        trace.append(f"fallback:{reason}:{tier}")

    pool = [m for m in capable if m.tier == tier] or capable
    chosen = min(pool, key=lambda m: m.price_in)

    # Учёт кэша: если диалог уже идёт на другой модели, сравниваем стоимость хода.
    current = next((m for m in capable if m.name == req.current_model), None)
    if current is not None and current.name != chosen.name and req.history_tokens:
        stay = turn_cost(current, req.history_tokens, req.new_tokens, req.expected_output)
        switch = turn_cost(chosen, 0, req.history_tokens + req.new_tokens, req.expected_output,
                           cache_write=req.history_tokens + req.new_tokens)
        trace.append(f"cache_check:stay=${stay:.4f}:switch=${switch:.4f}")
        if stay <= switch:
            return RouteDecision(current, "cache_stay", trace)

    return RouteDecision(chosen, reason, trace)


# ---------- Thinking Mode ----------

@dataclass
class ThinkingPolicy:
    """Рассуждаем на этапе планирования, исполняем без рассуждений."""
    plan_iterations: int = 2          # сколько первых итераций цикла думаем
    budget_tokens: int = 4_000        # потолок thinking-токенов на итерацию
    replan_on_error: bool = True      # ошибка инструмента → снова думаем один шаг

    def budget(self, iteration: int, last_step_failed: bool = False) -> int:
        """Сколько thinking-токенов разрешить на итерации (1-based). 0 — режим выключен."""
        if iteration <= self.plan_iterations:
            return self.budget_tokens
        if self.replan_on_error and last_step_failed:
            return self.budget_tokens // 2
        return 0
