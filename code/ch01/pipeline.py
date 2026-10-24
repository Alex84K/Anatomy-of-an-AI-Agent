"""Глава 1. Скелет пайплайна «Блокируй рано, трать поздно».

Каждый этап — объект с методом run(ctx). Этап может:
  * дополнить контекст запроса (RequestContext);
  * отклонить запрос, бросив StageReject — дальше ничего не тратится.

Реализации этапов здесь учебные (регулярки, грубый подсчёт токенов).
В проекте их заменяют настоящими: классификатор инъекций, RAG, токенайзер модели.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class RequestContext:
    user_id: str
    query: str
    now_iso: str = ""                        # время запроса — динамика, кладём в КОНЕЦ контекста
    enrichment: dict[str, Any] = field(default_factory=dict)
    memory: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    model: str = ""
    thinking: bool = False
    prompt: list[dict] = field(default_factory=list)   # собранный контекст для LLM
    raw_output: str = ""
    output: dict[str, Any] | None = None
    trace: list[dict] = field(default_factory=list)


class StageReject(Exception):
    def __init__(self, stage: str, reason: str, user_message: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.user_message = stage, reason, user_message


class Stage(Protocol):
    name: str
    def run(self, ctx: RequestContext) -> RequestContext: ...


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)            # грубо; в проде — токенайзер модели


# ---------- 1. Валидация ($0) ----------

class Validate:
    name = "validate"
    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars
    def run(self, ctx):
        q = ctx.query.strip()
        if not q:
            raise StageReject(self.name, "empty", "Напишите, пожалуйста, ваш вопрос.")
        if len(q) > self.max_chars:
            raise StageReject(self.name, "too_long", "Сообщение слишком длинное.")
        if re.search(r"[\x00-\x08\x0e-\x1f]", q):
            raise StageReject(self.name, "binary", "Сообщение содержит недопустимые символы.")
        ctx.query = q
        return ctx


# ---------- 2. Входная безопасность ----------

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"игнорируй (все )?(предыдущие|прошлые) инструкции",
    r"(покажи|выведи|reveal|print).{0,20}(system prompt|системн\w* промпт)",
]

def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in INJECTION_PATTERNS)


class InputSecurity:
    name = "input_security"
    def run(self, ctx):
        if looks_like_injection(ctx.query):
            raise StageReject(self.name, "prompt_injection", "Не могу выполнить этот запрос.")
        return ctx


# ---------- 3–4. Обогащение, память, RAG (подключаются функциями) ----------

class Enrich:
    name = "enrich"
    def __init__(self, fetch: Callable[[RequestContext], dict]):
        self.fetch = fetch
    def run(self, ctx):
        ctx.enrichment = self.fetch(ctx)
        return ctx


class Retrieve:
    name = "retrieve"
    def __init__(self, memory: Callable[[str], list[str]], rag: Callable[[str], list[str]]):
        self.memory, self.rag = memory, rag
    def run(self, ctx):
        ctx.memory = self.memory(ctx.user_id)
        ctx.retrieved = self.rag(ctx.query)
        return ctx


# ---------- 5. Фильтрация найденного (непрямые инъекции, мусор) ----------

class ContentFilter:
    name = "content_filter"
    def run(self, ctx):
        kept = [d for d in ctx.retrieved if d.strip() and not looks_like_injection(d)]
        ctx.trace.append({"stage": self.name, "dropped": len(ctx.retrieved) - len(kept)})
        ctx.retrieved = kept
        return ctx


# ---------- 6. Сборка контекста и бюджет ----------

class Assemble:
    """Порядок важен для KV-кэша: статичное — в начало, динамичное — в конец."""
    name = "assemble"
    def __init__(self, system_prompt: str, budget_tokens: int = 3000):
        self.system_prompt, self.budget = system_prompt, budget_tokens

    def run(self, ctx):
        used = approx_tokens(self.system_prompt) + approx_tokens(ctx.query)
        # Greedy: берём документы и историю, пока влезают в бюджет
        docs = []
        for d in ctx.retrieved:
            if used + approx_tokens(d) > self.budget:
                break
            docs.append(d); used += approx_tokens(d)
        history = []
        for m in reversed(ctx.history):                  # свежие сообщения важнее
            t = approx_tokens(json.dumps(m, ensure_ascii=False))
            if used + t > self.budget:
                break
            history.insert(0, m); used += t

        dynamic = json.dumps({"now": ctx.now_iso, "context": ctx.enrichment,
                              "memory": ctx.memory, "documents": docs},
                             sort_keys=True, ensure_ascii=False)   # стабильный порядок ключей
        ctx.prompt = (
            [{"role": "system", "content": self.system_prompt}]   # без времени и переменных!
            + history
            + [{"role": "user", "content": f"<context>{dynamic}</context>\n\n{ctx.query}"}]
        )
        ctx.trace.append({"stage": self.name, "tokens": used, "docs": len(docs)})
        return ctx


# ---------- 7. Выбор модели и режима ----------

class Route:
    name = "route"
    def __init__(self, classify: Callable[[str], str], models: dict[str, str]):
        self.classify, self.models = classify, models
    def run(self, ctx):
        kind = self.classify(ctx.query)            # "simple" | "complex"
        ctx.model = self.models[kind]
        ctx.thinking = kind == "complex"
        return ctx


# ---------- 8. Генерация ----------

class Generate:
    name = "generate"
    def __init__(self, call: Callable[[RequestContext], str]):
        self.call = call                           # здесь может быть и run_agent(...)
    def run(self, ctx):
        ctx.raw_output = self.call(ctx)
        return ctx


# ---------- 9. Выходная безопасность ----------

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")

class OutputSecurity:
    name = "output_security"
    def __init__(self, system_prompt: str, canary: str):
        self.system_prompt, self.canary = system_prompt, canary
    def run(self, ctx):
        out = ctx.raw_output
        # Canary-токен — уникальная строка в системном промпте; появилась в ответе = утечка
        if self.canary in out or self.system_prompt[:80] in out:
            raise StageReject(self.name, "prompt_leak", "Не могу показать этот ответ.")
        ctx.raw_output = PHONE.sub("[телефон скрыт]", EMAIL.sub("[email скрыт]", out))
        return ctx


# ---------- 10. Форматирование и валидация ----------

class ValidateOutput:
    name = "validate_output"
    def __init__(self, required: dict[str, type]):
        self.required = required
    def run(self, ctx):
        try:
            data = json.loads(ctx.raw_output)
        except json.JSONDecodeError:
            raise StageReject(self.name, "invalid_json", "Не удалось сформировать ответ, попробуйте ещё раз.")
        for k, t in self.required.items():
            if not isinstance(data.get(k), t):
                raise StageReject(self.name, f"bad_field:{k}", "Не удалось сформировать ответ, попробуйте ещё раз.")
        ctx.output = data
        return ctx


# ---------- Запуск с трассировкой ----------

def run_pipeline(stages: list[Stage], ctx: RequestContext) -> RequestContext:
    for stage in stages:
        t0 = time.monotonic()
        try:
            ctx = stage.run(ctx)
            ctx.trace.append({"stage": stage.name, "ok": True,
                              "ms": round((time.monotonic() - t0) * 1000, 2)})
        except StageReject as r:
            ctx.trace.append({"stage": stage.name, "ok": False, "reason": r.reason})
            ctx.output = {"error": r.reason, "message": r.user_message}
            return ctx
    return ctx
