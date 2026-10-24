"""Глава 1. Защищённый цикл агента (Agent Loop) с барьерами в коде.

Без внешних зависимостей. LLM подключается через протокол `LLM`:
любой клиент (Anthropic, OpenAI, локальная модель) оборачивается
в функцию, возвращающую `LLMResponse`.

Барьеры (все — в коде, не в промпте):
  1. лимит итераций;
  2. бюджет токенов на задачу;
  3. лимит одинаковых вызовов инструмента подряд;
  4. Human-in-the-Loop для необратимых действий;
  + лимит времени на всю задачу и таймаут на каждый инструмент.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ---------- Модель данных ----------

@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]

    def key(self) -> str:
        # sort_keys — стабильный порядок ключей (важно и для KV-кэша, и для сравнения)
        return self.name + ":" + json.dumps(self.args, sort_keys=True, ensure_ascii=False)


@dataclass
class LLMResponse:
    text: str | None                 # финальный ответ (если инструмент не вызывается)
    tool_call: ToolCall | None       # запрос на вызов инструмента
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def __call__(self, messages: list[dict], tools: list["Tool"] | None) -> LLMResponse: ...


class ToolError(Exception):
    """Ошибка, текст которой — инструкция для модели, а не стектрейс."""


@dataclass
class Tool:
    name: str                        # глагол_существительное: get_menu, create_order
    description: str                 # что делает и когда применять (позитивные формулировки)
    schema: dict[str, Any]           # JSON Schema аргументов
    fn: Callable[..., Any]
    irreversible: bool = False       # True → требуется подтверждение человека
    timeout_s: float = 10.0


@dataclass
class LoopLimits:
    max_iterations: int = 5          # саппорт ~5, кодинг-агент ~25
    max_total_tokens: int = 20_000
    max_same_call_repeats: int = 2   # сколько раз подряд можно повторить идентичный вызов
    max_wall_seconds: float = 60.0


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: str                 # final | max_iterations | token_budget | wall_clock | stuck
    iterations: int
    total_tokens: int
    messages: list[dict] = field(default_factory=list)


Approver = Callable[[ToolCall], bool]
Tracer = Callable[..., None]


def _noop_trace(event: str, **fields: Any) -> None:
    pass


# ---------- Вспомогательное ----------

_executor = ThreadPoolExecutor(max_workers=4)


def _run_tool(tool: Tool, args: dict) -> str:
    """Выполняет инструмент с таймаутом. Любая ошибка превращается в понятную инструкцию."""
    future = _executor.submit(tool.fn, **args)
    try:
        result = future.result(timeout=tool.timeout_s)
        return json.dumps(result, sort_keys=True, ensure_ascii=False)
    except FutureTimeout:
        return (f"ОШИБКА: {tool.name} не ответил за {tool.timeout_s} с. "
                "Не повторяй вызов сразу; сообщи пользователю, что сервис временно недоступен.")
    except ToolError as e:
        return f"ОШИБКА: {e}"
    except TypeError as e:  # неверные аргументы
        return (f"ОШИБКА: неверные аргументы для {tool.name} ({e}). "
                f"Схема аргументов: {json.dumps(tool.schema, ensure_ascii=False)}")
    except Exception as e:  # не отдаём модели стектрейс
        return (f"ОШИБКА: внутренний сбой {tool.name} ({type(e).__name__}). "
                "Не повторяй вызов; предложи пользователю попробовать позже.")


# ---------- Цикл ----------

def run_agent(
    llm: LLM,
    tools: list[Tool],
    messages: list[dict],
    limits: LoopLimits = LoopLimits(),
    approve: Approver = lambda call: False,   # по умолчанию необратимое запрещено
    trace: Tracer = _noop_trace,
) -> AgentResult:
    by_name = {t.name: t for t in tools}
    messages = list(messages)                 # не мутируем вход; добавляем только в конец
    total_tokens = 0
    started = time.monotonic()
    last_key, repeats = None, 0
    finalize_requested = False

    def finish(answer: str | None, reason: str, i: int) -> AgentResult:
        trace("loop.end", stop_reason=reason, iterations=i, total_tokens=total_tokens)
        return AgentResult(answer, reason, i, total_tokens, messages)

    for i in range(1, limits.max_iterations + 1):
        if time.monotonic() - started > limits.max_wall_seconds:
            return finish(None, "wall_clock", i - 1)

        # После просьбы завершиться инструменты не передаём — модель обязана ответить текстом
        resp = llm(messages, None if finalize_requested else tools)
        total_tokens += resp.input_tokens + resp.output_tokens
        trace("loop.iteration", i=i, input_tokens=resp.input_tokens,
              output_tokens=resp.output_tokens, tool=resp.tool_call.name if resp.tool_call else None)

        if resp.tool_call is None:
            messages.append({"role": "assistant", "content": resp.text})
            return finish(resp.text, "final", i)

        if finalize_requested:               # модель проигнорировала инструкцию
            return finish(None, "stuck", i)

        if total_tokens >= limits.max_total_tokens:
            return finish(None, "token_budget", i)

        call = resp.tool_call
        messages.append({"role": "assistant", "tool_call": {"name": call.name, "args": call.args}})

        # Барьер 3: идентичные вызовы подряд
        key = call.key()
        repeats = repeats + 1 if key == last_key else 0
        last_key = key
        if repeats >= limits.max_same_call_repeats:
            finalize_requested = True
            messages.append({"role": "tool", "name": call.name, "content":
                f"Инструмент {call.name} уже вызывался {repeats + 1} раза с теми же аргументами, "
                "новых данных не будет. Данные собраны — сформируй финальный ответ сейчас."})
            trace("guard.repeat", tool=call.name, repeats=repeats + 1)
            continue

        tool = by_name.get(call.name)
        if tool is None:
            content = (f"ОШИБКА: инструмента {call.name} нет. "
                       f"Доступны: {', '.join(sorted(by_name))}.")
        elif tool.irreversible and not approve(call):   # Барьер 4: HITL
            content = (f"Действие {call.name} отклонено человеком. Не повторяй его; "
                       "уточни у пользователя, что изменить.")
            trace("guard.hitl_rejected", tool=call.name)
        else:
            t0 = time.monotonic()
            content = _run_tool(tool, call.args)
            trace("tool.call", tool=call.name, ms=round((time.monotonic() - t0) * 1000),
                  error=content.startswith("ОШИБКА"))
        messages.append({"role": "tool", "name": call.name, "content": content})

    # Барьер 1: лимит итераций. Даём модели один шанс ответить без инструментов —
    # но только если бюджет позволяет.
    if total_tokens < limits.max_total_tokens:
        messages.append({"role": "user", "content":
            "Лимит шагов исчерпан. Ответь пользователю по уже собранным данным, без инструментов."})
        resp = llm(messages, None)
        total_tokens += resp.input_tokens + resp.output_tokens
        if resp.tool_call is None and resp.text:
            messages.append({"role": "assistant", "content": resp.text})
            return finish(resp.text, "max_iterations", limits.max_iterations)
    return finish(None, "max_iterations", limits.max_iterations)
