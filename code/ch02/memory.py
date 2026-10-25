"""Глава 2. Память агента: четыре типа, фоновое чтение и запись, сжатие контекста.

Без внешних зависимостей.
  - Procedural / Semantic / Episodic / Working — четыре типа памяти по CoALA,
    у каждого свой жизненный цикл и своё место в контексте;
  - prefetch — выбор релевантных файлов памяти по заголовкам в фоне,
    не блокируя основной запрос;
  - EpisodicWriter — фоновая запись уроков после ответа с курсором
    и семантикой at-least-once (поэтому запись идемпотентна);
  - compact — лестница сжатия контекста: от дешёвых шагов к дорогим.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from retrieval import tokens


# ---------- Четыре типа памяти ----------

@dataclass
class Procedural:
    """Как действовать: системный промпт и описания инструментов. Меняет разработчик, между релизами."""
    system_prompt: str
    tool_descriptions: list[str]

    def render(self) -> str:
        return self.system_prompt + "\n\n# Инструменты\n" + "\n".join(self.tool_descriptions)


@dataclass
class MemoryFile:
    """Единица эпизодической/пользовательской памяти: заголовок отдельно от тела.

    По заголовку (description) решают, читать ли файл; тело грузят только при необходимости.
    """
    name: str
    description: str
    body: str


@dataclass
class Episode:
    """Урок из опыта: [контекст] → [действие] → [результат] → [вывод]."""
    context: str
    action: str
    outcome: str
    lesson: str
    source: str = ""                  # откуда взят урок (id сообщения) — якорь происхождения

    def key(self) -> str:
        # Ключ идемпотентности: один и тот же урок, записанный дважды, — одна запись
        raw = json.dumps([self.context, self.action, self.lesson], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Working:
    """Что происходит прямо сейчас: сообщения текущей сессии. Живёт до конца сессии."""
    messages: list[dict] = field(default_factory=list)

    def size(self) -> int:
        return sum(tokens(m.get("content", "")) for m in self.messages)


# ---------- Фоновый prefetch памяти ----------

Selector = Callable[[str, list[tuple[str, str]]], list[str]]
_pool = ThreadPoolExecutor(max_workers=2)


def start_prefetch(query: str, files: list[MemoryFile], selector: Selector, limit: int = 5) -> Future:
    """Запускает выбор релевантных файлов памяти в фоне и сразу возвращает Future.

    selector — дешёвая модель: видит только (имя, описание), возвращает имена файлов.
    Основной запрос в это время продолжает подготовку (пайплайн, роутинг).
    """
    manifest = [(f.name, f.description) for f in files]
    by_name = {f.name: f for f in files}

    def job() -> list[MemoryFile]:
        names = selector(query, manifest)
        # Модель может вернуть мусор или лишнее — фильтруем и ограничиваем в коде
        picked = [by_name[n] for n in dict.fromkeys(names) if n in by_name]
        return picked[:limit]

    return _pool.submit(job)


def collect_prefetch(fut: Future, wait_s: float) -> list[MemoryFile]:
    """Забирает результат prefetch. Не успел или упал — работаем без памяти, а не ждём."""
    try:
        return fut.result(timeout=wait_s)
    except FutureTimeout:
        return []
    except Exception:
        return []


# ---------- Фоновая запись эпизодической памяти ----------

Extractor = Callable[[list[dict]], list[Episode]]


class EpisodicWriter:
    """Извлекает уроки из новых сообщений и дописывает их в append-only журнал.

    Курсор — индекс первого необработанного сообщения. Он сдвигается только
    после успешной записи. Если процесс упал между записью и сдвигом курсора,
    те же сообщения обработаются ещё раз (at-least-once) — дубликаты отсекает ключ.
    """

    def __init__(self, log_path: Path, extractor: Extractor):
        self.log_path, self.extractor = log_path, extractor
        self.cursor = 0
        self._keys = set()
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                self._keys.add(json.loads(line)["key"])

    def run(self, messages: list[dict]) -> int:
        """Обрабатывает сообщения после курсора. Возвращает число новых записей."""
        fresh = messages[self.cursor:]
        if not fresh:
            return 0
        episodes = self.extractor(fresh)          # может бросить исключение → курсор не двигается
        written = 0
        with self.log_path.open("a", encoding="utf-8") as f:
            for ep in episodes:
                k = ep.key()
                if k in self._keys:
                    continue
                f.write(json.dumps({"key": k, **ep.__dict__}, ensure_ascii=False) + "\n")
                self._keys.add(k)
                written += 1
        self.cursor = len(messages)               # только после успешной записи
        return written

    def lessons(self) -> list[Episode]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            d.pop("key")
            out.append(Episode(**d))
        return out


# ---------- Лестница сжатия контекста ----------

@dataclass
class CompactConfig:
    window: int = 8_000               # размер окна модели (токены)
    offload_over_chars: int = 8_000   # результат инструмента длиннее — уносим на диск
    keep_head_chars: int = 2_000      # сколько оставить модели от вынесенного результата
    micro_at: float = 0.6             # доля окна, после которой чистим старые результаты инструментов
    auto_at: float = 0.8              # доля окна, после которой делаем резюме истории
    keep_recent: int = 6              # сколько последних сообщений не трогаем никогда


Summarizer = Callable[[list[dict]], str]


def _size(msgs: list[dict]) -> int:
    return sum(tokens(m.get("content", "")) for m in msgs)


def compact(messages: list[dict], cfg: CompactConfig, store: Path,
            summarize: Summarizer) -> tuple[list[dict], list[str]]:
    """Сжимает историю, поднимаясь по лестнице, пока контекст не влезет. Возвращает (сообщения, шаги).

    Уровень 1 — offload: большие результаты инструментов на диск, модели — начало и путь к файлу.
               Применяется к новому результату сразу, поэтому префикс не меняется и кэш цел.
    Уровень 2 — microcompact: содержимое старых результатов инструментов заменяется заглушкой.
    Уровень 3 — autocompact: старая часть истории заменяется резюме, последние сообщения сохраняются.
    """
    steps: list[str] = []
    out = []
    store.mkdir(parents=True, exist_ok=True)

    # Уровень 1
    for i, m in enumerate(messages):
        c = m.get("content", "")
        if m.get("role") == "tool" and len(c) > cfg.offload_over_chars:
            path = store / f"tool_{i}.txt"
            path.write_text(c, encoding="utf-8")
            m = {**m, "content": c[:cfg.keep_head_chars] +
                 f"\n…[полный результат: {path}, {len(c)} символов; прочитай файл, если нужно]"}
            steps.append(f"offload:{i}")
        out.append(m)

    # Уровень 2
    if _size(out) > cfg.window * cfg.micro_at:
        cut = len(out) - cfg.keep_recent
        for i in range(max(0, cut)):
            if out[i].get("role") == "tool" and not out[i]["content"].startswith("[очищено"):
                out[i] = {**out[i], "content": "[очищено: старый результат инструмента]"}
                steps.append(f"micro:{i}")

    # Уровень 3
    if _size(out) > cfg.window * cfg.auto_at and len(out) > cfg.keep_recent:
        head, tail = out[:-cfg.keep_recent], out[-cfg.keep_recent:]
        summary = summarize(head)
        out = [{"role": "user", "content": "[Резюме предыдущей части сессии]\n" + summary}] + tail
        steps.append("auto")

    return out, steps


# ---------- Сборка контекста из всех типов памяти ----------

def build_context(proc: Procedural, knowledge: str, recalled: list[MemoryFile],
                  lessons: list[Episode], work: Working) -> list[dict]:
    """Статичное — в начало (кэшируется), динамичное — в конец.

    Порядок: процедурная (system) → рабочая (история сессии, только дописывается) →
    блок этого хода: найденные знания, воспоминания, уроки. Найденное меняется
    от хода к ходу, поэтому стоит после истории и не ломает её кэш.
    """
    dynamic = []
    if knowledge:
        dynamic.append(knowledge)
    if recalled:
        dynamic.append("<memory>\n" + "\n".join(f"## {f.name}\n{f.body}" for f in recalled) + "\n</memory>")
    if lessons:
        dynamic.append("<lessons>\n" + "\n".join(f"- {e.lesson}" for e in lessons) + "\n</lessons>")
    msgs = [{"role": "system", "content": proc.render()}] + work.messages
    if dynamic:
        msgs.append({"role": "user", "content": "\n\n".join(dynamic)})
    return msgs
