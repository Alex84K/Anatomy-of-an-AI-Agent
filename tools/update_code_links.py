#!/usr/bin/env python3
"""Обновляет номера строк в ссылках на код в Markdown-файлах.

Формат ссылки в тексте (символ указывается в title ссылки):

    [`run_agent()`](../code/ch01/agent_loop.py#L112-L184 "code:run_agent")
    [тест](../code/ch01/test_ch01.py "code:AgentLoopTests.test_token_budget")

Скрипт находит функцию/класс/метод по имени (через ast) и проставляет
актуальный диапазон строк #Lstart-Lend. Номера писать вручную не нужно.

Запуск:
    python3 tools/update_code_links.py           # обновить файлы
    python3 tools/update_code_links.py --check   # только проверить (для CI), код выхода 1 при расхождениях
"""
from __future__ import annotations

import ast
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_GLOBS = ["textbook/**/*.md", "*.md"]
LINK = re.compile(r'\]\((?P<path>[^)\s#]+\.py)(?:#L\d+(?:-L\d+)?)?\s+"code:(?P<sym>[\w.]+)"\)')


@lru_cache(maxsize=None)
def symbols(py: Path) -> dict[str, tuple[int, int]]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int]] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                name = prefix + child.name
                out[name] = (start, child.end_lineno or child.lineno)
                walk(child, name + ".")
            elif isinstance(child, (ast.Assign, ast.AnnAssign)) and not prefix:
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = (child.lineno, child.end_lineno or child.lineno)

    walk(tree, "")
    return out


def process(md: Path, check: bool) -> list[str]:
    text = md.read_text(encoding="utf-8")
    errors: list[str] = []

    def repl(m: re.Match) -> str:
        py = (md.parent / m["path"]).resolve()
        where = f"{md.relative_to(ROOT)}: {m['path']} :: {m['sym']}"
        if not py.exists():
            errors.append(f"нет файла — {where}")
            return m[0]
        rng = symbols(py).get(m["sym"])
        if rng is None:
            errors.append(f"нет символа — {where}")
            return m[0]
        a, b = rng
        anchor = f"#L{a}" if a == b else f"#L{a}-L{b}"
        return f']({m["path"]}{anchor} "code:{m["sym"]}")'

    # Примеры внутри блоков кода (```) не трогаем: это иллюстрации формата, а не ссылки
    parts = re.split(r"(^```.*?^```)", text, flags=re.M | re.S)
    new = "".join(part if part.startswith("```") else LINK.sub(repl, part) for part in parts)
    if new != text:
        if check:
            errors.append(f"устаревшие номера строк — {md.relative_to(ROOT)}")
        else:
            md.write_text(new, encoding="utf-8")
            print(f"обновлено: {md.relative_to(ROOT)}")
    return errors


def main() -> int:
    check = "--check" in sys.argv
    files = sorted({p for g in MD_GLOBS for p in ROOT.glob(g)})
    errors = [e for f in files for e in process(f, check)]
    for e in errors:
        print("ОШИБКА:", e, file=sys.stderr)
    if not errors:
        print("ссылки на код актуальны" if check else "готово")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
