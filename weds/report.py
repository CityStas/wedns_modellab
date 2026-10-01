"""Форматирование вывода: человекочитаемо и --json для агентов.

Маркеры статуса — ASCII, чтобы не ломаться в консоли с кодировкой cp866.
"""

from __future__ import annotations

import json
import sys

from .checks import FAIL, PASS, SKIP, WARN, CheckResult

MARKS = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}


def enable_utf8() -> None:
    """Русский текст в выводе на Windows без кракозябр."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def print_checks(results: list[CheckResult], summary: dict | None = None,
                 title: str | None = None) -> None:
    if title:
        print(title)
        print("=" * len(title))
    width = max((len(r.name) for r in results), default=10)
    for r in results:
        print(f"{MARKS.get(r.status, '[????]')} {r.name:<{width}}  {r.summary}")
    if summary:
        print()
        print(f"Итог: {summary['verdict']}  "
              f"(ok {summary['pass']}, warn {summary['warn']}, "
              f"fail {summary['fail']}, skip {summary['skip']})")
    fixes = [r for r in results if r.fix]
    if fixes:
        print()
        print("Что делать:")
        for r in fixes:
            print(f"  - {r.name}: {r.fix}")


def print_models(models: list, server_url: str) -> None:
    print(f"Сервер: {server_url}")
    if not models:
        print("  (моделей не найдено)")
        return
    idw = max(len(m.id) for m in models)
    print(f"  {'ID':<{idw}}  {'СОСТОЯНИЕ':<11} {'КОНТЕКСТ':>9} {'КВАНТ':<9} АРХ")
    for m in models:
        ctx = str(m.context) if m.context else "-"
        print(f"  {m.id:<{idw}}  {m.state:<11} {ctx:>9} "
              f"{(m.quantization or '-'):<9} {m.arch or '-'}")


def print_kv(data: dict, indent: int = 0) -> None:
    """Плоский дамп словаря — для bench/tune."""
    pad = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            print_kv(v, indent + 2)
        elif isinstance(v, list):
            print(f"{pad}{k}:")
            for item in v:
                if isinstance(item, dict):
                    print(f"{pad}  - " + ", ".join(f"{ik}={iv}" for ik, iv in item.items()))
                else:
                    print(f"{pad}  - {item}")
        else:
            print(f"{pad}{k}: {v}")


def emit(payload: dict, as_json: bool, human: str | None = None) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif human:
        print(human)
    else:
        print_kv(payload)


def print_curve(curve_desc: dict, projected: float | None = None) -> None:
    print("Кривая памяти движка:")
    print(f"  вес модели:      {curve_desc['weights_gib']} ГиБ")
    print(f"  цена контекста:  {curve_desc['kib_per_token_base']} КиБ/токен (базовая точность оценщика)")
    for s in curve_desc.get("samples", []):
        print(f"    ctx {s['ctx']:>7} -> {s['total_gib']} ГиБ")
    if projected is not None:
        print(f"  проекция итого:  {projected} ГиБ (с выбранным квантом KV)")
