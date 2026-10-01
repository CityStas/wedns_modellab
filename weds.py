#!/usr/bin/env python3
"""Точка входа Wednesday.

Запуск без установки:
    python weds.py check
    python weds.py doctor
    python weds.py mcp          # MCP-сервер для любого агента

Только стандартная библиотека — зависимостей нет.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weds.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
