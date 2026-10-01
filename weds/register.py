"""Регистрация локальной модели в конфигах агентских обвязок.

Ключевой момент: НИКОГДА не пишем availableModels. Этот ключ схлопывает
дропдаун моделей до перечисленных id и вырезает все облачные модели.
Если он нужен — пользователь добавит его сам.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

TARGETS = ("workbuddy", "codebuddy", "generic")


@dataclass
class Registration:
    target: str
    path: Path
    model_id: str
    added: bool
    note: str | None = None

    def describe(self) -> dict:
        return {
            "target": self.target,
            "path": str(self.path),
            "model_id": self.model_id,
            "added": self.added,
            "note": self.note,
        }


def target_path(target: str, workspace: Path | None = None) -> Path:
    """Куда писать models.json для данной обвязки."""
    if target == "workbuddy":
        # У WorkBuddy путь переопределён через WORKBUDDY_CONFIG_DIR.
        env = os.environ.get("WORKBUDDY_CONFIG_DIR")
        base = Path(env) if env else Path.home() / ".workbuddy-ai"
        return base / "models.json"
    if target == "codebuddy":
        env = os.environ.get("CODEBUDDY_CONFIG_DIR")
        base = Path(env) if env else Path.home() / ".codebuddy"
        return base / "models.json"
    if workspace:
        return workspace / "models.json"
    return Path.cwd() / "models.json"


def build_entry(model_id: str, base_url: str, *, max_input: int = 49152,
                max_output: int = 8192, temperature: float = 0.4,
                supports_tools: bool = True, supports_reasoning: bool = True,
                name: str | None = None, api_key: str = "local") -> dict:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/v1/chat/completions"
    return {
        "id": model_id,
        "name": name or f"{model_id} (local)",
        "vendor": "local",
        "url": url,
        "apiKey": api_key,
        "maxInputTokens": max_input,
        "maxOutputTokens": max_output,
        "temperature": temperature,
        "supportsToolCall": supports_tools,
        "supportsImages": False,
        "supportsReasoning": supports_reasoning,
    }


def register(target: str, entry: dict, *, path: Path | None = None,
             dry_run: bool = False) -> Registration:
    """Добавляет или обновляет модель в models.json, не трогая остальные записи."""
    dest = path or target_path(target)

    data: dict = {"models": []}
    if dest.is_file():
        try:
            loaded = json.loads(dest.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
            if not isinstance(data.get("models"), list):
                data["models"] = []
        except json.JSONDecodeError:
            return Registration(target, dest, entry["id"], False,
                                note=f"существующий файл не парсится как JSON: {dest}")

    models = data["models"]
    model_id = entry["id"]
    replaced = False
    for i, m in enumerate(models):
        if isinstance(m, dict) and m.get("id") == model_id:
            models[i] = {**m, **entry}
            replaced = True
            break
    if not replaced:
        models.append(entry)

    note = None
    if "availableModels" in data:
        note = ("В файле есть availableModels — он ограничивает список моделей в UI. "
                "Убедись, что новый id там перечислен, иначе модель не появится.")

    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return Registration(target, dest, model_id, not replaced, note=note)


def env_snippet(model_id: str, base_url: str, api_key: str = "local") -> str:
    """Переменные окружения для клиентов, которые не читают models.json."""
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    lines = [
        f"OPENAI_BASE_URL={base}",
        f"OPENAI_API_BASE={base}",
        f"OPENAI_API_KEY={api_key}",
        f"OPENAI_MODEL={model_id}",
    ]
    return "\n".join(lines)


def openai_client_snippet(model_id: str, base_url: str, api_key: str = "local") -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return (
        "from openai import OpenAI\n"
        f'client = OpenAI(base_url="{base}", api_key="{api_key}")\n'
        "resp = client.chat.completions.create(\n"
        f'    model="{model_id}",\n'
        '    messages=[{"role": "user", "content": "hi"}],\n'
        ")\n"
        "print(resp.choices[0].message.content)\n"
    )
