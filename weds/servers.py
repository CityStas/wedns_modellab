"""Адаптеры локальных инференс-серверов.

Все общаются по OpenAI-совместимому /v1/chat/completions, но у каждого свои
эндпоинты для списка моделей, свой способ узнать реальный размер контекста
и свои особенности загрузки.

Только стандартная библиотека — никаких зависимостей, чтобы запускалось
где угодно и на голом Python.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT = 900  # секунд; префилл 40k токенов на слабой карте идёт минуты


# --------------------------------------------------------------------------- #
# Результат запроса
# --------------------------------------------------------------------------- #

@dataclass
class ChatResult:
    ok: bool
    elapsed: float = 0.0
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None
    error_type: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def gen_tps(self) -> float:
        return self.completion_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def is_context_overflow(self) -> bool:
        return self.error_type == "exceed_context_size_error"


@dataclass
class ModelInfo:
    id: str
    state: str = "unknown"          # loaded / not-loaded / unknown
    context: int | None = None      # загруженный контекст, если известен
    max_context: int | None = None  # максимум, который поддерживает модель
    quantization: str | None = None
    publisher: str | None = None
    arch: str | None = None
    capabilities: list = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        return self.state == "loaded"

    def describe(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "context": self.context,
            "max_context": self.max_context,
            "quantization": self.quantization,
            "publisher": self.publisher,
            "arch": self.arch,
            "capabilities": self.capabilities,
        }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _http_json(url: str, payload: dict | None = None, timeout: float = 30.0):
    """Возвращает (status, parsed_json_or_None, raw_text). Не бросает на 4xx/5xx."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return r.status, None, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw), raw
        except json.JSONDecodeError:
            return e.code, None, raw
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        return 0, None, str(e)


# --------------------------------------------------------------------------- #
# Базовый сервер
# --------------------------------------------------------------------------- #

class LocalServer:
    """Общий интерфейс. Наследники переопределяют список моделей и метаданные."""

    kind = "generic"

    def __init__(self, base_url: str, api_key: str = "local", timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ---- обязательное ---------------------------------------------------- #

    def probe(self) -> bool:
        raise NotImplementedError

    def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    # ---- общее ----------------------------------------------------------- #

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        timeout: float | None = None,
        retries: int = 1,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Локальные серверы иногда рвут соединение на больших промптах —
        # один повтор снимает эту плавающую ошибку.
        last: ChatResult | None = None
        for attempt in range(max(1, retries + 1)):
            t0 = time.perf_counter()
            status, body, raw = _http_json(
                f"{self.base_url}/v1/chat/completions",
                payload,
                timeout=timeout or self.timeout,
            )
            elapsed = time.perf_counter() - t0

            if status == 0:
                last = ChatResult(ok=False, elapsed=elapsed, error=raw,
                                  error_type="transport")
                if attempt < retries:
                    time.sleep(1.0)
                    continue
                return last

            if status != 200 or not isinstance(body, dict):
                err = (body or {}).get("error", {}) if isinstance(body, dict) else {}
                msg = err.get("message") or raw[:500]
                return ChatResult(
                    ok=False,
                    elapsed=elapsed,
                    error=msg,
                    error_type=err.get("type") or f"http_{status}",
                    raw=body if isinstance(body, dict) else {},
                )

            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = body.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}

            return ChatResult(
                ok=True,
                elapsed=elapsed,
                content=message.get("content") or "",
                reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
                tool_calls=message.get("tool_calls") or [],
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
                reasoning_tokens=details.get("reasoning_tokens") or 0,
                raw=body,
            )

        return last or ChatResult(ok=False, error="no attempt made", error_type="transport")

    def context_length(self, model: str) -> int | None:
        """Загруженный контекст для модели, если сервер его отдаёт."""
        for m in self.list_models():
            if m.id == model:
                return m.context
        return None


# --------------------------------------------------------------------------- #
# LM Studio / Bionic
# --------------------------------------------------------------------------- #

class LMStudioServer(LocalServer):
    kind = "lmstudio"

    def probe(self) -> bool:
        status, _, _ = _http_json(f"{self.base_url}/api/v0/models", timeout=5)
        if status == 200:
            return True
        status, _, _ = _http_json(f"{self.base_url}/v1/models", timeout=5)
        return status == 200

    def list_models(self) -> list[ModelInfo]:
        status, body, _ = _http_json(f"{self.base_url}/api/v0/models", timeout=10)
        if status != 200 or not isinstance(body, dict):
            status, body, _ = _http_json(f"{self.base_url}/v1/models", timeout=10)
            if status != 200 or not isinstance(body, dict):
                return []
            return [
                ModelInfo(id=m.get("id", "?"), state="unknown")
                for m in body.get("data", [])
                if m.get("id")
            ]

        out = []
        for m in body.get("data", []):
            if m.get("type") not in (None, "llm"):
                continue
            out.append(
                ModelInfo(
                    id=m.get("id", "?"),
                    state=m.get("state", "unknown"),
                    context=m.get("loaded_context_length"),
                    max_context=m.get("max_context_length"),
                    quantization=m.get("quantization"),
                    publisher=m.get("publisher"),
                    arch=m.get("arch"),
                    capabilities=m.get("capabilities") or [],
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

class OllamaServer(LocalServer):
    kind = "ollama"

    def probe(self) -> bool:
        status, _, _ = _http_json(f"{self.base_url}/api/tags", timeout=5)
        return status == 200

    def list_models(self) -> list[ModelInfo]:
        status, body, _ = _http_json(f"{self.base_url}/api/tags", timeout=10)
        if status != 200 or not isinstance(body, dict):
            return []
        return [
            ModelInfo(
                id=m.get("name", "?"),
                state="loaded" if m.get("size_vram") else "not-loaded",
                quantization=(m.get("details") or {}).get("quantization_level"),
                arch=(m.get("details") or {}).get("family"),
            )
            for m in body.get("models", [])
            if m.get("name")
        ]


# --------------------------------------------------------------------------- #
# llama.cpp server
# --------------------------------------------------------------------------- #

class LlamaCppServer(LocalServer):
    kind = "llamacpp"

    def probe(self) -> bool:
        status, _, _ = _http_json(f"{self.base_url}/v1/models", timeout=5)
        return status == 200

    def list_models(self) -> list[ModelInfo]:
        status, body, _ = _http_json(f"{self.base_url}/v1/models", timeout=10)
        if status != 200 or not isinstance(body, dict):
            return []
        n_ctx = None
        pstatus, props, _ = _http_json(f"{self.base_url}/props", timeout=10)
        if pstatus == 200 and isinstance(props, dict):
            n_ctx = props.get("n_ctx") or (props.get("default_generation_settings") or {}).get("n_ctx")
        return [
            ModelInfo(id=m.get("id", "?"), state="loaded", context=n_ctx)
            for m in body.get("data", [])
            if m.get("id")
        ]


class GenericOpenAIServer(LocalServer):
    kind = "openai-compatible"

    def probe(self) -> bool:
        status, _, _ = _http_json(f"{self.base_url}/v1/models", timeout=5)
        return status == 200

    def list_models(self) -> list[ModelInfo]:
        status, body, _ = _http_json(f"{self.base_url}/v1/models", timeout=10)
        if status != 200 or not isinstance(body, dict):
            return []
        return [
            ModelInfo(id=m.get("id", "?"), state="unknown")
            for m in body.get("data", [])
            if m.get("id")
        ]


# --------------------------------------------------------------------------- #
# Автообнаружение
# --------------------------------------------------------------------------- #

# Порт -> класс. Порядок важен: сначала самые вероятные.
KNOWN_PORTS = [
    (1234, LMStudioServer),
    (11434, OllamaServer),
    (8080, LlamaCppServer),
    (8000, GenericOpenAIServer),
    (5000, GenericOpenAIServer),
    (3000, GenericOpenAIServer),
]


def make_server(kind: str, base_url: str, api_key: str = "local", timeout: float = DEFAULT_TIMEOUT) -> LocalServer:
    kinds = {
        "lmstudio": LMStudioServer,
        "bionic": LMStudioServer,
        "ollama": OllamaServer,
        "llamacpp": LlamaCppServer,
        "generic": GenericOpenAIServer,
    }
    cls = kinds.get(kind.lower())
    if cls is None:
        raise ValueError(f"unknown server kind: {kind} (expected: {', '.join(kinds)})")
    return cls(base_url, api_key=api_key, timeout=timeout)


def discover(host: str = "127.0.0.1", extra_ports: list[int] | None = None,
             timeout: float = 3.0, request_timeout: float = DEFAULT_TIMEOUT) -> list[LocalServer]:
    """Сканирует известные порты и возвращает все живые серверы.

    `timeout` — только на проверку порта. У найденных серверов он сразу
    заменяется на `request_timeout`: иначе реальные запросы (а префилл 38k
    токенов идёт десятки секунд) падают с 'timed out'.
    """
    ports = list(KNOWN_PORTS)
    for p in extra_ports or []:
        ports.insert(0, (p, GenericOpenAIServer))

    found: list[LocalServer] = []
    seen: set[int] = set()
    for port, cls in ports:
        if port in seen:
            continue
        seen.add(port)
        srv = cls(f"http://{host}:{port}", timeout=timeout)
        try:
            if srv.probe():
                srv.timeout = request_timeout
                found.append(srv)
        except Exception:
            continue
    return found
