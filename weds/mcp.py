"""MCP-сервер (stdio) — чтобы проверку можно было звать из любого агента.

Реализован минимальный корректный набор JSON-RPC методов:
initialize, tools/list, tools/call, ping. Всё, что не протокол, идёт в stderr,
потому что stdout — это канал протокола.

Подключение:
    Claude Code / Cursor / Codex / WorkBuddy -> MCP-сервер, команда:
    python <путь>/Wednesday/weds.py mcp
"""

from __future__ import annotations

import json
import sys
import traceback

from . import bench as bench_mod
from . import checks as checks_mod
from . import lmstudio, register as register_mod, tune
from .checks import Thresholds
from .servers import discover, make_server

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "wednesday"
SERVER_VERSION = "1.0.0"

TOOLS = [
    {
        "name": "local_llm_discover",
        "description": "Найти живые локальные инференс-серверы (LM Studio/Bionic, Ollama, "
                       "llama.cpp) на стандартных портах и перечислить их модели.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_llm_list_models",
        "description": "Список моделей локального сервера с состоянием загрузки, "
                       "загруженным контекстом и квантизацией.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "server_url": {"type": "string", "description": "например http://127.0.0.1:1234"},
                "kind": {"type": "string", "enum": ["lmstudio", "ollama", "llamacpp", "generic"]},
            },
        },
    },
    {
        "name": "local_llm_check",
        "description": "Проверить, готова ли локальная модель к агентской работе: влезает ли "
                       "системный промпт агента в контекст, работает ли tool-calling, "
                       "валиден ли JSON аргументов, работает ли кэш префикса, какая скорость.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "server_url": {"type": "string"},
                "kind": {"type": "string", "enum": ["lmstudio", "ollama", "llamacpp", "generic"]},
                "agent_tokens": {"type": "integer",
                                 "description": "размер промпта агента в токенах (WorkBuddy ~38000)"},
                "fast": {"type": "boolean", "description": "без тяжёлых проверок"},
            },
        },
    },
    {
        "name": "local_llm_bench",
        "description": "Замерить скорость: генерация tok/s, префилл tok/s, ускорение от кэша "
                       "префикса. Тяжёлая операция — гонять по одному прогону.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "server_url": {"type": "string"},
                "gen_tokens": {"type": "integer"},
                "prefill_tokens": {"type": "integer"},
                "prefix_tokens": {"type": "integer"},
            },
        },
    },
    {
        "name": "local_llm_tune",
        "description": "Подобрать параметры загрузки под железо (контекст, GPU-оффлоад, "
                       "KV-кеш, батч) и, если apply=true, записать их в per-model конфиг "
                       "LM Studio с бэкапом.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "ключ модели из lms ls"},
                "vram_gib": {"type": "number", "description": "вся VRAM карты"},
                "budget_gib": {"type": "number",
                               "description": "бюджет памяти под модель напрямую; "
                                              "обход пессимистичной проекции оценщика"},
                "agent_tokens": {"type": "integer"},
                "context": {"type": "integer"},
                "prefer_q8": {"type": "boolean"},
                "kv_in_ram": {"type": "boolean",
                              "description": "держать KV-кеш в системной RAM вместо VRAM"},
                "apply": {"type": "boolean", "description": "записать конфиг"},
            },
            "required": ["model"],
        },
    },
    {
        "name": "local_llm_register",
        "description": "Прописать локальную модель в конфиг агентской обвязки "
                       "(workbuddy/codebuddy — models.json, env — переменные окружения).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "server_url": {"type": "string"},
                "target": {"type": "array", "items": {"type": "string",
                            "enum": ["workbuddy", "codebuddy", "generic", "env"]}},
                "max_input": {"type": "integer"},
                "name": {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
        },
    },
    {
        "name": "local_llm_logs",
        "description": "Разобрать логи движка LM Studio: ошибки переполнения контекста, "
                       "реально применённые параметры загрузки, вес модели.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
    {
        "name": "local_llm_doctor",
        "description": "Полная диагностика одной командой: серверы, модели, логи, набор проверок.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "agent_tokens": {"type": "integer"},
                "fast": {"type": "boolean"},
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Обработчики
# --------------------------------------------------------------------------- #

def _resolve(args: dict):
    url = args.get("server_url")
    kind = args.get("kind")
    if url:
        return make_server(kind or "generic", url, timeout=900.0)
    found = discover(timeout=3.0)
    if not found:
        raise RuntimeError(
            "локальный инференс-сервер не найден; запусти LM Studio / Ollama / llama.cpp "
            "или передай server_url"
        )
    chosen = None
    model = args.get("model")
    if model:
        needle = model.lower()
        for srv in found:
            try:
                if any(needle in m.id.lower() for m in srv.list_models()):
                    chosen = srv
                    break
            except Exception:
                continue
    srv = chosen or found[0]
    # У серверов из discover() таймаут короткий — только для проверки порта.
    srv.timeout = 900.0
    return srv


def _pick(srv, args) -> str:
    if args.get("model"):
        return args["model"]
    info = checks_mod.pick_model(srv.list_models(), None)
    if not info:
        raise RuntimeError("на сервере нет моделей")
    return info.id


def h_discover(args: dict) -> dict:
    out = []
    for srv in discover(timeout=3.0):
        try:
            models = srv.list_models()
        except Exception:
            models = []
        out.append({"kind": srv.kind, "base_url": srv.base_url,
                    "models": [m.describe() for m in models]})
    return {"servers": out, "found": len(out)}


def h_list_models(args: dict) -> dict:
    srv = _resolve(args)
    return {"server": srv.base_url, "kind": srv.kind,
            "models": [m.describe() for m in srv.list_models()]}


def h_check(args: dict) -> dict:
    srv = _resolve(args)
    model = _pick(srv, args)
    th = Thresholds(agent_prompt_tokens=int(args.get("agent_tokens", 38000)))
    results = checks_mod.run_all(srv, model, th, skip_slow=bool(args.get("fast")))
    summary = checks_mod.summarize(results)
    return {"model": model, "server": srv.base_url, "summary": summary,
            "checks": [r.to_dict() for r in results]}


def h_bench(args: dict) -> dict:
    srv = _resolve(args)
    model = _pick(srv, args)
    return bench_mod.run(
        srv, model,
        gen_tokens=int(args.get("gen_tokens", 128)),
        prefill_tokens=int(args.get("prefill_tokens", 8000)),
        prefix_tokens=int(args.get("prefix_tokens", 6000)),
    )


def h_tune(args: dict) -> dict:
    home = lmstudio.find_home()
    if not home:
        raise RuntimeError("каталог LM Studio не найден")
    hw = tune.detect_hardware()
    vram = args.get("vram_gib")
    vram_bytes = int(float(vram) * tune.GIB) if vram else hw.vram_bytes
    budget = args.get("budget_gib")
    rec = tune.recommend(
        home, args["model"],
        vram_bytes=vram_bytes,
        budget_bytes=int(float(budget) * tune.GIB) if budget else None,
        agent_prompt_tokens=int(args.get("agent_tokens", 38000)),
        context=args.get("context"),
        prefer_q8=bool(args.get("prefer_q8")),
        offload_kv=False if args.get("kv_in_ram") else None,
        cpu_threads=hw.cpu_physical_cores,
    )
    out = {"hardware": hw.describe(), "recommendation": rec.describe(), "applied": None}
    if args.get("apply"):
        out["applied"] = tune.apply(home, rec)
    return out


def h_register(args: dict) -> dict:
    srv = _resolve(args)
    model = _pick(srv, args)
    entry = register_mod.build_entry(
        model, srv.base_url,
        max_input=int(args.get("max_input", 49152)),
        name=args.get("name"),
    )
    out: dict = {"entry": entry, "targets": []}
    for target in (args.get("target") or ["workbuddy"]):
        if target == "env":
            out["env"] = register_mod.env_snippet(model, srv.base_url)
            continue
        out["targets"].append(
            register_mod.register(target, entry, dry_run=bool(args.get("dry_run"))).describe()
        )
    return out


def h_logs(args: dict) -> dict:
    home = lmstudio.find_home()
    if not home:
        raise RuntimeError("каталог LM Studio не найден")
    f = lmstudio.analyze_log(home)
    limit = int(args.get("limit", 5))
    return {"home": str(home), "log": f.log_path,
            "context_errors": f.context_errors[-limit:],
            "loads": f.loads[-limit:],
            "memory_model": lmstudio.memory_model(f)}


def h_doctor(args: dict) -> dict:
    out: dict = {"servers": [], "log": None, "checks": None}
    found = discover(timeout=3.0)
    for srv in found:
        try:
            models = srv.list_models()
        except Exception:
            models = []
        out["servers"].append({"kind": srv.kind, "base_url": srv.base_url,
                               "models": [m.describe() for m in models]})
    home = lmstudio.find_home()
    if home:
        f = lmstudio.analyze_log(home)
        out["log"] = {"path": f.log_path, "context_errors": f.context_errors[-3:],
                      "memory_model": lmstudio.memory_model(f)}
    if found:
        try:
            srv = _resolve(args)
            model = _pick(srv, args)
            th = Thresholds(agent_prompt_tokens=int(args.get("agent_tokens", 38000)))
            results = checks_mod.run_all(srv, model, th, skip_slow=bool(args.get("fast")))
            out["checks"] = {"model": model, "summary": checks_mod.summarize(results),
                             "results": [r.to_dict() for r in results]}
        except Exception as e:  # noqa: BLE001
            out["checks_error"] = str(e)
    return out


HANDLERS = {
    "local_llm_discover": h_discover,
    "local_llm_list_models": h_list_models,
    "local_llm_check": h_check,
    "local_llm_bench": h_bench,
    "local_llm_tune": h_tune,
    "local_llm_register": h_register,
    "local_llm_logs": h_logs,
    "local_llm_doctor": h_doctor,
}


# --------------------------------------------------------------------------- #
# Протокол
# --------------------------------------------------------------------------- #

def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _result(msg_id, result) -> None:
    _send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def _error(msg_id, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})


def handle(msg: dict) -> None:
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # Уведомления ответа не требуют.
    if msg_id is None and method and method.startswith("notifications/"):
        return

    if method == "initialize":
        _result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
        return

    if method == "ping":
        _result(msg_id, {})
        return

    if method == "tools/list":
        _result(msg_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if not handler:
            _error(msg_id, -32602, f"unknown tool: {name}")
            return
        try:
            data = handler(args)
            _result(msg_id, {
                "content": [{"type": "text",
                             "text": json.dumps(data, indent=2, ensure_ascii=False)}],
                "isError": False,
            })
        except Exception as e:  # noqa: BLE001
            _result(msg_id, {
                "content": [{"type": "text", "text": f"Ошибка: {e}"}],
                "isError": True,
            })
        return

    if msg_id is not None:
        _error(msg_id, -32601, f"method not found: {method}")


def serve() -> int:
    """Читает JSON-RPC построчно из stdin, отвечает в stdout."""
    print(f"[{SERVER_NAME}] MCP-сервер запущен, инструментов: {len(TOOLS)}", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"[{SERVER_NAME}] не JSON: {line[:200]}", file=sys.stderr)
            continue
        try:
            handle(msg)
        except Exception:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
    return 0
