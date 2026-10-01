"""CLI. Точка входа — weds.py в корне проекта."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bench as bench_mod
from . import checks as checks_mod
from . import lmstudio, register as register_mod, report, tune
from .checks import Thresholds
from .servers import discover, make_server

PROG = "weds"


# --------------------------------------------------------------------------- #
# Выбор сервера
# --------------------------------------------------------------------------- #

def resolve_server(args):
    """Явный --server-url или автообнаружение.

    Важно: у серверов из discover() таймаут короткий (нужен только для проверки
    порта). Перед возвратом подменяем его на рабочий, иначе реальные запросы
    падают с 'timed out'.
    """
    if args.server_url:
        kind = args.kind or "generic"
        if kind == "generic":
            from .servers import LMStudioServer, OllamaServer, LlamaCppServer
            probe = LMStudioServer(args.server_url, timeout=5)
            if probe.probe():
                probe.timeout = args.timeout
                return probe
            probe = OllamaServer(args.server_url, timeout=5)
            if probe.probe():
                probe.timeout = args.timeout
                return probe
            probe = LlamaCppServer(args.server_url, timeout=5)
            if probe.probe():
                probe.timeout = args.timeout
                return probe
        srv = make_server(kind, args.server_url, timeout=args.timeout)
        return srv

    found = discover(timeout=3.0)
    if not found:
        return None
    chosen = None
    if args.model:
        needle = args.model.lower()
        for srv in found:
            try:
                if any(needle in m.id.lower() for m in srv.list_models()):
                    chosen = srv
                    break
            except Exception:
                continue
    srv = chosen or found[0]
    srv.timeout = args.timeout
    return srv


def resolve_model(srv, args) -> str | None:
    """Канонический id модели.

    Пользователь вправе передать подстроку ('ornith') — серверу же нужно точное
    имя, иначе 404. Поэтому имя разрешается через список моделей сервера.
    Если ничего не подошло, возвращаем как есть: проверка `model` внятно
    пожалуется, что такой модели нет.
    """
    try:
        models = srv.list_models()
    except Exception:
        return args.model or None
    info = checks_mod.pick_model(models, args.model)
    if info is not None:
        return info.id
    return args.model or None


def need_server(args):
    srv = resolve_server(args)
    if srv is None:
        report.emit(
            {"ok": False, "error": "локальный инференс-сервер не найден",
             "hint": "Запусти LM Studio / Ollama / llama.cpp, либо укажи --server-url"},
            args.json,
            "Локальный сервер не найден. Запусти LM Studio / Ollama / llama.cpp "
            "или укажи --server-url http://127.0.0.1:1234",
        )
        sys.exit(2)
    return srv


# --------------------------------------------------------------------------- #
# Команды
# --------------------------------------------------------------------------- #

def cmd_discover(args) -> int:
    if args.server_url:
        # Явный адрес: проверяем ровно его, а не стандартные порты. Иначе на
        # вопрос «виден ли мой сервер на 8080» команда отвечает про 1234.
        explicit = resolve_server(args)
        found = [explicit] if explicit is not None and explicit.probe() else []
    else:
        found = discover(timeout=3.0)
    payload = {"servers": []}
    for srv in found:
        try:
            models = srv.list_models()
        except Exception:
            models = []
        payload["servers"].append({
            "kind": srv.kind,
            "base_url": srv.base_url,
            "models": len(models),
            "loaded": [m.id for m in models if m.loaded],
        })
    if args.json:
        report.emit(payload, True)
    else:
        if not found:
            if args.server_url:
                print(f"Сервер на {args.server_url} не отвечает.")
                print("Проверь, что он запущен, и что адрес указан с портом.")
            else:
                print("Живых локальных серверов не найдено.")
                print("Проверял порты: 1234 (LM Studio), 11434 (Ollama), 8080 (llama.cpp), "
                      "8000/5000/3000 (generic).")
                print("Если сервер на другом порту или на другой машине — "
                      "укажи адрес: --server-url http://host:port")
        else:
            for s in payload["servers"]:
                print(f"{s['kind']:<18} {s['base_url']:<28} моделей: {s['models']}"
                      + (f", загружено: {', '.join(s['loaded'])}" if s["loaded"] else ""))
    # Код возврата одинаков для обоих режимов вывода: скрипт, читающий --json,
    # должен видеть ту же правду, что и человек.
    return 0 if found else 1


def cmd_models(args) -> int:
    srv = need_server(args)
    models = srv.list_models()
    if args.json:
        report.emit({"server": srv.base_url, "kind": srv.kind,
                     "models": [m.describe() for m in models]}, True)
    else:
        report.print_models(models, f"{srv.base_url} ({srv.kind})")
    return 0


def cmd_check(args) -> int:
    srv = need_server(args)
    model = resolve_model(srv, args)
    if not model:
        report.emit({"ok": False, "error": "не удалось определить модель"},
                    args.json, "Модель не определена. Укажи --model <id>")
        return 2

    th = Thresholds(
        agent_prompt_tokens=args.agent_tokens,
        min_gen_tps=args.min_gen_tps,
    )
    results = checks_mod.run_all(srv, model, th, skip_slow=args.fast)
    summary = checks_mod.summarize(results)
    payload = {
        "ok": summary["verdict"] != "FAIL",
        "model": model,
        "server": srv.base_url,
        "kind": srv.kind,
        "summary": summary,
        "checks": [r.to_dict() for r in results],
    }
    if args.json:
        report.emit(payload, True)
    else:
        report.print_checks(results, summary, title=f"Проверка {model} на {srv.base_url}")
    return 0 if summary["verdict"] != "FAIL" else 1


def cmd_bench(args) -> int:
    srv = need_server(args)
    model = resolve_model(srv, args)
    if not model:
        report.emit({"ok": False, "error": "не удалось определить модель"},
                    args.json, "Модель не определена. Укажи --model <id>")
        return 2

    skip = tuple(s.strip() for s in args.skip.split(",") if s.strip())
    data = bench_mod.run(
        srv, model,
        gen_tokens=args.gen_tokens,
        prefill_tokens=args.prefill_tokens,
        prefix_tokens=args.prefix_tokens,
        skip=skip,
    )
    report.emit(data, args.json)
    return 0


def cmd_logs(args) -> int:
    home = lmstudio.find_home()
    if not home:
        report.emit({"ok": False, "error": "каталог LM Studio не найден"},
                    args.json, "Каталог LM Studio не найден (нет ~/.lmstudio-home-pointer).")
        return 2

    findings = lmstudio.analyze_log(home)
    payload = {
        "home": str(home),
        "log": findings.log_path,
        "context_errors": findings.context_errors[-args.limit:],
        "loads": findings.loads[-args.limit:],
        "estimates": findings.estimates[-args.limit:],
        "memory_model": lmstudio.memory_model(findings),
    }
    if args.json:
        report.emit(payload, True)
    else:
        print(f"Лог: {findings.log_path or '(не найден)'}")
        if findings.context_errors:
            print("\nОшибки переполнения контекста:")
            for e in findings.context_errors[-args.limit:]:
                print(f"  промпт {e['prompt_tokens']} токенов > контекст {e['n_ctx']}"
                      f"  -> поднять контекст минимум до {int(e['prompt_tokens'] * 1.1)}")
        if findings.loads:
            print("\nПоследние загрузки:")
            for l in findings.loads[-args.limit:]:
                print(f"  слотов {l['n_slots']}, контекст {l['n_ctx']}")
        mm = payload["memory_model"]
        if mm:
            print("\nВес модели по данным движка: "
                  f"{round(mm['weights_bytes'] / tune.GIB, 2)} ГиБ")
    return 0


def cmd_tune(args) -> int:
    home = lmstudio.find_home()
    if not home:
        report.emit({"ok": False, "error": "каталог LM Studio не найден"},
                    args.json, "Каталог LM Studio не найден.")
        return 2

    model = args.model
    if not model:
        code, out = lmstudio.run_lms(home, ["ls"], timeout=60)
        report.emit({"ok": False, "error": "нужен --model <key>",
                     "available": out.strip()[:800]},
                    args.json, "Укажи --model <key>. Доступные модели: weds models")
        return 2

    hw = tune.detect_hardware()
    vram = args.vram_gib * tune.GIB if args.vram_gib else hw.vram_bytes
    budget = args.budget_gib * tune.GIB if args.budget_gib else None

    rec = tune.recommend(
        home, model,
        vram_bytes=int(vram) if vram else None,
        budget_bytes=int(budget) if budget else None,
        agent_prompt_tokens=args.agent_tokens,
        context=args.context,
        prefer_q8=args.prefer_q8,
        offload_kv=False if args.kv_in_ram else None,
        cpu_threads=hw.cpu_physical_cores,
    )

    payload = {"ok": True, "hardware": hw.describe(), "recommendation": rec.describe(),
               "applied": False}

    if args.apply:
        result = tune.apply(home, rec)
        payload["applied"] = result
        payload["ok"] = result.get("ok", False)

    if args.json:
        report.emit(payload, True)
    else:
        print("Железо:")
        report.print_kv(hw.describe(), indent=2)
        print()
        if rec.curve:
            report.print_curve(
                rec.curve.describe(),
                round(rec.curve.total_for(rec.context, rec.k_cache_quant) / tune.GIB, 2),
            )
            print()
        print("Рекомендация:")
        report.print_kv(rec.describe(), indent=2)
        if not args.apply:
            print("\nПрименить: добавь --apply")
    return 0 if payload["ok"] else 1


def cmd_restore(args) -> int:
    home = lmstudio.find_home()
    if not home:
        report.emit({"ok": False, "error": "каталог LM Studio не найден"}, args.json)
        return 2
    cfg = lmstudio.find_config_by_key(home, args.model)
    if cfg is None:
        report.emit({"ok": False, "error": f"конфиг '{args.model}' не найден"}, args.json)
        return 2
    result = tune.restore(cfg)
    report.emit(result, args.json)
    return 0 if result.get("ok") else 1


def cmd_register(args) -> int:
    srv = need_server(args)
    model = resolve_model(srv, args)
    if not model:
        report.emit({"ok": False, "error": "не удалось определить модель"}, args.json)
        return 2

    entry = register_mod.build_entry(
        model, srv.base_url,
        max_input=args.max_input,
        name=args.name,
    )
    payload = {"entry": entry, "targets": []}

    for target in (args.target or ["workbuddy"]):
        if target == "env":
            payload["env"] = register_mod.env_snippet(model, srv.base_url)
            payload["openai_snippet"] = register_mod.openai_client_snippet(model, srv.base_url)
            continue
        if target == "print":
            continue
        res = register_mod.register(target, entry, dry_run=args.dry_run)
        payload["targets"].append(res.describe())

    if args.json:
        report.emit(payload, True)
    else:
        if "env" in payload:
            print("Переменные окружения:")
            print(payload["env"])
            print()
            print("Пример на openai-python:")
            print(payload["openai_snippet"])
            print()
        for t in payload["targets"]:
            if args.dry_run:
                verb = "будет добавлена" if t["added"] else "будет обновлена"
            else:
                verb = "добавлена" if t["added"] else "обновлена"
            print(f"{t['target']}: модель {verb} в {t['path']}")
            if t["note"]:
                print(f"  ! {t['note']}")
        if args.dry_run:
            print("\nЭто был --dry-run, файл не изменён.")
        if not payload["targets"] and "env" not in payload:
            print("Запись не выполнена (--dry-run).")
    return 0


def cmd_load(args) -> int:
    home = lmstudio.find_home()
    if not home:
        report.emit({"ok": False, "error": "каталог LM Studio не найден"}, args.json)
        return 2
    if args.unload:
        code, out = lmstudio.unload_all(home)
        report.emit({"ok": code == 0, "output": out.strip()[-500:]}, args.json,
                    out.strip()[-500:] if out.strip() else "Выгружено.")
        return 0 if code == 0 else 1

    code, out = lmstudio.load_model(
        home, args.model, context=args.context, gpu=args.gpu,
        parallel=args.parallel, ttl=args.ttl,
    )
    tail = "\n".join(l for l in out.replace("\r", "\n").splitlines()
                     if l.strip() and "Loading" not in l)[-600:]
    report.emit({"ok": code == 0, "model": args.model, "output": tail}, args.json, tail)
    return 0 if code == 0 else 1


def cmd_gguf(args) -> int:
    from . import gguf as gguf_mod

    path = Path(args.path)
    files = gguf_mod.find_ggufs(path)
    if not files:
        report.emit({"ok": False, "error": f"GGUF не найдены в {path}"}, args.json,
                    f"GGUF не найдены в {path}")
        return 1

    out = []
    for f in files:
        try:
            info = gguf_mod.parse(f)
            out.append(info.describe())
        except (ValueError, OSError, Exception) as e:  # noqa: BLE001
            out.append({"path": str(f), "error": str(e)})

    report.emit({"files": out}, args.json)
    return 0


def cmd_doctor(args) -> int:
    """Полная диагностика: серверы, модель, логи, проверки."""
    payload: dict = {"servers": [], "log": None, "checks": None}
    exit_code = 0

    if args.server_url:
        explicit = resolve_server(args)
        found = [explicit] if explicit is not None and explicit.probe() else []
    else:
        found = discover(timeout=3.0)
    for srv in found:
        try:
            models = srv.list_models()
        except Exception:
            models = []
        payload["servers"].append({
            "kind": srv.kind, "base_url": srv.base_url,
            "models": [m.describe() for m in models],
        })

    home = lmstudio.find_home()
    if home:
        findings = lmstudio.analyze_log(home)
        payload["log"] = {
            "path": findings.log_path,
            "context_errors": findings.context_errors[-3:],
            "last_load": findings.loads[-1] if findings.loads else None,
            "memory_model": lmstudio.memory_model(findings),
        }

    if found:
        srv = found[0]
        model = resolve_model(srv, args)
        if model:
            th = Thresholds(agent_prompt_tokens=args.agent_tokens)
            results = checks_mod.run_all(srv, model, th, skip_slow=args.fast)
            summary = checks_mod.summarize(results)
            payload["checks"] = {"model": model, "summary": summary,
                                 "results": [r.to_dict() for r in results]}
            if summary["verdict"] == "FAIL":
                exit_code = 1
    else:
        exit_code = 2

    if args.json:
        report.emit(payload, True)
        return exit_code

    if not found:
        print("Локальный сервер не найден. Запусти LM Studio / Ollama / llama.cpp.")
    for s in payload["servers"]:
        loaded = [m["id"] for m in s["models"] if m["state"] == "loaded"]
        print(f"Сервер: {s['base_url']} ({s['kind']}), моделей {len(s['models'])}"
              + (f", загружено: {', '.join(loaded)}" if loaded else ", ничего не загружено"))

    lg = payload["log"]
    if lg:
        print(f"\nЛог движка: {lg['path']}")
        if lg["context_errors"]:
            print("  Найдены ошибки переполнения контекста:")
            for e in lg["context_errors"]:
                print(f"    промпт {e['prompt_tokens']} > контекст {e['n_ctx']}"
                      f" -> нужно минимум {int(e['prompt_tokens'] * 1.1)}")
        else:
            print("  Ошибок переполнения контекста нет.")
        if lg["memory_model"]:
            print(f"  Вес модели: {round(lg['memory_model']['weights_bytes'] / tune.GIB, 2)} ГиБ")

    ch = payload["checks"]
    if ch:
        print()
        report.print_checks(
            [checks_mod.CheckResult(r["check"], r["status"], r["summary"],
                                    r.get("detail", {}), r.get("fix"))
             for r in ch["results"]],
            ch["summary"],
            title=f"Проверка {ch['model']}",
        )
    return exit_code


def cmd_mcp(args) -> int:
    from .mcp import serve
    return serve()


# --------------------------------------------------------------------------- #
# Парсер
# --------------------------------------------------------------------------- #

COMMON_HELP = {
    "json": "машинный вывод",
    "server_url": "адрес сервера, например http://127.0.0.1:1234",
    "kind": "тип сервера (иначе определяется автоматически)",
    "model": "id модели",
    "timeout": "таймаут запроса, сек",
    "agent_tokens": "размер системного промпта агента в токенах (по умолчанию 38000)",
}


def _add_common(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """Общие флаги. С suppress=True значения не перетирают уже разобранные
    на верхнем уровне — так флаги работают и до, и после подкоманды."""
    def add(name, *flags, **kw):
        if suppress:
            kw["default"] = argparse.SUPPRESS
        parser.add_argument(*flags, **kw)

    add("json", "--json", action="store_true", help=COMMON_HELP["json"])
    add("server_url", "--server-url", help=COMMON_HELP["server_url"])
    add("kind", "--kind", choices=["lmstudio", "ollama", "llamacpp", "generic"],
        help=COMMON_HELP["kind"])
    add("model", "--model", help=COMMON_HELP["model"])
    add("timeout", "--timeout", type=float, default=900.0, help=COMMON_HELP["timeout"])
    add("agent_tokens", "--agent-tokens", type=int, default=38000,
        help=COMMON_HELP["agent_tokens"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Проверка локальных LLM на готовность к агентской работе.",
    )
    _add_common(p)

    sub = p.add_subparsers(dest="command", required=True)

    def new(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_, parents=[_common_sub()])

    sp = new("discover", "найти живые локальные серверы")
    sp.set_defaults(func=cmd_discover)

    sp = new("models", "список моделей и их состояние")
    sp.set_defaults(func=cmd_models)

    sp = new("check", "проверки готовности к агентской работе")
    sp.add_argument("--fast", action="store_true", help="без тяжёлых проверок")
    sp.add_argument("--min-gen-tps", type=float, default=12.0,
                    help="порог приемлемой скорости генерации")
    sp.set_defaults(func=cmd_check)

    sp = new("bench", "замеры скорости")
    sp.add_argument("--gen-tokens", type=int, default=128)
    sp.add_argument("--prefill-tokens", type=int, default=8000)
    sp.add_argument("--prefix-tokens", type=int, default=6000)
    sp.add_argument("--skip", default="", help="пропустить: gen,prefill,prefix")
    sp.set_defaults(func=cmd_bench)

    sp = new("logs", "разбор логов движка LM Studio")
    sp.add_argument("--limit", type=int, default=5)
    sp.set_defaults(func=cmd_logs)

    sp = new("tune", "подобрать и применить параметры загрузки")
    sp.add_argument("--context", type=int, help="задать контекст вручную")
    sp.add_argument("--vram-gib", type=float, help="бюджет VRAM, если не определяется")
    sp.add_argument("--budget-gib", type=float,
                    help="бюджет памяти под модель напрямую, минуя расчёт "
                         "«доля VRAM минус резерв под рабочий стол»")
    sp.add_argument("--prefer-q8", action="store_true", help="предпочесть q8_0 KV-кеш")
    sp.add_argument("--kv-in-ram", action="store_true",
                    help="держать KV-кеш в системной RAM, а не в VRAM "
                         "(медленнее, но экономит видеопамять)")
    sp.add_argument("--apply", action="store_true", help="записать конфиг (с бэкапом)")
    sp.set_defaults(func=cmd_tune)

    sp = new("restore", "откатить конфиг из бэкапа")
    sp.set_defaults(func=cmd_restore)

    sp = new("register", "прописать модель в конфиг агентской обвязки")
    sp.add_argument("--target", action="append",
                    choices=["workbuddy", "codebuddy", "generic", "env"],
                    help="куда прописать (можно несколько раз); env — вывести переменные")
    sp.add_argument("--max-input", type=int, default=49152)
    sp.add_argument("--name", help="отображаемое имя модели")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_register)

    sp = new("load", "загрузить/выгрузить модель через lms")
    sp.add_argument("--context", type=int)
    sp.add_argument("--gpu", help="off | max | 0..1")
    sp.add_argument("--parallel", type=int)
    sp.add_argument("--ttl", type=int)
    sp.add_argument("--unload", action="store_true", help="выгрузить всё")
    sp.set_defaults(func=cmd_load)

    sp = new("gguf", "метаданные GGUF")
    sp.add_argument("path", help="файл или каталог")
    sp.set_defaults(func=cmd_gguf)

    sp = new("doctor", "полная диагностика одной командой")
    sp.add_argument("--fast", action="store_true")
    sp.set_defaults(func=cmd_doctor)

    sp = new("mcp", "запустить MCP-сервер (stdio)")
    sp.set_defaults(func=cmd_mcp)

    return p


_COMMON_SUB: argparse.ArgumentParser | None = None


def _common_sub() -> argparse.ArgumentParser:
    """Заготовка с общими флагами для parents=[...] у подкоманд."""
    global _COMMON_SUB
    if _COMMON_SUB is None:
        holder = argparse.ArgumentParser(add_help=False)
        _add_common(holder, suppress=True)
        _COMMON_SUB = holder
    return _COMMON_SUB


def main(argv: list[str] | None = None) -> int:
    report.enable_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"Ошибка: {e}", file=sys.stderr)
        return 1
