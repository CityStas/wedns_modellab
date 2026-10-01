#!/usr/bin/env python3
"""Дымовые тесты Wednesday.

Проверяют то, что не требует живого инференс-сервера: разбор GGUF, парсинг
конфигов, логику подбора контекста, протокол MCP, разбор ошибок из логов.

Запуск:
    python tests/smoke_test.py

Живой сервер не нужен. Тесты, требующие сервера, помечены и пропускаются,
если порт 1234 не отвечает.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from weds import checks, lmstudio, register, tune  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"[ OK ] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def server_up(port: int = 1234) -> bool:
    with socket.socket() as s:
        s.settimeout(1.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# --------------------------------------------------------------------------- #

def test_gguf_parser() -> None:
    """Парсер GGUF на синтетическом файле — живая модель не нужна."""
    import struct

    def s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    # Минимальный GGUF: магия, версия, 0 тензоров, 3 пары ключ-значение.
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 3)
    blob += s("general.architecture") + struct.pack("<I", 8) + s("qwen35")
    blob += s("qwen35.block_count") + struct.pack("<I", 4) + struct.pack("<I", 32)
    blob += s("qwen35.full_attention_interval") + struct.pack("<I", 4) + struct.pack("<I", 4)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.gguf"
        p.write_bytes(blob)
        from weds import gguf
        info = gguf.parse(p)
        check("gguf: архитектура читается", info.arch == "qwen35", info.arch)
        check("gguf: block_count = 32", info.get("block_count") == 32)
        check("gguf: гибрид распознан", info.is_hybrid)
        check("gguf: слоёв с полным attention = 8", info.full_attention_layers == 8,
              str(info.full_attention_layers))

        bad = Path(td) / "bad.gguf"
        bad.write_bytes(b"NOTGGUF" + b"\x00" * 32)
        try:
            gguf.parse(bad)
            check("gguf: мусорный файл отвергается", False, "исключение не брошено")
        except ValueError:
            check("gguf: мусорный файл отвергается", True)


def test_context_overflow_parsing() -> None:
    """Разбор сообщения 'request (N tokens) exceeds the available context size (M)'."""
    line = ('send_error: task id = 0, error: request (37955 tokens) exceeds '
            'the available context size (32768 tokens), try increasing it')
    m = lmstudio._CTX_ERR.search(line)
    check("лог: ошибка переполнения парсится", bool(m))
    if m:
        check("лог: числа верные", (int(m.group(1)), int(m.group(2))) == (37955, 32768),
              f"{m.group(1)}/{m.group(2)}")


def test_estimate_parsing() -> None:
    """Разбор вывода lms --estimate-only с неразрывными пробелами."""
    sample = (
        "Model: test\n"
        "Context Length: 32\u00a0768\n"
        "GPU Offload: 100%\n"
        "Estimated GPU Memory:   7.28 GiB\n"
        "Estimated Total Memory: 7.28 GiB\n"
    )
    parsed = tune._parse_estimate(sample)
    check("tune: оценка парсится", parsed is not None, str(parsed))
    if parsed:
        ctx, total = parsed
        check("tune: контекст 32768 с неразрывным пробелом", ctx == 32768, str(ctx))
        check("tune: 7.28 GiB в байтах", abs(total - int(7.28 * tune.GIB)) < 1024)


def test_memory_curve() -> None:
    """Кривая памяти: вес + цена контекста из двух замеров."""
    curve = tune.MemoryCurve(
        weights_bytes=int(5.8 * tune.GIB),
        bytes_per_token=47 * 1024,
        samples=[(8192, int(6.17 * tune.GIB)), (32768, int(7.28 * tune.GIB))],
    )
    q8 = curve.max_context(int(6.5 * tune.GIB), "q8_0")
    q4 = curve.max_context(int(6.5 * tune.GIB), "q4_0")
    check("tune: q4_0 даёт вдвое больший контекст, чем q8_0", q4 > q8 * 1.9,
          f"q4={q4} q8={q8}")
    check("tune: проекция считается по кванту",
          curve.total_for(32768, "q4_0") < curve.total_for(32768, "q8_0"))


def test_config_roundtrip() -> None:
    """Чтение и запись per-model конфига с сохранением чужих полей."""
    original = {
        "preset": "",
        "operation": {"fields": [{"key": "llm.prediction.temperature", "value": 0.6}]},
        "load": {"fields": [
            {"key": "llm.load.contextLength", "value": 32768},
            {"key": "llm.load.llama.kCacheQuantizationType",
             "value": {"checked": True, "value": "q8_0"}},
        ]},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.gguf.json"
        p.write_text(json.dumps(original), encoding="utf-8")

        cfg = lmstudio.read_config(p)
        check("конфиг: чтение contextLength", cfg.get_load_simple("context") == 32768)
        check("конфиг: развёртка {checked,value}",
              cfg.get_load_simple("k_cache_quant") == "q8_0")

        backup = lmstudio.write_config(p, load={"context": 49152, "k_cache_quant": "q4_0"})
        check("конфиг: бэкап создан", backup is not None and backup.is_file())

        cfg2 = lmstudio.read_config(p)
        check("конфиг: contextLength обновлён", cfg2.get_load_simple("context") == 49152)
        check("конфиг: квант KV обновлён", cfg2.get_load_simple("k_cache_quant") == "q4_0")
        check("конфиг: обёртка {checked,value} сохранена",
              isinstance(cfg2.get_load("k_cache_quant"), dict))
        check("конфиг: чужие поля не потеряны",
              cfg2.get_op("temperature") == 0.6)

        restored = tune.restore(p)
        check("конфиг: откат из бэкапа", restored.get("ok") is True)
        cfg3 = lmstudio.read_config(p)
        check("конфиг: после отката снова 32768",
              cfg3.get_load_simple("context") == 32768)


def test_backup_not_overwritten() -> None:
    """Повторная запись не должна перетирать бэкап.

    Иначе второй `--apply` сохраняет промежуточное состояние, оригинал теряется,
    и `restore` откатывает не туда, куда рассчитывает пользователь.
    """
    original = {
        "preset": "",
        "operation": {"fields": []},
        "load": {"fields": [{"key": "llm.load.contextLength", "value": 32768}]},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.gguf.json"
        p.write_text(json.dumps(original), encoding="utf-8")

        lmstudio.write_config(p, load={"context": 49152})
        lmstudio.write_config(p, load={"context": 98304})
        lmstudio.write_config(p, load={"context": 131072})

        bk = p.with_suffix(p.suffix + ".weds-backup")
        saved = json.loads(bk.read_text(encoding="utf-8"))
        vals = {f["key"]: f["value"] for f in saved["load"]["fields"]}
        check("конфиг: бэкап хранит исходное состояние, а не промежуточное",
              vals.get("llm.load.contextLength") == 32768,
              str(vals.get("llm.load.contextLength")))

        check("конфиг: текущее значение всё равно применилось",
              lmstudio.read_config(p).get_load_simple("context") == 131072)

        tune.restore(p)
        check("конфиг: restore возвращает именно оригинал",
              lmstudio.read_config(p).get_load_simple("context") == 32768)


def test_register_no_available_models() -> None:
    """Регистрация не должна добавлять availableModels — он режет дропдаун."""
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "models.json"
        dest.write_text(json.dumps({"models": [
            {"id": "cloud-model", "url": "https://example.com/v1/chat/completions"}
        ]}), encoding="utf-8")

        entry = register.build_entry("local-model", "http://127.0.0.1:1234")
        register.register("generic", entry, path=dest)

        data = json.loads(dest.read_text(encoding="utf-8"))
        ids = [m["id"] for m in data["models"]]
        check("register: облачная модель не потеряна", "cloud-model" in ids, str(ids))
        check("register: локальная добавлена", "local-model" in ids)
        check("register: availableModels не появился", "availableModels" not in data)
        check("register: url дополнен до /chat/completions",
              entry["url"].endswith("/v1/chat/completions"), entry["url"])

        # Повторная регистрация не должна дублировать запись.
        register.register("generic", entry, path=dest)
        data2 = json.loads(dest.read_text(encoding="utf-8"))
        check("register: повтор не дублирует",
              len([m for m in data2["models"] if m["id"] == "local-model"]) == 1)


def test_build_prompt_nonce() -> None:
    """Nonce обязан менять начало промпта — иначе замеры попадают в чужой кэш."""
    a = checks.build_prompt(4.0, 1000, nonce="aaaa")
    b = checks.build_prompt(4.0, 1000, nonce="bbbb")
    plain = checks.build_prompt(4.0, 1000)
    check("промпт: nonce меняет начало", not a.startswith(plain[:20]))
    check("промпт: разные nonce дают разные промпты", a != b)
    check("промпт: размер примерно соответствует", abs(len(a) - 4000) < 400, str(len(a)))
    check("промпт: без nonce начинается с filler", plain.startswith(checks.FILLER[:20]))


def test_summarize() -> None:
    results = [
        checks.CheckResult("a", checks.PASS, ""),
        checks.CheckResult("b", checks.WARN, ""),
    ]
    s = checks.summarize(results)
    check("сводка: WARN при отсутствии FAIL", s["verdict"] == "WARN", str(s))
    results.append(checks.CheckResult("c", checks.FAIL, ""))
    check("сводка: FAIL перебивает всё", checks.summarize(results)["verdict"] == "FAIL")


def test_hardware_probe() -> None:
    """Детект железа. Главное — VRAM должна определяться даже без NVML:
    nvidia-smi в песочницах падает, и без обхода через реестр tune слепо
    занижает контекст."""
    hw = tune.detect_hardware()
    check("железо: физические ядра определены",
          hw.cpu_physical_cores is None or hw.cpu_physical_cores > 0,
          str(hw.cpu_physical_cores))
    check("железо: RAM определена", hw.ram_bytes is None or hw.ram_bytes > 0,
          str(hw.ram_bytes))

    if os.name == "nt":
        name, vram = tune._vram_from_registry()
        check("железо: VRAM читается из реестра", vram is not None and vram > 0,
              f"{name} / {vram}")
        if vram:
            check("железо: VRAM кратна ГиБ (64-битное поле, не обрезано)",
                  vram % (1024 ** 3) == 0 or vram % (256 * 1024 ** 2) == 0,
                  str(vram))
        check("железо: источник VRAM заполнен",
              hw.vram_bytes is None or hw.source != "unknown", hw.source)


def test_budget_override() -> None:
    """Явный бюджет должен обходить расчёт «доля VRAM минус резерв» и давать
    контекст не меньше, чем консервативная проекция.

    Проверяется на синтетической кривой — детерминированно и без вызовов lms.
    """
    curve = tune.MemoryCurve(
        weights_bytes=int(5.8 * tune.GIB),
        bytes_per_token=int(47.4 * 1024),
        samples=[(8192, int(6.17 * tune.GIB)), (32768, int(7.28 * tune.GIB))],
    )

    # 8 ГиБ карта: бюджет по умолчанию = 8 * 0.92 - 0.8 = 6.56 ГиБ.
    default_budget = int(8 * tune.GIB * tune.BUDGET_FRACTION) - tune.DESKTOP_RESERVE_BYTES
    fits_default = curve.max_context(default_budget, "q4_0")
    fits_manual = curve.max_context(int(7 * tune.GIB), "q4_0")

    check("tune: ручной бюджет даёт больше контекста, чем расчётный",
          fits_manual > fits_default, f"{fits_default} -> {fits_manual}")
    check("tune: q4_0 вдвое дешевле q8_0",
          abs(curve.effective_bytes_per_token("q4_0") * 2
              - curve.effective_bytes_per_token("q8_0")) <= 2,
          f"{curve.effective_bytes_per_token('q4_0')} / "
          f"{curve.effective_bytes_per_token('q8_0')}")
    check("tune: F16 вдвое дороже q8_0",
          abs(curve.effective_bytes_per_token("F16")
              - curve.effective_bytes_per_token("q8_0") * 2) <= 2)

    # recommend() без lms отдаёт контекст по промпту агента, а не мусор.
    hw = tune.detect_hardware()
    check("железо: VRAM определена или честно помечена как неизвестная",
          hw.vram_bytes is None or hw.source != "unknown", hw.source)


def test_cli_accepts_budget_flag() -> None:
    """--budget-gib должен быть в CLI: это основной обход пессимистичной проекции."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "weds.py"), "tune", "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    check("cli: tune знает --budget-gib", "--budget-gib" in proc.stdout)
    check("cli: tune знает --vram-gib", "--vram-gib" in proc.stdout)


def test_explicit_server_url_wins() -> None:
    """Явный --server-url должен проверяться вместо стандартных портов.

    Без этого `discover --server-url http://host:8080` отвечает про сервер на
    1234 — и человек делает вывод, что его сервер виден, хотя это не так.
    """
    for cmd in ("discover", "doctor"):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "weds.py"), cmd, "--json",
             "--server-url", "http://127.0.0.1:59998",
             *(["--fast"] if cmd == "doctor" else [])],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        )
        check(f"cli: {cmd} уважает --server-url (мёртвый порт -> код 2 или 1)",
              proc.returncode in (1, 2), f"rc={proc.returncode}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            data = None
        check(f"cli: {cmd} не подсунул сервер по умолчанию",
              data is not None and data.get("servers") == [], str(proc.stdout[:200]))


def test_offload_kv() -> None:
    """KV-кеш в VRAM — самый дешёвый способ ускорить генерацию, и он должен
    попадать в конфиг. Замерено: 34 → 47 tok/s на 9B Q4_K_M."""
    check("tune: offload_kv маппится в ключ LM Studio",
          lmstudio.LOAD_KEYS.get("offload_kv") == "llm.load.offloadKVCacheToGpu")

    rec = tune.Recommendation(model_key="m", context=49152, vram_budget_bytes=tune.GIB)
    check("tune: offload_kv включён по умолчанию", rec.offload_kv is True)
    check("tune: offload_kv попадает в load_fields",
          rec.load_fields().get("offload_kv") is True)
    check("tune: offload_kv попадает в отчёт",
          rec.describe().get("offload_kv") is True)

    off = tune.Recommendation(model_key="m", context=49152, vram_budget_bytes=tune.GIB,
                              offload_kv=False)
    check("tune: offload_kv можно выключить",
          off.load_fields().get("offload_kv") is False)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "weds.py"), "tune", "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    check("cli: tune знает --kv-in-ram", "--kv-in-ram" in proc.stdout)


def test_cli_help() -> None:
    """CLI должен собираться, а общие флаги работать и до, и после подкоманды."""
    for argv in (["--help"], ["check", "--help"], ["tune", "--help"]):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "weds.py"), *argv],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        check(f"cli: weds {' '.join(argv)}", proc.returncode == 0, proc.stderr[:200])

    for argv in (["--json", "discover"], ["discover", "--json"]):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "weds.py"), *argv],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        ok = proc.returncode == 0 and proc.stdout.strip().startswith("{")
        check(f"cli: флаги работают ({' '.join(argv)})", ok, proc.stderr[:200])


def test_mcp_protocol() -> None:
    """MCP: рукопожатие, список инструментов, вызов, обработка ошибок."""
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "weds.py"), "mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )

    def rpc(payload):
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    try:
        init = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "smoke", "version": "1"}}})
        check("mcp: initialize отдаёт serverInfo",
              init["result"]["serverInfo"]["name"] == "wednesday")

        tools = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = [t["name"] for t in tools["result"]["tools"]]
        check("mcp: 8 инструментов", len(names) == 8, str(len(names)))
        check("mcp: есть local_llm_check", "local_llm_check" in names)

        bad = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "nope", "arguments": {}}})
        check("mcp: неизвестный инструмент даёт ошибку", "error" in bad)

        nf = rpc({"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}})
        check("mcp: неизвестный метод даёт ошибку", "error" in nf)
    finally:
        proc.terminate()


def test_live_server() -> None:
    """Проверки на живом сервере — только если он запущен."""
    if not server_up():
        print("[SKIP] живой сервер на 1234 не отвечает — интеграционные тесты пропущены")
        return

    from weds.servers import LMStudioServer

    srv = LMStudioServer("http://127.0.0.1:1234", timeout=120)
    models = srv.list_models()
    check("live: список моделей получен", len(models) > 0, str(len(models)))

    r = checks.check_server(srv)
    check("live: check_server = PASS", r.status == checks.PASS, r.summary)

    if models:
        r = checks.check_model_present(srv, models[0].id)
        check("live: модель находится", r.status == checks.PASS, r.summary)

        r = checks.check_tool_calling(srv, models[0].id)
        check("live: tool-calling работает", r.status == checks.PASS, r.summary)


def main() -> int:
    print(f"Wednesday smoke tests — {ROOT}\n")
    test_gguf_parser()
    test_context_overflow_parsing()
    test_estimate_parsing()
    test_memory_curve()
    test_config_roundtrip()
    test_backup_not_overwritten()
    test_register_no_available_models()
    test_build_prompt_nonce()
    test_summarize()
    test_hardware_probe()
    test_budget_override()
    test_cli_accepts_budget_flag()
    test_offload_kv()
    test_explicit_server_url_wins()
    test_cli_help()
    test_mcp_protocol()
    test_live_server()

    print(f"\nИтог: {len(PASSED)} ok, {len(FAILED)} fail")
    if FAILED:
        print("Провалено:")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
