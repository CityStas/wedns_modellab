"""Приёмочный прогон через API приложения: bench -> start -> measure -> stop.

Проверяется не библиотека, а рабочий путь UI: те же четыре эндпоинта, которые
дёргают кнопки. Если этот тест зелёный, страница работает.

Запуск (приложение должно быть поднято):
    python -m modellab.gui --port 8090
    python tests/accept_api.py [--base http://127.0.0.1:8090] [--model PATH]

По умолчанию берётся самая большая модель из каталога - то есть Ternary-Bonsai
2-27B, ради которой всё это и делалось.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  OK   " if cond else "  FAIL ") + name
          + (f"  {detail}" if detail else ""), flush=True)
    if not cond:
        fails.append(name)


def api(base: str, path: str, body: dict | None = None, timeout: float = 60.0):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def wait_job(base: str, jid: str, timeout: float, every: float = 5.0) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        time.sleep(every)
        j = api(base, f"/api/job?id={jid}")
        line = f"    [{time.time() - t0:6.1f}s] {j['state']}: {j['progress'][:110]}"
        if line != last:
            print(line, flush=True)
            last = line
        if j["state"] != "running":
            return j
    return {"state": "timeout", "error": f"не завершилось за {timeout} с"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args()
    base = a.base

    print("== система ==", flush=True)
    st = api(base, "/api/state")
    sysinfo = st["system"]
    print(f"  CPU : {sysinfo['cpu']['name']} ({sysinfo['cpu']['physical']} ядер)")
    print(f"  GPU : {sysinfo['gpu']['name']} {sysinfo['gpu']['vram_total_mb']} МБ")
    print(f"  RAM : {sysinfo['mem']['total_mb']} МБ, свободно {sysinfo['mem']['avail_mb']}")
    print(f"  бинарник: {sysinfo['exe']['path']}")
    check("бинарник найден", sysinfo["exe"].get("ok") is True)
    check("llama-bench найден", bool(sysinfo.get("bench_exe")),
          str(sysinfo.get("bench_exe")))
    check("предстартовая проверка пройдена", st["preflight"]["ok"] is True,
          st["preflight"].get("reason", ""))

    model = a.model
    if not model:
        ms = sorted(api(base, "/api/models"), key=lambda m: -(m.get("size_mb") or 0))
        model = ms[0]["path"]
        print(f"  модель по умолчанию: {ms[0]['name']} "
              f"({ms[0]['size_mb']} МБ, {ms[0].get('layers')} слоёв, "
              f"{ms[0].get('arch')})")

    # -- 1. потолок железа ------------------------------------------------
    bench = {}
    if not a.skip_bench:
        print("== потолок: llama-bench ==", flush=True)
        j = api(base, "/api/bench", {"model": model, "ngl": a.ngl, "p": 512,
                                     "n": 128, "r": 3, "fa": 1,
                                     "ctk": "q4_0", "ctv": "q4_0"})
        j = wait_job(base, j["id"], 900)
        bench = j.get("result") or {}
        check("bench завершился", j["state"] == "done", j.get("error", ""))
        check("потолок генерации измерен", bool(bench.get("tg")),
              f"tg={bench.get('tg')} tok/s")
        check("потолок префилла измерен", bool(bench.get("pp")),
              f"pp={bench.get('pp')} tok/s")
        check("bench шёл с --load-mode none", bench.get("load_mode") == "none",
              str(bench.get("load_mode")))
        check("сторож не сработал на bench", not (bench.get("mem") or {}).get("tripped"))
        if bench.get("tg"):
            print(f"  потолок: генерация {bench['tg']} tok/s, "
                  f"промпт {bench.get('pp')} tok/s")

    # -- 2. сервер --------------------------------------------------------
    print("== подъём сервера ==", flush=True)
    j = api(base, "/api/start", {"model": model, "ctx": a.ctx, "ngl": a.ngl})
    j = wait_job(base, j["id"], 1800)
    check("сервер поднялся", j["state"] == "done", j.get("error", ""))
    eng = (j.get("result") or {}).get("engine") or {}
    health = (j.get("result") or {}).get("health") or {}
    print(f"  готов за {health.get('t_end')} с, load-mode {eng.get('load_mode')}, "
          f"слоёв на GPU {eng.get('layers_gpu')}/{eng.get('layers_total')}, "
          f"graph splits {eng.get('graph_splits')}")
    check("все слои на GPU", eng.get("all_layers_on_gpu") is True)
    check("режим загрузки none", eng.get("load_mode") == "none", str(eng.get("load_mode")))
    check("граф не рвётся (splits <= 2)", (eng.get("graph_splits") or 9) <= 2,
          str(eng.get("graph_splits")))
    check("движок отдал бюджет VRAM", bool(eng.get("vram_mib")),
          f"VRAM {eng.get('vram_mib')} МБ, хост {eng.get('host_mib')} МБ")
    check("нет ошибок в логе движка", not eng.get("errors"), str(eng.get("errors")))
    dev = (eng.get("devices") or {}).get("CUDA0") or {}
    if dev:
        print(f"  бюджет: веса {dev.get('model_mb')} + контекст {dev.get('context_mb')}"
              f" + вычисления {dev.get('compute_mb')} = {dev.get('self_mb')} МБ "
              f"при свободных {dev.get('free_mb')} МБ")

    # -- 3. замер ---------------------------------------------------------
    print("== замер ==", flush=True)
    j = api(base, "/api/measure", {"prompt_chars": 400, "max_tokens": 128,
                                   "repeats": 2})
    j = wait_job(base, j["id"], 900)
    m = j.get("result") or {}
    check("замер прошёл", j["state"] == "done" and m.get("ok") is True,
          j.get("error") or m.get("err", ""))
    check("устойчивая скорость измерена", (m.get("tps_steady") or 0) > 0,
          f"{m.get('tps_steady')} tok/s")
    check("TTFT измерен", (m.get("ttft_s") or 0) > 0, f"{m.get('ttft_s')} с")
    check("host_ratio посчитан", m.get("host_ratio") is not None,
          f"{m.get('host_ratio')}")
    hr = m.get("host_ratio")
    check("копии весов в системной памяти нет (host_ratio < 0.5)",
          hr is not None and hr < 0.5, f"{hr}")
    min_avail = (m.get("mem") or {}).get("min_avail_mb")
    check("памяти хватило с запасом", (min_avail or 0) > 1500,
          f"минимум {min_avail} МБ свободных")
    check("сторож не сработал", not (m.get("mem") or {}).get("watchdog", {}).get("tripped"))
    if m.get("ok"):
        print(f"  скорость {m['tps_steady']} tok/s "
              f"(разброс ±{m.get('tps_steady_spread')}, дрожание {m.get('jitter_pct')}%), "
              f"префилл {m.get('prefill_tps')} tok/s, TTFT {m.get('ttft_s')} с")
        print(f"  host_ratio {hr}, рабочий набор {m['mem'].get('proc_ws_mb')} МБ "
              f"на файл {m.get('model_size_mb')} МБ")
        if bench.get("tg"):
            print(f"  от потолка железа: {m.get('ceiling_pct')}%")

    # -- 4. состояние и остановка -----------------------------------------
    print("== состояние ==", flush=True)
    st = api(base, "/api/state")
    check("сервер числится живым", st["server"].get("running") is True)
    check("есть точка для графика", len(st.get("timeline") or []) > 2,
          f"{len(st.get('timeline') or [])} точек")
    check("история пополнилась", len(st.get("history") or []) >= 1)
    check("движок виден в состоянии", bool(st.get("engine")))
    log = api(base, "/api/log?tail=50").get("log") or ""
    check("лог читается", len(log) > 500, f"{len(log)} символов")
    check("лог без 'unavailable_error'", "unavailable_error" not in log)

    print("== остановка ==", flush=True)
    j = api(base, "/api/stop", {})
    j = wait_job(base, j["id"], 120, every=2.0)
    check("остановка выполнена", j["state"] == "done", j.get("error", ""))
    st = api(base, "/api/state")
    check("сервер не числится живым", st["server"].get("running") is False)
    check("порт закрыт", st["server"].get("running") is False)

    print()
    print(f"ИТОГ: {'всё зелёное' if not fails else 'провалено: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
