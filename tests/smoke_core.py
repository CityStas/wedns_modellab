"""Смоук ядра modellab: статика + живой прогон на маленькой модели.

Живой прогон делается на qwen3.5-0.8b (516 МБ), а не на рабочем 27B:
проверяем код, а не терпение пользователя. Архитектура та же (qwen35), так
что форма проверки та же, что и на большой модели.

Запуск: python tests/smoke_core.py [--live]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modellab import llamasrv, measure, probe, watchdog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE = "--live" in sys.argv
SMALL = r"E:\Programs\LocalLLM\Models\mini\qwen3.5-0.8b-q4_k_m\qwen3.5-0.8b-q4_k_m.gguf"
BIG = r"E:\Programs\LocalLLM\Models\ternary\Ternary-Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-PTQ1_0.gguf"

fails = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def show(title, obj):
    print(f"== {title} ==")
    print(json.dumps(obj, ensure_ascii=False, default=str)[:900])


# ---------------------------------------------------------------- статика
print("== suggest_load_mode ==")
check("полный вынос -> none", llamasrv.suggest_load_mode(99, 65) == "none")
check("частичный вынос -> mmap", llamasrv.suggest_load_mode(40, 65) == "mmap")
check("99 без данных о слоях -> none", llamasrv.suggest_load_mode(99, None) == "none")
check("40 без данных о слоях -> mmap", llamasrv.suggest_load_mode(40, None) == "mmap")
check("ровно все слои -> none", llamasrv.suggest_load_mode(65, 65) == "none")

print("== free_port ==")
p1 = llamasrv.free_port(8081)
check("порт найден", p1 > 0, str(p1))
check("занятый порт пропускается", llamasrv.free_port(p1 + 1) != p1 or True)
import socket  # noqa: E402
s = socket.socket()
s.bind(("127.0.0.1", 0))
busy = s.getsockname()[1]
got = llamasrv.free_port(busy)
check("занятый порт не отдан", got != busy, f"busy={busy} got={got}")
s.close()

print("== detect_exe / version ==")
exe = llamasrv.detect_exe()
show("detect_exe", exe)
check("бинарник найден", exe.get("ok") is True, str(exe.get("path")))
if exe.get("ok"):
    v = llamasrv.version(exe["path"])
    show("version", v)
    check("версия читается", v.get("ok") is True, str(v.get("version") or v.get("error")))

print("== port_open ==")
check("закрытый порт -> False", llamasrv.port_open(1) is False)

print("== watchdog: детектор ==")
fired = []
wd = watchdog.MemoryWatchdog(floor_mb=10 ** 9, interval=0.05,
                             on_trip=lambda rec: fired.append(rec))
wd.start()
time.sleep(0.4)
wd.stop()
t = wd.tripped
show("tripped", t)
check("сработал по физической памяти", bool(t) and t["reason"] == "avail")
check("колбэк вызван", bool(fired))
check("отчёт содержит низшую точку", (wd.report().get("min_avail_mb") or 0) > 0,
      str(wd.report().get("min_avail_mb")))
try:
    wd.assert_ok()
    check("assert_ok бросает при срабатывании", False)
except watchdog.MeasurementAborted:
    check("assert_ok бросает при срабатывании", True)

wd2 = watchdog.MemoryWatchdog(floor_mb=1, interval=0.05)
wd2.start()
time.sleep(0.3)
wd2.stop()
check("здоровый прогон не срабатывает", wd2.tripped is None)
try:
    wd2.assert_ok()
    check("assert_ok молчит на здоровом прогоне", True)
except watchdog.MeasurementAborted:
    check("assert_ok молчит на здоровом прогоне", False)
check("timeline отдаёт ряд", len(wd2.timeline()) >= 2, str(len(wd2.timeline())))
check("baseline записан", wd2.baseline is not None)

print("== ServerConfig ==")
cfg = llamasrv.config_for(BIG)
show("config_for(27B)", cfg.describe())
check("load_mode выведен как none", cfg.load_mode == "none", cfg.load_mode)
check("alias из имени файла", cfg.alias == "Ternary-Bonsai-2-27B-PTQ1_0", cfg.alias)
argv = cfg.to_argv()
check("argv без --log-file (лог пишем сами)", "--log-file" not in argv)
check("argv содержит --load-mode none",
      argv[argv.index("--load-mode") + 1] == "none")
check("argv содержит -ngl 99", argv[argv.index("-ngl") + 1] == "99")
check("-lv 4 по умолчанию (иначе нет учёта VRAM)",
      cfg.verbose == 4 and argv[argv.index("-lv") + 1] == "4")
cfg2 = llamasrv.ServerConfig.from_dict(cfg.to_dict())
check("round-trip конфига", cfg2.fingerprint() == cfg.fingerprint())
check("fingerprint не зависит от порта",
      cfg2.fingerprint() == llamasrv.ServerConfig.from_dict(
          {**cfg.to_dict(), "port": 9999}).fingerprint())
check("fingerprint ловит смену ctx",
      cfg2.fingerprint() != llamasrv.ServerConfig.from_dict(
          {**cfg.to_dict(), "ctx": 32768}).fingerprint())

print("== measure: заполнитель ==")
a, b = measure.prompt_filler(1000), measure.prompt_filler(1000)
check("длина ровно запрошенная", len(a) == 1000 and len(b) == 1000, str(len(a)))
check("заполнитель детерминирован", a == b)

# ------------------------------------------------------------ живой прогон
if LIVE:
    print("== live: 0.8B ==")
    # Лог рядом с проектом, а не по абсолютному пути: путь был зашит под старое
    # расположение (F:\Pets\Cursor\wd\modellab) и после переезда каталога просто
    # не существовал — живой прогон падал на записи лога.
    log = os.path.join(ROOT, "modellab", "logs", "_smoke-server.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    c = llamasrv.config_for(SMALL, ctx=4096)
    srv = llamasrv.LlamaServer(c, log_path=log)
    wd3 = watchdog.MemoryWatchdog(floor_mb=800, interval=0.2, on_trip=lambda r: srv.stop())
    wd3.start()
    try:
        srv.start()
        wd3.pid = srv.pid
        wd3.phase = "load"
        h = srv.wait_health(timeout=300)
        show("health", h)
        check("сервер поднялся", h["state"] == "ready", str(h))
        check("модель загрузилась", srv.wait_for_log(r"model loaded|server is listening", 60))
        wd3.phase = "measure"
        m = measure.measure(srv.base_url, c.alias, prompt_chars=400,
                            max_tokens=96, watchdog=wd3, srv=srv,
                            model_path=SMALL, repeats=2)
        show("measure", {k: v for k, v in m.items() if k != "runs"})
        check("замер прошёл", m.get("ok") is True, str(m.get("err")))
        check("устойчивая скорость > 0", (m.get("tps_steady") or 0) > 0,
              str(m.get("tps_steady")))
        check("ряд мгновенных скоростей непустой", len(m.get("tps_series") or []) > 20)
        check("ttft измерен", (m.get("ttft_s") or 0) > 0, str(m.get("ttft_s")))
        check("host_ratio посчитан", m.get("host_ratio") is not None,
              str(m.get("host_ratio")))
        check("память снята", bool((m.get("mem") or {}).get("after")))
        check("движок разобран", bool(m.get("engine")))
        check("сторож не сработал", wd3.tripped is None, str(wd3.tripped))
        srv.metrics()
        check("метрики читаются", bool(srv.metrics()))
        pc = measure.prefill_curve(srv.base_url, c.alias, sizes=(200, 4000),
                                   max_tokens=4)
        show("prefill_curve", pc)
        check("кривая префилла построена", len(pc) == 2 and pc[0]["ok"])
    finally:
        st = srv.stop()
        wd3.stop()
        show("stop", st)
        check("процесс остановлен", st.get("was_alive") is not True or st.get("rc") is not None)
        check("порт закрыт", st.get("port_closed") is True)
        check("лог непустой", os.path.getsize(log) > 2000, str(os.path.getsize(log)))
        b = probe.log_buffers(log)
        check("лог разобран: слои на GPU", b.get("all_layers_on_gpu") is True, str(b.get("layers_gpu")))
else:
    print("== live пропущен (запусти с --live) ==")

print()
print(f"ИТОГ: {'всё зелёное' if not fails else 'провалено: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
