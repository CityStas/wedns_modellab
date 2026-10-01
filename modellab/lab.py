"""Оркестратор: одно состояние, одна тяжёлая задача за раз, ничего в фоне.

Почему именно так, а не «пусть UI дёргает функции напрямую».

Приложение измеряет память. Значит, всё, что работает параллельно с замером,
портит замер. Один поток на тяжёлые операции и запрет на второй одновременный
прогон - это не удобство, а условие корректности: два сервера, поднятые
одновременно, делят VRAM, и обе цифры становятся мусором.

Долгие операции (подъём сервера ~40-60 с, замер, llama-bench) выполняются в
рабочем потоке, а UI опрашивает состояние. Никаких синхронных HTTP-запросов
на минуту: браузер бы отвалился по таймауту, а прогресс был бы не виден.

Сервер останавливается через atexit: приложение, которое оставляет
llama-server держать 5.7 ГБ после закрытия вкладки, - это ровно та проблема,
из-за которой всё это писалось.
"""

from __future__ import annotations

import atexit
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import llamasrv, measure, probe, watchdog

CREATE_NO_WINDOW = llamasrv.CREATE_NO_WINDOW
# Каталог моделей по умолчанию. Путь этой машины оставлен как рабочий, но
# переопределяется переменной окружения или ключом --models: иначе клон
# репозитория находит ноль моделей и выглядит сломанным.
MODELS_ROOT = os.environ.get("MODELLAB_MODELS") or r"E:\Programs\LocalLLM\Models"
DEFAULT_LOG_DIR = str(Path(__file__).resolve().parent / "logs")


@dataclass
class Job:
    id: str
    kind: str
    state: str = "running"          # running | done | failed | aborted
    started: float = field(default_factory=time.time)
    finished: float | None = None
    progress: str = ""
    result: dict = field(default_factory=dict)
    error: str = ""
    log: list[str] = field(default_factory=list)

    def say(self, msg: str) -> None:
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.progress = msg
        del self.log[:-400]

    def to_dict(self, log_lines: int = 60) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state,
                "started": self.started, "finished": self.finished,
                "elapsed": round((self.finished or time.time()) - self.started, 1),
                "progress": self.progress, "result": self.result,
                "error": self.error, "log": self.log[-log_lines:]}


def find_llama_bench(exe: str | None = None) -> str | None:
    """llama-bench рядом с llama-server: он даёт потолок железа, с которым
    потом сравнивается реальность сервера."""
    if exe and Path(exe).is_file():
        return exe
    srv = llamasrv.detect_exe().get("path")
    if srv:
        cand = Path(srv).with_name("llama-bench.exe")
        if cand.is_file():
            return str(cand)
    import shutil
    return shutil.which("llama-bench")


def parse_bench(text: str) -> dict:
    """Разобрать markdown-таблицу llama-bench.

    Нужны две строки: pp512 (обработка промпта) и tg128 (генерация). Именно
    они дают потолок, недостижимый для сервера: у сервера сверху ещё HTTP,
    слоты, шаблон чата и захват CUDA-графа на первом токене.
    """
    out: dict = {"rows": [], "vram": {}}
    m = re.search(r"VRAM:\s*(\d+)\s*MiB,\s*(\d+)\s*MiB free", text)
    if m:
        out["vram"] = {"total_mb": int(m.group(1)), "free_mb": int(m.group(2))}
    m = re.search(r"CPU:\s*(.+)", text)
    if m:
        out["cpu"] = m.group(1).strip()
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line or "t/s" in line.split("|")[-2:][0]:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        test = cells[-2]
        if test not in ("pp512", "tg128"):
            continue
        num = re.search(r"([\d.]+)\s*(?:±\s*([\d.]+))?", cells[-1])
        if not num:
            continue
        row = {"model": cells[0], "size": cells[1], "backend": cells[3],
               "ngl": cells[4], "test": test,
               "tps": float(num.group(1)),
               "std": float(num.group(2)) if num.group(2) else None}
        out["rows"].append(row)
        out["pp" if test == "pp512" else "tg"] = row["tps"]
        if row["std"] is not None:
            out[("pp" if test == "pp512" else "tg") + "_std"] = row["std"]
    return out


def need_mb(model_path: str, load_mode: str) -> int:
    """Сколько системной памяти понадобится этому конфигу.

    Разница в разы, и она и есть весь смысл выбора режима загрузки:
    при `none` веса уезжают на карту, в системной памяти остаётся рантайм;
    при `mmap` там же лежит копия всего файла модели.
    """
    size = (probe.model_file(model_path).get("size_mb") or 0)
    if load_mode in ("none", "dio"):
        return 600
    return int(size * 1.1) + 600


class Lab:
    """Состояние приложения и единственный рабочий поток."""

    def __init__(self, models_root: str = MODELS_ROOT,
                 log_dir: str = DEFAULT_LOG_DIR,
                 floor_mb: int = watchdog.DEFAULT_FLOOR_MB,
                 commit_floor_mb: int = watchdog.DEFAULT_COMMIT_FLOOR_MB):
        self.models_root = models_root
        self.log_dir = log_dir
        self.floor_mb = floor_mb
        self.commit_floor_mb = commit_floor_mb
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.srv: llamasrv.LlamaServer | None = None
        self.wd: watchdog.MemoryWatchdog | None = None
        self.jobs: dict[str, Job] = {}
        self.job_order: list[str] = []
        self.current: str | None = None
        self.last_measure: dict = {}
        self.last_bench: dict = {}
        # Результат A/B живёт отдельно от job: state() отдаёт только ТЕКУЩЕЕ
        # задание, а оно обнуляется в finally сразу после завершения. Без
        # этого поля сравнение исчезало бы с экрана ровно в тот момент, когда
        # его становится интересно читать.
        self.last_ab: dict = {}
        self.last_config: dict = {}
        self.timeline: list[list[float]] = []
        self.history: list[dict] = []
        self._models_cache: dict = {"t": 0.0, "data": []}
        self._sys_cache: dict = {"t": 0.0, "data": {}}
        self._pre_cache: dict = {"t": 0.0, "data": {}}
        atexit.register(self.shutdown)

    def preflight(self, model: str | None = None,
                  load_mode: str | None = None) -> dict:
        """Проверка перед стартом. Кэш на 5 секунд: обход списка процессов
        стоит недёшево, а состояние меняется не мгновенно."""
        with self.lock:
            fresh = time.time() - self._pre_cache["t"] < 5
            cached = dict(self._pre_cache["data"]) if fresh else {}
        if cached and model is None and load_mode is None:
            return cached
        mode = load_mode or self.last_config.get("load_mode") or "none"
        mdl = model or self.last_config.get("model") or ""
        data = probe.preflight(self.floor_mb, need_mb(mdl, mode) if mdl else 0)
        data["model"] = mdl
        data["load_mode"] = mode
        with self.lock:
            self._pre_cache = {"t": time.time(), "data": data}
        return data

    # -- железо и модели ---------------------------------------------------

    def system(self, refresh: bool = False) -> dict:
        """Железо и бинарники. Кэш на две минуты, и это принципиально.

        Определение железа запускает внешние процессы (nvidia-smi, powershell
        для числа ядер, llama-server --version). UI опрашивает состояние
        примерно раз в секунду, и без кэша это означало бы десятки запусков
        процессов в минуту - приложение, которое тормозит машину, показывая,
        сколько на ней свободно памяти.
        """
        with self.lock:
            if not refresh and time.time() - self._sys_cache["t"] < 120:
                return self._sys_cache["data"]
        exe = llamasrv.detect_exe()
        bench = find_llama_bench()
        data = {"cpu": probe.cpu_info(), "gpu": probe.gpu_info(),
                "mem": probe.system_memory(),
                "exe": exe, "bench_exe": bench,
                "exe_version": llamasrv.version(exe["path"]) if exe.get("ok") else {},
                "platform": os.name}
        with self.lock:
            self._sys_cache = {"t": time.time(), "data": data}
        return data

    def models(self, refresh: bool = False) -> list[dict]:
        with self.lock:
            if not refresh and time.time() - self._models_cache["t"] < 30:
                return self._models_cache["data"]
        rows: list[dict] = []
        try:
            from weds.gguf import parse as gguf_parse
        except Exception:  # noqa: BLE001
            gguf_parse = None
        for p in sorted(Path(self.models_root).rglob("*.gguf")):
            if not p.is_file():
                continue
            mf = probe.model_file(str(p))
            row = {"name": p.name, "path": str(p), "size_mb": mf.get("size_mb"),
                   "mtime": mf.get("mtime")}
            if gguf_parse:
                try:
                    g = gguf_parse(p)
                    row.update({"arch": g.arch, "layers": g.get("block_count"),
                                "ctx_train": g.get("context_length"),
                                "hybrid": g.is_hybrid,
                                "kv_layers": g.full_attention_layers})
                except Exception as e:  # noqa: BLE001
                    row["meta_error"] = str(e)[:120]
            rows.append(row)
        with self.lock:
            self._models_cache = {"t": time.time(), "data": rows}
        return rows

    def model_info(self, path: str) -> dict:
        for m in self.models():
            if os.path.normcase(m["path"]) == os.path.normcase(path):
                return m
        return {}

    # -- состояние ---------------------------------------------------------

    def mem_now(self) -> dict:
        pid = self.srv.pid if self.srv and self.srv.alive else None
        sn = probe.snapshot(pid)
        out = sn.to_dict()
        out["system"] = probe.system_memory()
        model = self.last_config.get("model") or ""
        mf = probe.model_file(model) if model else {}
        if pid:
            out["host_ratio"] = probe.host_ratio(mf.get("size_mb"), sn.proc_ws_mb)
            out["model_size_mb"] = mf.get("size_mb")
        return out

    def state(self) -> dict:
        with self.lock:
            job = self.jobs.get(self.current) if self.current else None
            srv_state: dict = {"running": False}
            if self.srv:
                srv_state = {"running": self.srv.alive, "pid": self.srv.pid,
                             "port": self.srv.cfg.port, "health": self.srv.health,
                             "uptime_s": round(time.time() - self.srv._started, 1)
                             if self.srv._started else None,
                             "cfg": self.srv.cfg.describe()}
            return {"server": srv_state,
                    "job": job.to_dict() if job else None,
                    "system": self.system(),
                    "preflight": self.preflight(),
                    "floor_mb": self.floor_mb,
                    "mem": self.mem_now(),
                    "timeline": self.timeline[-240:],
                    "last_measure": self.last_measure,
                    "last_bench": self.last_bench,
                    "last_ab": self.last_ab,
                    "last_config": self.last_config,
                    "history": self.history[-30:],
                    "engine": self.engine_now()}

    def engine_now(self) -> dict:
        if not (self.srv and self.srv.log_path):
            return {}
        b = probe.log_buffers(self.srv.log_path)
        return {k: b[k] for k in
                ("load_mode", "graph_splits", "n_ctx", "all_layers_on_gpu",
                 "layers_gpu", "layers_total", "vram_mib", "host_mib",
                 "kv_mib", "model_mib", "cpu_mapped_mib", "vram_free_mb",
                 "vram_total_mb", "projected_vram_mb", "devices",
                 "cpu_assigned_layers", "note", "vram_hint", "errors")
                if k in b}

    def log_tail(self, tail: int = 200) -> str:
        if not (self.srv and self.srv.log_path):
            return ""
        return self.srv.log_text(tail)

    # -- рабочий поток -----------------------------------------------------

    def _busy(self) -> bool:
        with self.lock:
            job = self.jobs.get(self.current) if self.current else None
            return bool(job and job.state == "running")

    def _spawn(self, kind: str, fn) -> Job:
        with self.lock:
            if self._busy():
                raise RuntimeError("уже идёт прогон: два замера одновременно "
                                   "испортят оба")
            job = Job(id=uuid.uuid4().hex[:8], kind=kind)
            self.jobs[job.id] = job
            self.job_order.append(job.id)
            self.current = job.id

        def wrapper():
            try:
                fn(job)
                job.state = "done"
            except watchdog.MeasurementAborted as e:
                job.state = "aborted"
                job.error = str(e)
                job.say(f"ПРЕРВАНО: {e}")
            except Exception as e:  # noqa: BLE001
                job.state = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.say(f"ОШИБКА: {job.error}")
            finally:
                job.finished = time.time()
                job.progress = job.progress or job.state
                with self.lock:
                    self.current = None

        threading.Thread(target=wrapper, name=f"job-{kind}", daemon=True).start()
        return job

    def job(self, jid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(jid)
            return j.to_dict() if j else None

    # -- операции ----------------------------------------------------------

    def _make_server(self, cfg: llamasrv.ServerConfig) -> llamasrv.LlamaServer:
        log = str(Path(self.log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg.alias}.log")
        return llamasrv.LlamaServer(cfg, log_path=log)

    def start(self, overrides: dict) -> Job:
        """Поднять сервер под замер. Конфиг собирается через config_for,
        поэтому --load-mode выводится из числа слоёв, а не берётся с потолка."""
        model = overrides.get("model")
        if not model:
            raise ValueError("не задана модель")
        cfg = llamasrv.config_for(model, **{k: v for k, v in overrides.items()
                                            if k != "model" and v is not None})

        def work(job: Job) -> None:
            self.stop_now(job)
            pre = probe.preflight(self.floor_mb, need_mb(model, cfg.load_mode))
            if not pre["ok"]:
                job.say("ОТКАЗ ДО СТАРТА: " + pre["reason"])
                raise RuntimeError(pre["reason"])
            with self.lock:
                self.last_config = cfg.to_dict()
                self.timeline = []
            job.say(f"модель: {Path(model).name}")
            job.say(f"режим загрузки: --load-mode {cfg.load_mode}, -ngl {cfg.ngl}, "
                    f"ctx {cfg.ctx}, kv {cfg.cache_type_k}")
            job.say(f"память: свободно {pre['avail_mb']} МБ, "
                    f"конфигу нужно около {pre['need_mb']} МБ, "
                    f"порог {self.floor_mb} МБ")
            srv = self._make_server(cfg)
            wd = watchdog.MemoryWatchdog(floor_mb=self.floor_mb,
                                         commit_floor_mb=self.commit_floor_mb,
                                         on_trip=lambda rec: self._trip(job, rec))
            wd.start()
            with self.lock:
                self.srv, self.wd = srv, wd
            try:
                srv.start()
                wd.pid = srv.pid
                wd.phase = "load"
                job.say(f"запущен, pid {srv.pid}, порт {cfg.port}")
                h = srv.wait_health(timeout=1800,
                                    on_tick=lambda el, st: job.say(
                                        f"загрузка {el:.0f} с, /health {st}"))
                wd.phase = "measure"
                if h["state"] != "ready":
                    raise RuntimeError(f"сервер не поднялся: {h}")
                job.say(f"готов за {h['t_end']} с (503 на {h['t_503']} с)")
                job.result = {"health": h, "engine": self.engine_now(),
                              "mem": wd.report()}
            except Exception:
                with self.lock:
                    self.srv = None
                wd.stop()
                srv.stop()
                raise

        return self._spawn("start", work)

    def _trip(self, job: Job, rec: dict) -> None:
        job.say(f"СТОРОЖ: {rec['reason']} {rec['value_mb']} МБ < "
                f"{rec['limit_mb']} МБ на фазе {rec['phase']} - останавливаю")
        if self.srv:
            self.srv.stop()

    def stop_now(self, job: Job | None = None) -> dict:
        with self.lock:
            srv, wd = self.srv, self.wd
            self.srv, self.wd = None, None
        out: dict = {}
        if wd:
            wd.stop()
        if srv:
            out = srv.stop()
            if job:
                job.say(f"сервер остановлен (rc={out.get('rc')}, "
                        f"порт закрыт: {out.get('port_closed')})")
        return out

    def stop(self) -> Job:
        def work(job: Job) -> None:
            job.result = self.stop_now(job) or {"note": "сервер не был запущен"}
        return self._spawn("stop", work)

    def measure(self, params: dict) -> Job:
        """Замер на уже поднятом сервере. Если сервер не поднят - поднимает."""
        def work(job: Job) -> None:
            srv = self.srv
            if not (srv and srv.alive):
                job.say("сервер не поднят, поднимаю по последнему конфигу")
                self._start_sync(job, params.get("config") or {})
                srv = self.srv
            assert srv
            wd = self.wd
            job.say(f"замер: промпт {params.get('prompt_chars', 400)} симв., "
                    f"{params.get('max_tokens', 128)} токенов, "
                    f"повторов {params.get('repeats', 2)}")
            t0 = time.time()
            m = measure.measure(srv.base_url, srv.cfg.alias,
                                prompt_chars=int(params.get("prompt_chars", 400)),
                                max_tokens=int(params.get("max_tokens", 128)),
                                watchdog=wd, srv=srv,
                                model_path=srv.cfg.model,
                                repeats=int(params.get("repeats", 2)))
            m["bench"] = self.last_bench
            if m.get("ok") and self.last_bench.get("tg"):
                ceil_ = self.last_bench["tg"]
                m["ceiling_pct"] = round(m["tps_steady"] / ceil_ * 100, 1)
            m["engine"] = self.engine_now()
            with self.lock:
                self.last_measure = m
                self.history.append({"t": time.time(),
                                     "tps": m.get("tps_steady"),
                                     "host_ratio": m.get("host_ratio"),
                                     "ctx": srv.cfg.ctx, "ngl": srv.cfg.ngl,
                                     "load_mode": srv.cfg.load_mode})
                if wd:
                    self.timeline = wd.timeline()
            if not m.get("ok"):
                job.say(f"замер не удался: {m.get('err')}")
                job.result = m
                return
            job.say(f"устойчивая скорость {m['tps_steady']} tok/s "
                    f"(разброс {m.get('tps_steady_spread')}), "
                    f"префилл {m.get('prefill_tps')} tok/s, "
                    f"TTFT {m.get('ttft_s')} с")
            job.say(f"host_ratio {m.get('host_ratio')} "
                    f"(рабочий набор {m['mem'].get('proc_ws_mb')} МБ "
                    f"на файл {m.get('model_size_mb')} МБ), "
                    f"свободно {m['mem'].get('min_avail_mb')} МБ минимум")
            job.say(f"итого {time.time() - t0:.1f} с")
            job.result = m
            if wd:
                wd.assert_ok()

        return self._spawn("measure", work)

    def _start_sync(self, job: Job, overrides: dict) -> None:
        cfg = llamasrv.config_for(overrides.get("model") or self.last_config.get("model"),
                                  **{k: v for k, v in overrides.items()
                                     if k not in ("model",) and v is not None})
        pre = probe.preflight(self.floor_mb, need_mb(cfg.model, cfg.load_mode))
        if not pre["ok"]:
            job.say("ОТКАЗ ДО СТАРТА: " + pre["reason"])
            raise RuntimeError(pre["reason"])
        with self.lock:
            self.last_config = cfg.to_dict()
            self.timeline = []
        srv = self._make_server(cfg)
        wd = watchdog.MemoryWatchdog(floor_mb=self.floor_mb,
                                     commit_floor_mb=self.commit_floor_mb,
                                     on_trip=lambda rec: self._trip(job, rec))
        wd.start()
        with self.lock:
            self.srv, self.wd = srv, wd
        srv.start()
        wd.pid = srv.pid
        wd.phase = "load"
        job.say(f"pid {srv.pid}, порт {cfg.port}, загружаю")
        h = srv.wait_health(timeout=1800,
                            on_tick=lambda el, st: job.say(
                                f"загрузка {el:.0f} с, /health {st}"))
        wd.phase = "measure"
        if h["state"] != "ready":
            raise RuntimeError(f"сервер не поднялся: {h}")
        job.say(f"готов за {h['t_end']} с")

    def search(self, params: dict) -> Job:
        """Подбор конфигурации. Логика в search.py, здесь только запуск.

        Импорт внутри метода: search.py тянет measure/llamasrv/probe, а те
        ничего из lab не тянут, но импорт на уровне модуля сделал бы порядок
        загрузки хрупким без всякой пользы.
        """
        from . import search as search_mod

        def work(job: Job) -> None:
            job.result = search_mod.run(self, job, params)

        return self._spawn("search", work)

    def ab(self, params: dict) -> Job:
        """Перемежающееся сравнение конфигураций (см. search.ab)."""
        from . import search as search_mod

        def work(job: Job) -> None:
            job.result = search_mod.ab(self, job, params)
            with self.lock:
                self.last_ab = job.result

        return self._spawn("ab", work)

    def bench(self, params: dict) -> Job:
        """Потолок железа через llama-bench. Сервер останавливается: две
        копии модели на одной карте дадут цифры, которые нельзя сравнивать."""
        def work(job: Job) -> None:
            exe = find_llama_bench(params.get("bench_exe"))
            if not exe:
                raise FileNotFoundError("llama-bench.exe не найден")
            model = params.get("model") or self.last_config.get("model")
            if not model:
                raise ValueError("не задана модель")
            if self.srv and self.srv.alive:
                job.say("останавливаю сервер: bench и сервер не могут делить карту")
                self.stop_now(job)
            cmd = [exe, "-m", model, "-ngl", str(params.get("ngl", 99)),
                   "-p", str(params.get("p", 512)), "-n", str(params.get("n", 128)),
                   "-r", str(params.get("r", 3)), "-o", "md"]
            # --load-mode и здесь обязателен. У llama-bench свой дефолт `auto`,
            # то есть mmap, а это тот самый сценарий, из-за которого машина с
            # 16 ГБ уходила в своп: веса уже на карте, но копия 5.67 ГБ
            # остаётся в системной памяти. Потолок, снятый так, ещё и неверен -
            # он измерен на другой конфигурации памяти.
            layers = llamasrv.model_layers(model)
            lm = params.get("load_mode") or llamasrv.suggest_load_mode(
                params.get("ngl", 99), layers)
            pre = probe.preflight(self.floor_mb, need_mb(model, lm))
            if not pre["ok"]:
                job.say("ОТКАЗ ДО СТАРТА: " + pre["reason"])
                raise RuntimeError(pre["reason"])
            cmd += ["-lm", str(lm)]
            job.say(f"режим загрузки для bench: {lm}"
                    + (f" (слоёв в модели {layers})" if layers else ""))
            if params.get("fa"):
                cmd += ["-fa", str(params["fa"])]
            if params.get("ctk"):
                cmd += ["-ctk", str(params["ctk"])]
            if params.get("ctv"):
                cmd += ["-ctv", str(params["ctv"])]
            job.say("llama-bench: " + " ".join(cmd[1:]))

            holder: dict = {}

            def trip(rec: dict) -> None:
                job.say(f"СТОРОЖ: {rec['reason']} {rec['value_mb']} МБ < "
                        f"{rec['limit_mb']} МБ - останавливаю llama-bench")
                proc = holder.get("p")
                if proc and proc.poll() is None:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True,
                                   creationflags=CREATE_NO_WINDOW)

            wd = watchdog.MemoryWatchdog(floor_mb=self.floor_mb,
                                         commit_floor_mb=self.commit_floor_mb,
                                         on_trip=trip)
            wd.start()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1,
                                 creationflags=CREATE_NO_WINDOW)
            holder["p"] = p
            wd.pid = p.pid
            lines: list[str] = []
            assert p.stdout
            try:
                for line in p.stdout:
                    line = line.rstrip()
                    lines.append(line)
                    if line.startswith("|") and ("pp" in line or "tg" in line):
                        job.say(line)
            finally:
                p.wait()
                wd.stop()
            text = "\n".join(lines)
            res = parse_bench(text)
            res["rc"] = p.returncode
            res["cmd"] = cmd
            res["load_mode"] = lm
            res["layers"] = layers
            res["mem"] = wd.report()
            with self.lock:
                self.last_bench = res
            if res.get("tg"):
                job.say(f"потолок железа при load-mode {lm}: генерация {res['tg']} tok/s, "
                        f"промпт {res.get('pp')} tok/s")
            else:
                job.say("llama-bench не отдал строк tg128/pp512")
                job.say("\n".join(lines[-8:]))
            job.result = res
            wd.assert_ok()

        return self._spawn("bench", work)

    # -- завершение --------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self.stop_now()
        except Exception:  # noqa: BLE001
            pass

    def dump(self) -> dict:
        return {"state": self.state(), "system": self.system(),
                "models": self.models()}
