"""Запуск llama-server как объекта: конфиг, argv, здоровье, лог, остановка.

Отдельный модуль, потому что в проекте уже дважды наступали на одни и те же
грабли, и обе теперь закрыты здесь по умолчанию.

1. Лог, перенаправленный в файл, обрезается. Если отдать stdout ребёнка файлу
   напрямую, буферизация блоками означает, что при `terminate()` последние
   килобайты не сбрасываются: наблюдался лог в 1171 байт от прогона, который
   отработал 15 минут. Здесь stdout читается потоком (`bufsize=1`), и файл
   пишем мы сами, а не движок через `--log-file`. Поэтому в argv флага
   `--log-file` нет намеренно.

2. Убийство одного процесса не останавливает сервер. У проекта есть
   супервизор, который поднимает llama-server заново через 5 секунд, так что
   `taskkill /F /IM llama-server.exe` бесполезен: нужно валить дерево
   процессов (`/T`) по конкретному pid и потом убеждаться, что порт закрыт.

Плюс `--load-mode` не выставляется «на глаз»: см. `suggest_load_mode`.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from . import probe

IS_WINDOWS = sys.platform == "win32"
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

# Режимы загрузки, которые понимает движок.
LOAD_MODES = ("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")


def suggest_load_mode(ngl: int | str, n_layer: int | None) -> str:
    """Выбрать режим загрузки по тому, влезает ли модель на карту целиком.

    Логика неочевидная и стоит того, чтобы жить в коде:

      * все слои на GPU -> `none`. Движок прочитает файл, загрузит веса на
        карту и отпустит системный буфер. Замерено на Ternary-Bonsai-2-27B,
        ctx 65536: рабочий набор сервера 945 МБ против 6072 МБ при mmap,
        свободная память 6.9 ГБ против 1.9 ГБ, decode 17.9 против 9.87 tok/s.
      * часть слоёв на CPU -> `mmap`. Веса остаются в системной памяти и
        должны быть вытесняемыми; без mmap это анонимная память, которую
        Windows не может выбросить, и машина уходит в своп.
    """
    if not n_layer:
        return "none" if str(ngl).strip() in ("99", "999", "-1") else "mmap"
    try:
        n = int(ngl)
    except (TypeError, ValueError):
        return "mmap"
    return "none" if n >= n_layer else "mmap"


def free_port(preferred: int, host: str = "127.0.0.1", tries: int = 20) -> int:
    """Первый свободный порт, начиная с preferred."""
    for offset in range(tries):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"нет свободного порта в диапазоне {preferred}..{preferred + tries - 1}")


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


# --------------------------------------------------------------------------
# поиск бинарника
# --------------------------------------------------------------------------

_EXE_HINTS = [
    r"E:\Programs\LocalLLM\bonsai\bin\llama-server.exe",
    r"E:\Programs\LocalLLM\apps\llama.cpp\bin\llama-server.exe",
]


def model_layers(model_path: str) -> int | None:
    """Число слоёв из заголовка GGUF, без запуска движка.

    Нужно до старта: от него зависит выбор `--load-mode`. Парсер берём из
    `weds.gguf` (там он уже отлажен на гибридных архитектурах), но падение
    импорта не должно ломать приложение - тогда вернём None, а решение о
    режиме примет `suggest_load_mode` по строке `-ngl`.
    """
    try:
        from weds.gguf import parse
        return parse(model_path).get("block_count")
    except Exception:  # noqa: BLE001 - отсутствие парсера не критично
        return None


def config_for(model_path: str, **overrides) -> ServerConfig:
    """Конфиг по умолчанию для конкретной модели.

    `--load-mode` выводится из числа слоёв, а не задаётся константой: при
    полном выносе на карту нужен `none` (иначе 5.67 ГБ остаются в системной
    памяти и машина с 16 ГБ уходит в своп), при частичном - `mmap`.
    """
    ngl = overrides.pop("ngl", 99)
    layers = model_layers(model_path)
    cfg = ServerConfig(model=model_path, ngl=ngl,
                       load_mode=overrides.pop(
                           "load_mode", suggest_load_mode(ngl, layers)),
                       **overrides)
    cfg.exe = cfg.exe or (detect_exe().get("path") or "")
    return cfg


def detect_exe(explicit: str | None = None) -> dict:
    """Найти llama-server.exe.

    Порядок: явный путь, переменная окружения, известные места этого проекта,
    PATH, затем бэкенды LM Studio (там лежит свой llama-server, он же и
    запускается движком). Возвращает и источник: UI должен показывать, откуда
    взялся бинарник, потому что сборки различаются поддержкой ggml-типов, и
    PTQ1_0 (type 143) читает только форк PrismML-Eng.
    """
    if explicit and Path(explicit).is_file():
        return {"path": explicit, "source": "задан явно", "ok": True}
    env = os.environ.get("LLAMA_SERVER")
    if env and Path(env).is_file():
        return {"path": env, "source": "LLAMA_SERVER", "ok": True}
    for hint in _EXE_HINTS:
        if Path(hint).is_file():
            return {"path": hint, "source": "проект", "ok": True}
    found = shutil.which("llama-server")
    if found:
        return {"path": found, "source": "PATH", "ok": True}
    for root in (Path.home() / ".lmstudio" / "extensions" / "backends",
                 Path.home() / ".cache" / "lm-studio" / "extensions" / "backends"):
        if not root.is_dir():
            continue
        cands = sorted(root.rglob("llama-server.exe"), reverse=True)
        if cands:
            return {"path": str(cands[0]), "source": "LM Studio", "ok": True}
    return {"path": None, "source": None,
            "ok": False, "error": "llama-server.exe не найден"}


def version(exe: str, timeout: float = 20.0) -> dict:
    """Сборка движка. Нужна, чтобы отличить сток от форка: PTQ1_0 читает
    только форк, и по номеру сборки это видно до запуска."""
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True,
                           timeout=timeout,
                           creationflags=CREATE_NO_WINDOW)
        txt = ((p.stdout or "") + (p.stderr or "")).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "error": str(e)[:200]}
    out: dict = {"ok": True, "raw": txt[:400]}
    import re
    m = re.search(r"version:\s*(\S+)", txt)
    if m:
        out["version"] = m.group(1)
    m = re.search(r"built with .*? for (\S+)", txt)
    if m:
        out["backend"] = m.group(1)
    m = re.search(r"b(\d{4,})", txt)
    if m:
        out["build"] = int(m.group(1))
    return out


# --------------------------------------------------------------------------
# конфигурация
# --------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Набор флагов запуска. Дефолты повторяют рабочий start-bonsai.cmd,
    чтобы приложение из коробки воспроизводило проверенную конфигурацию."""

    model: str
    exe: str = ""
    alias: str = ""
    host: str = "127.0.0.1"
    port: int = 8081
    ctx: int = 65536
    ngl: int = 99
    np: int = 1
    load_mode: str = "none"
    flash_attn: str = "on"
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"
    temp: float = 0.4
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    reasoning_budget: int = 1024
    jinja: bool = True
    metrics: bool = True
    # ДЕФОЛТЫ СООТВЕТСТВУЮТ ПРОДУ, А НЕ ЗАМЕРУ: флаги не передаются, движок
    # берёт свои значения (cache-ram 8192, ctx-checkpoints 32) - ровно то, с
    # чем работает start-bonsai.cmd. Замер тогда показывает то, что
    # пользователь получит на самом деле, а не другой режим.
    #
    # Замерено на Ternary-Bonsai-2-27B: за серию из пяти замеров рабочий
    # набор сервера вырос 902 -> 1845 -> 2302 -> 2448 -> 3055 МБ, а decode
    # упал с 17.29 до 7.2 tok/s. Причина в логе: `created context checkpoint
    # N of 32 (size = 149.626 MiB)`. Контрольные точки контекста дублируют
    # рекуррентное состояние в системной памяти, их до 32 штук, и за серию
    # их накопилось 23, то есть 3.4 ГБ. Дальше Windows начинает вытеснять
    # страницы, и скорость падает вдвое - тот же механизм, что и с полной
    # копией весов, только медленнее и незаметнее.
    #
    # Для живого чата кеш промптов полезен (повторный системный промпт не
    # пересчитывается), поэтому значение вынесено в конфиг: не задано -
    # дефолт движка, 0 - выключить совсем. Но знать про этот эффект надо в
    # обоих случаях: долгий сеанс с растущим контекстом копит контрольные
    # точки и постепенно замедляется.
    cache_ram: int | None = None
    # --ctx-checkpoints 0 выключает контрольные точки контекста совсем.
    # Замерено: за серию из пяти запросов сервер создал 23 контрольные точки
    # по 149.626 МБ = 3.4 ГБ, после чего decode упал с 17.29 до 7.2 tok/s.
    # С `--cache-ram 0` рост остановился, но точки всё равно создавались
    # (12 штук за серию): это разные подсистемы, и cache-ram их не выключает.
    #
    # Цена выключения измерена перемежающимся сравнением на трёх парах
    # прогонов: дефолтная конфигурация оказалась быстрее на 1.19 tok/s во
    # всех трёх парах, то есть эффект реальный, а не шум. Итог: 0 - долгий
    # сеанс с предсказуемой памятью, дефолт - максимум скорости. Это выбор
    # под задачу, а не «правильное значение».
    ctx_checkpoints: int | None = None
    # -lv 4 по умолчанию, и это не «на всякий случай». Только на четвёртом
    # уровне движок печатает `memory_breakdown_print` и строки `buffer size`,
    # то есть собственный учёт памяти по устройствам. На дефолтном третьем
    # этих строк НЕТ (проверено: лог загруженной модели 5.6 КБ без единой
    # строки про буферы). А это единственный доступный источник VRAM, когда
    # NVML заблокирован и nvidia-smi отдаёт «Failed to initialize NVML» -
    # без него приложение не может показать, влезла модель на карту или нет.
    # Лог пишется в файл, а не в консоль пользователя, так что шум не мешает.
    verbose: int | None = 4
    batch: int | None = None
    ubatch: int | None = None
    verbose: int | None = 4
    extra: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Путь нормализуется сразу. Один и тот же файл приходит то как
        # `E:\Models\x.gguf`, то как `E:/Models/x.gguf` (Git Bash и MSYS
        # переписывают обратные слэши в аргументах командной строки), а
        # сравнивается он строкой - в списке моделей, в профилях, в ключе
        # профиля. Без нормализации одно и то же выглядит как два разных
        # пути, и UI выбирает не ту модель, на которой шёл замер.
        self.model = os.path.normpath(self.model) if self.model else self.model
        if not self.alias:
            self.alias = Path(self.model).stem

    def to_argv(self) -> list[str]:
        a = [self.exe or "llama-server.exe",
             "-m", self.model,
             "--alias", self.alias,
             "--host", self.host, "--port", str(self.port),
             "-ngl", str(self.ngl), "-c", str(self.ctx), "-np", str(self.np),
             "--load-mode", self.load_mode,
             "-fa", self.flash_attn,
             "--cache-type-k", self.cache_type_k,
             "--cache-type-v", self.cache_type_v,
             "--temp", str(self.temp),
             "--top-p", str(self.top_p), "--top-k", str(self.top_k),
             "--min-p", str(self.min_p),
             "--reasoning-budget", str(self.reasoning_budget)]
        if self.jinja:
            a.append("--jinja")
        if self.metrics:
            a.append("--metrics")
        if self.cache_ram is not None:
            a += ["--cache-ram", str(self.cache_ram)]
        if self.ctx_checkpoints is not None:
            a += ["--ctx-checkpoints", str(self.ctx_checkpoints)]
        if self.batch:
            a += ["-b", str(self.batch)]
        if self.ubatch:
            a += ["-ub", str(self.ubatch)]
        if self.verbose is not None:
            a += ["-lv", str(self.verbose)]
        a += list(self.extra)
        return a

    def fingerprint(self) -> str:
        """Ключ профиля: конфиг без пути и порта.

        Порт и путь не входят намеренно - иначе один и тот же набор настроек
        считался бы новым при каждом запуске, и кэш профилей не работал бы.
        """
        import hashlib
        skip = {"exe", "port", "host", "model", "extra"}
        parts = [f"{k}={v}" for k, v in sorted(self.__dict__.items())
                 if k not in skip]
        parts.append("extra=" + ",".join(sorted(self.extra)))
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

    def describe(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "extra"} | \
               ({"extra": self.extra} if self.extra else {})

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "ServerConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------
# сервер
# --------------------------------------------------------------------------

class LlamaServer:
    """Запущенный llama-server с живым логом и честной остановкой."""

    def __init__(self, cfg: ServerConfig, log_path: str | None = None,
                 console: bool = False, env: dict | None = None):
        self.cfg = cfg
        self.log_path = log_path
        self.console = console
        self.env = env or {}
        self.proc: subprocess.Popen | None = None
        self.lines: deque[str] = deque(maxlen=6000)
        self.health: dict = {"state": "down", "t_503": None, "t_200": None,
                             "t_start": None, "t_end": None}
        self._reader: threading.Thread | None = None
        self._logfh = None
        self._started = 0.0

    # -- свойства ----------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    @property
    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    # -- запуск ------------------------------------------------------------

    def start(self, wait: bool = False, timeout: float = 900.0) -> "LlamaServer":
        cfg = self.cfg
        if not Path(cfg.model).is_file():
            raise FileNotFoundError(f"нет файла модели: {cfg.model}")
        exe = cfg.exe or detect_exe().get("path")
        if not exe:
            raise FileNotFoundError("llama-server.exe не найден")
        cfg.exe = exe
        cfg.port = free_port(cfg.port, cfg.host)

        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._logfh = open(self.log_path, "w", encoding="utf-8", errors="replace")

        env = {**os.environ, **self.env}
        flags = 0 if self.console else CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            cfg.to_argv(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, bufsize=1, text=True,
            encoding="utf-8", errors="replace", env=env,
            creationflags=flags)
        self._started = time.time()
        self.health["t_start"] = 0.0
        self._reader = threading.Thread(target=self._pump, name="llama-log",
                                        daemon=True)
        self._reader.start()
        if wait:
            self.wait_health(timeout)
        return self

    def _pump(self) -> None:
        """Читать stdout построчно и дублировать в файл.

        Именно чтение потоком, а не `stdout=file`: при блоковой буферизации
        последние строки теряются, если процесс убивают, а не завершают.
        """
        assert self.proc and self.proc.stdout
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\r\n")
                self.lines.append(line)
                if self._logfh:
                    self._logfh.write(line + "\n")
                    self._logfh.flush()
        except (ValueError, OSError):
            pass
        finally:
            if self._logfh:
                try:
                    self._logfh.close()
                except OSError:
                    pass
                self._logfh = None

    # -- здоровье ----------------------------------------------------------

    def health_once(self, timeout: float = 1.0) -> str:
        """`ready` (200), `loading` (503) или `down`."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health",
                                        timeout=timeout) as r:
                body = r.read(200)
            if r.status == 200 and b'"ok"' in body:
                return "ready"
            return "loading"
        except urllib.error.HTTPError as e:
            return "loading" if e.code in (503, 500) else "down"
        except (urllib.error.URLError, OSError, ValueError):
            return "down"

    def wait_health(self, timeout: float = 900.0, interval: float = 0.5,
                    on_tick=None) -> dict:
        """Ждать готовности, попутно фиксируя переход 503 -> 200.

        Тайминги нужны отдельно от «готово/не готово»: 503 на старте - это
        нормальная фаза загрузки весов, и знать, сколько она длится, полезно
        (на Ternary-Bonsai 27B при ctx 65536 это ~46-53 с).
        """
        t0 = time.time()
        while True:
            if self.proc and self.proc.poll() is not None:
                self.health.update(state="dead", t_end=round(time.time() - t0, 2),
                                   rc=self.proc.returncode)
                return self.health
            state = self.health_once()
            el = time.time() - t0
            if state == "loading" and self.health["t_503"] is None:
                self.health["t_503"] = round(el, 2)
            if state == "ready":
                self.health.update(state="ready", t_503=self.health["t_503"],
                                   t_200=round(el, 2), t_end=round(el, 2))
                return self.health
            if el > timeout:
                self.health.update(state="timeout", t_end=round(el, 2))
                return self.health
            if on_tick:
                on_tick(round(el, 2), state)
            time.sleep(interval)

    # -- метрики и лог -----------------------------------------------------

    def metrics(self, timeout: float = 5.0) -> dict:
        """Счётчики Prometheus. Дают то, чего нет в логе: сколько токенов
        префилла взято из кеша (prompt_tokens_total считает только
        некешированные)."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/metrics",
                                        timeout=timeout) as r:
                txt = r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError):
            return {}
        out: dict = {}
        for line in txt.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name = parts[0].split("{")[0].strip()
            try:
                out[name] = float(parts[1])
            except ValueError:
                continue
        return out

    def log_text(self, tail: int | None = None) -> str:
        lines = list(self.lines)
        if tail:
            lines = lines[-tail:]
        return "\n".join(lines)

    def buffers(self) -> dict:
        """Разбор лога: что ушло в VRAM, что осталось в системной памяти."""
        return probe.log_buffers(self.log_path) if self.log_path else {}

    def wait_for_log(self, pattern: str, timeout: float = 300.0) -> bool:
        import re
        rx = re.compile(pattern)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if any(rx.search(l) for l in list(self.lines)):
                return True
            if not self.alive:
                return False
            time.sleep(0.25)
        return False

    # -- остановка ---------------------------------------------------------

    def stop(self, timeout: float = 10.0) -> dict:
        """Остановить сервер вместе с деревом процессов.

        `terminate()` не всегда достаточен: у проекта есть супервизор,
        поднимающий llama-server заново через 5 секунд, поэтому дерево валится
        принудительно по pid, а не по имени образа (по имени убьётся и чужой
        сервер, поднятый пользователем руками).
        """
        out: dict = {"was_alive": self.alive, "killed": False}
        if not self.proc:
            return out
        pid = self.proc.pid
        if self.alive:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if self.alive:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, timeout=20,
                               creationflags=CREATE_NO_WINDOW)
                out["killed"] = True
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                self.proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if self._reader:
            self._reader.join(timeout=3)
        if self._logfh:
            try:
                self._logfh.close()
            except OSError:
                pass
            self._logfh = None
        out["rc"] = self.proc.returncode
        out["uptime_s"] = round(time.time() - self._started, 1) if self._started else None
        out["port_closed"] = not port_open(self.cfg.port, self.cfg.host)
        return out

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False
