"""Сторож памяти: замер не имеет права уронить машину.

Зачем отдельный модуль, а не проверка внутри цикла бенчмарка.

На этой машине (16 ГБ, RTX 2060 SUPER 8 ГБ) уже случался полный фриз: сервер
поднимал модель, физическая память уходила в ноль, система уходила в своп и
работать становилось нельзя. Требование пользователя после этого - «всё должно
умещаться в моё железо», и оно распространяется на любой прогон, включая
заведомо неудачные конфиги. Значит, у каждого запуска сервера должен быть
внешний наблюдатель, который убьёт процесс раньше, чем система встанет.

Два критерия, а не один:

  * `avail_mb` - свободная физическая память. Полезно, но само по себе врёт:
    пик расхода приходится НА ЗАГРУЗКУ модели, и порог по физике срабатывал
    на здоровой загрузке (2486 МБ свободных при пороге 2500), прерывая
    нормальный прогон.

  * `commit_avail_mb` - запас коммита. Вот это и есть объективный признак
    фриза: когда он кончается, падают сами выделения памяти, и система
    перестаёт отзываться. Порог по коммиту срабатывает на реальной угрозе,
    а не на нормальном пике.

Пороги по умолчанию (1500 / 900 МБ) подобраны на этой машине: здоровая
загрузка при `--load-mode none` не опускает свободную физику ниже ~7.8 ГБ,
при mmap - ниже ~1.9 ГБ. То есть 1500 МБ не мешает ни одному рабочему
конфигу и ловит только то, что действительно вот-вот заморозит систему.

Сторож сам ничего не убивает: он детектор. Кого убивать, знает вызывающий
(`on_trip`), потому что это зависит от того, чем он управляет - сервером,
чужой программой или ничем.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from . import probe

DEFAULT_FLOOR_MB = 1500
DEFAULT_COMMIT_FLOOR_MB = 900


class MeasurementAborted(RuntimeError):
    """Замер недействителен: сторож сработал, цифрам доверять нельзя."""


@dataclass
class Sample:
    t: float
    avail_mb: int
    commit_avail_mb: int | None
    proc_ws_mb: int | None
    phase: str

    def to_dict(self) -> dict:
        return {"t": round(self.t, 3), "avail_mb": self.avail_mb,
                "commit_avail_mb": self.commit_avail_mb,
                "proc_ws_mb": self.proc_ws_mb, "phase": self.phase}


class MemoryWatchdog:
    """Считает свободную память в фоне и запоминает низшую точку.

    Использование:

        with MemoryWatchdog(on_trip=srv.kill) as wd:
            srv.start()
            wd.pid = srv.pid
            wd.phase = "load"
            srv.wait_health()
            wd.phase = "measure"
            ...
        wd.assert_ok()          # бросит, если прогон был невалиден
        print(wd.report())
    """

    def __init__(self, floor_mb: int = DEFAULT_FLOOR_MB,
                 commit_floor_mb: int = DEFAULT_COMMIT_FLOOR_MB,
                 interval: float = 0.25, on_trip=None, pid: int | None = None,
                 max_samples: int = 20000, phase: str = "idle"):
        self.floor_mb = floor_mb
        self.commit_floor_mb = commit_floor_mb
        self.interval = interval
        self.on_trip = on_trip
        self.pid = pid
        self.max_samples = max_samples
        self.phase = phase

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[Sample] = []
        self._tripped: dict | None = None
        self._t0 = 0.0
        self._baseline: Sample | None = None
        self._fires = 0

    # -- управление --------------------------------------------------------

    def start(self) -> "MemoryWatchdog":
        if self._thread and self._thread.is_alive():
            return self
        self._t0 = time.time()
        self._stop.clear()
        self._baseline = self._sample()
        self._thread = threading.Thread(target=self._loop, name="mem-watchdog",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> "MemoryWatchdog":
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        return self

    def __enter__(self) -> "MemoryWatchdog":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def kill(self) -> None:
        """Ручное срабатывание: останавливает наблюдение и зовёт on_trip."""
        self._fire({"reason": "manual", "value_mb": None,
                    "limit_mb": None, "phase": self.phase})

    # -- состояние ---------------------------------------------------------

    @property
    def tripped(self) -> dict | None:
        with self._lock:
            return dict(self._tripped) if self._tripped else None

    @property
    def baseline(self) -> Sample | None:
        return self._baseline

    @property
    def samples(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)

    def assert_ok(self) -> None:
        t = self.tripped
        if t:
            raise MeasurementAborted(
                f"сторож сработал ({t['reason']}: {t['value_mb']} МБ < "
                f"{t['limit_mb']} МБ, фаза {t['phase']}) - замер недействителен")

    # -- внутренности ------------------------------------------------------

    def _sample(self) -> Sample:
        sm = probe.system_memory()
        sn = Sample(t=time.time() - self._t0, avail_mb=sm.get("avail_mb") or 0,
                    commit_avail_mb=sm.get("commit_avail_mb"), proc_ws_mb=None,
                    phase=self.phase)
        if self.pid:
            pm = probe.process_memory(self.pid) or {}
            sn.proc_ws_mb = pm.get("ws_mb")
        return sn

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            sn = self._sample()
            with self._lock:
                if len(self._samples) < self.max_samples:
                    self._samples.append(sn)
                elif self._samples:
                    # Хвост важнее начала: пик обычно ближе к концу загрузки.
                    self._samples[-1] = sn
            if self._tripped:
                continue
            if self.commit_floor_mb and sn.commit_avail_mb is not None \
                    and sn.commit_avail_mb < self.commit_floor_mb:
                self._fire({"reason": "commit", "value_mb": sn.commit_avail_mb,
                            "limit_mb": self.commit_floor_mb, "phase": sn.phase,
                            "sample": sn.to_dict()})
            elif sn.avail_mb < self.floor_mb:
                self._fire({"reason": "avail", "value_mb": sn.avail_mb,
                            "limit_mb": self.floor_mb, "phase": sn.phase,
                            "sample": sn.to_dict()})

    def _fire(self, rec: dict) -> None:
        with self._lock:
            if self._tripped:
                return
            self._tripped = {**rec, "t": round(time.time() - self._t0, 3)}
            self._fires += 1
        cb = self.on_trip
        if cb:
            try:
                cb(rec)
            except Exception:  # noqa: BLE001 - убийство не должно падать
                pass

    # -- отчёт -------------------------------------------------------------

    def report(self) -> dict:
        with self._lock:
            samples = list(self._samples)
            tripped = dict(self._tripped) if self._tripped else None
        out: dict = {"floor_mb": self.floor_mb,
                     "commit_floor_mb": self.commit_floor_mb,
                     "phase": self.phase,
                     "tripped": tripped}
        if self._baseline:
            out["baseline"] = self._baseline.to_dict()
        if not samples:
            out["samples"] = 0
            return out
        low = min(samples, key=lambda s: s.avail_mb)
        out["samples"] = len(samples)
        out["seconds"] = round(samples[-1].t, 1)
        out["min_avail_mb"] = low.avail_mb
        out["min_avail_phase"] = low.phase
        out["min_avail_t"] = round(low.t, 2)
        commits = [s.commit_avail_mb for s in samples if s.commit_avail_mb]
        if commits:
            out["min_commit_avail_mb"] = min(commits)
        wss = [s.proc_ws_mb for s in samples if s.proc_ws_mb]
        if wss:
            out["min_proc_ws_mb"] = min(wss)
            out["max_proc_ws_mb"] = max(wss)
        if self._baseline:
            out["spent_mb"] = self._baseline.avail_mb - low.avail_mb
        return out

    def timeline(self, step: int = 1) -> list[list[float]]:
        """Ряд для графика: [[t, avail_mb, proc_ws_mb], ...]."""
        with self._lock:
            return [[round(s.t, 2), s.avail_mb, s.proc_ws_mb]
                    for s in self._samples[::step]]
