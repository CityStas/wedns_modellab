"""Подбор и применение параметров загрузки под конкретное железо.

Логика: измерить кривую памяти движка (вес модели + цена контекста в байтах
на токен), узнать бюджет VRAM и подобрать максимальный контекст, который
влезает с запасом. Затем записать конфиг.

Кривая снимается двумя вызовами `lms load --estimate-only` — модель при этом
не загружается, система не нагружается.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import lmstudio
from .lmstudio import LOAD_KEYS, WRAPPED_LOAD

GIB = 1024 ** 3

# Резерв VRAM под рабочий стол / композитор Windows.
DESKTOP_RESERVE_BYTES = int(0.8 * GIB)
# Доля VRAM, которую разрешаем занять модели.
BUDGET_FRACTION = 0.92

# Оценщик LM Studio считает KV по своей базовой точности. Эмпирически она
# соответствует примерно q8_0: реальный q4_0 даёт около половины от неё.
KV_FACTOR = {"q8_0": 1.0, "q4_0": 0.5, "F16": 2.0}

_EST_GPU = re.compile(r"Estimated GPU Memory:\s*([\d\s.,]+)\s*GiB")
_EST_CTX = re.compile(r"Context Length:\s*([\d\s]+)")


# --------------------------------------------------------------------------- #
# Железо
# --------------------------------------------------------------------------- #

@dataclass
class Hardware:
    gpu_name: str | None = None
    vram_bytes: int | None = None
    cpu_physical_cores: int | None = None
    ram_bytes: int | None = None
    source: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "gpu": self.gpu_name,
            "vram_gib": round(self.vram_bytes / GIB, 2) if self.vram_bytes else None,
            "cpu_physical_cores": self.cpu_physical_cores,
            "ram_gib": round(self.ram_bytes / GIB, 1) if self.ram_bytes else None,
            "source": self.source,
            "notes": self.notes,
        }


def _ps_run(command: str, timeout: float = 25.0) -> str | None:
    """Выполнить команду в PowerShell и вернуть stdout (или None).

    Общий хелпер: на Windows надёжнее всего узнавать железо через CIM, а не через
    сторонние библиотеки — зависимостей у проекта нет.
    """
    for shell in (["powershell", "-NoProfile", "-NonInteractive", "-Command"],
                  ["pwsh", "-NoProfile", "-NonInteractive", "-Command"]):
        try:
            proc = subprocess.run(shell + [command], capture_output=True,
                                  text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            continue
        # Код возврата PowerShell ненадёжен: скрипт, где последняя итерация
        # ничего не вывела, отдаёт 1 при полностью корректном выводе. Поэтому
        # непустой stdout считаем успехом.
        if proc.returncode == 0 or (proc.stdout or "").strip():
            return proc.stdout or ""
    return None


def _physical_cores() -> tuple[int | None, str]:
    """Физические ядра. os.cpu_count() даёт логические, а для llama.cpp нужно
    именно физическое число: hyper-threading на инференсе только мешает."""
    out = _ps_run("(Get-CimInstance Win32_Processor).NumberOfCores -join ','")
    if out:
        out = out.strip()
        if out and out[0].isdigit():
            cores = sum(int(x) for x in out.split(",") if x.strip().isdigit())
            if cores:
                return cores, "cim"

    logical = os.cpu_count()
    if not logical:
        return None, "unknown"
    # Эвристика: у большинства настольных CPU есть HT, значит физических вдвое меньше.
    return (max(1, logical // 2) if logical > 4 else logical), "heuristic(logical/2)"


# Класс «Видеоадаптеры» в реестре Windows.
_GPU_CLASS = (r"HKLM:\SYSTEM\CurrentControlSet\Control\Class"
              r"\{4d36e968-e325-11ce-bfc1-08002be10318}")


def _vram_from_registry() -> tuple[str | None, int | None]:
    """Реальный размер VRAM из реестра Windows.

    Зачем не CIM: `Win32_VideoController.AdapterRAM` — 32-битное поле, и для карт
    больше 4 ГиБ оно отдаёт мусор (на 8-гиговой 2060 Super вернуло 4.0 ГиБ).
    `HardwareInformation.qwMemorySize` — 64-битное, отдаёт точное значение.

    Это рабочий обход, когда NVML заблокирован (типовой случай в песочницах и
    под WSL), а nvidia-smi падает с «Failed to initialize NVML».
    """
    if os.name != "nt":
        return None, None
    out = _ps_run(
        "$base='" + _GPU_CLASS + "';"
        "Get-ChildItem $base -ErrorAction SilentlyContinue | ForEach-Object {"
        " $d=Get-ItemProperty -Path $_.PSPath -ErrorAction SilentlyContinue;"
        " $q=$d.'HardwareInformation.qwMemorySize';"
        " if ($q) { \"$($d.DriverDesc)|$q\" } }; exit 0"
    )
    if not out:
        return None, None

    best_name: str | None = None
    best = 0
    for line in out.splitlines():
        name, _, raw = line.strip().partition("|")
        raw = raw.strip()
        if not raw.isdigit():
            continue
        value = int(raw)
        if value > best:
            best, best_name = value, (name.strip() or None)
    return (best_name, best) if best else (None, None)


def _ram_bytes() -> int | None:
    """Объём системной RAM. Нужен, чтобы предупредить о нехватке, когда модель
    частично уходит на CPU."""
    if os.name == "nt":
        out = _ps_run("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        if out and out.strip().isdigit():
            return int(out.strip())
        return None
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def detect_hardware() -> Hardware:
    """Пробует nvidia-smi, затем реестр Windows. NVML часто заблокирован —
    тогда возвращаем частичные данные и честно об этом пишем."""
    hw = Hardware()

    cores, src = _physical_cores()
    hw.cpu_physical_cores = cores
    if cores and src.startswith("heuristic"):
        hw.notes.append(
            f"Число физических ядер определено эвристикой ({src}). "
            f"Проверь вручную: для llama.cpp нужно число физических ядер, не потоков."
        )

    hw.ram_bytes = _ram_bytes()

    # GPU через nvidia-smi
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            proc = subprocess.run(
                [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            )
            out = (proc.stdout or "").strip()
            if proc.returncode == 0 and out and "Failed" not in out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                if len(parts) >= 2:
                    hw.gpu_name = parts[0]
                    hw.vram_bytes = int(float(parts[1]) * 1024 * 1024)
                    hw.source = "nvidia-smi"
            else:
                hw.notes.append(f"nvidia-smi не отдал данные: {out[:120] or proc.stderr[:120]}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            hw.notes.append(f"nvidia-smi недоступен: {e}")
    else:
        hw.notes.append("nvidia-smi не найден в PATH")

    # Обход через реестр — работает и без NVML, и на картах AMD/Intel.
    if hw.vram_bytes is None:
        name, vram = _vram_from_registry()
        if vram:
            hw.gpu_name = hw.gpu_name or name
            hw.vram_bytes = vram
            hw.source = "registry(qwMemorySize)"
            hw.notes.append(
                "VRAM взята из реестра Windows — NVML недоступен. "
                "Значение точное, но это вся видеопамять без учёта занятого рабочим столом."
            )

    if hw.vram_bytes is None:
        hw.notes.append(
            "VRAM не определена автоматически (NVML заблокирован, реестр не ответил). "
            "Укажи бюджет вручную: weds tune --vram-gib 8"
        )

    return hw


# --------------------------------------------------------------------------- #
# Кривая памяти движка
# --------------------------------------------------------------------------- #

@dataclass
class MemoryCurve:
    """Вес модели + цена контекста. Цена измерена на базовой точности оценщика
    (≈q8_0), поэтому для других квантов KV применяется KV_FACTOR."""

    weights_bytes: int
    bytes_per_token: int
    samples: list[tuple[int, int]] = field(default_factory=list)
    model_key: str | None = None

    def effective_bytes_per_token(self, kv: str = "q8_0") -> int:
        return max(1, int(self.bytes_per_token * KV_FACTOR.get(kv, 1.0)))

    def total_for(self, context: int, kv: str = "q8_0") -> int:
        return self.weights_bytes + self.effective_bytes_per_token(kv) * context

    def max_context(self, budget_bytes: int, kv: str = "q8_0") -> int:
        per_token = self.effective_bytes_per_token(kv)
        if per_token <= 0:
            return 0
        return max(0, int((budget_bytes - self.weights_bytes) / per_token))

    def describe(self) -> dict:
        return {
            "weights_gib": round(self.weights_bytes / GIB, 2),
            "kib_per_token_base": round(self.bytes_per_token / 1024, 1),
            "samples": [{"ctx": c, "total_gib": round(t / GIB, 2)} for c, t in self.samples],
        }


def _parse_estimate(output: str) -> tuple[int, int] | None:
    """Разбирает вывод `lms load --estimate-only`.

    Внимание: LM Studio печатает разряды через неразрывный пробел ('32 768'),
    поэтому вычищаем все нецифровые символы, а не только обычный пробел.
    """
    ctx_m = _EST_CTX.search(output)
    gpu_m = _EST_GPU.search(output)
    if not gpu_m:
        return None

    gib_raw = gpu_m.group(1).replace("\xa0", "").replace(" ", "").replace(",", "")
    try:
        gib = float(gib_raw)
    except ValueError:
        return None

    ctx = 0
    if ctx_m:
        digits = re.sub(r"\D", "", ctx_m.group(1))
        ctx = int(digits) if digits else 0

    return ctx, int(gib * GIB)


def measure_memory_curve(home: Path, model_key: str,
                         points: tuple[int, ...] = (8192, 32768)) -> MemoryCurve | None:
    """Два замера оценки памяти дают вес модели и цену контекста на токен.

    Это единственный надёжный способ: ручной расчёт по метаданным GGUF для
    гибридных архитектур (attention + SSM) врёт в разы.
    """
    samples: list[tuple[int, int]] = []
    for ctx in points:
        code, out = lmstudio.run_lms(
            home, ["load", model_key, "-c", str(ctx), "--estimate-only", "-y"], timeout=120
        )
        parsed = _parse_estimate(out)
        if parsed and parsed[0] > 0:
            samples.append(parsed)

    if len(samples) < 2:
        return None

    (c1, t1), (c2, t2) = samples[0], samples[1]
    if c2 == c1:
        return None
    per_token = (t1 - t2) / (c1 - c2)
    weights = t1 - per_token * c1

    return MemoryCurve(
        weights_bytes=max(0, int(weights)),
        bytes_per_token=max(1, int(per_token)),
        samples=samples,
        model_key=model_key,
    )


# --------------------------------------------------------------------------- #
# Рекомендации
# --------------------------------------------------------------------------- #

@dataclass
class Recommendation:
    model_key: str
    context: int
    vram_budget_bytes: int
    curve: MemoryCurve | None = None
    parallel: int = 1
    offload: str = "max"
    flash_attention: bool = True
    eval_batch: int = 256
    physical_batch: int = 256
    context_checkpoints: int = 32
    k_cache_quant: str = "q4_0"
    v_cache_quant: str = "q4_0"
    offload_kv: bool = True
    cpu_threads: int | None = None
    agent_prompt_tokens: int = 38000
    notes: list[str] = field(default_factory=list)

    def load_fields(self) -> dict:
        out = {
            "auto_fit": False,
            "context": self.context,
            "parallel": self.parallel,
            "offload": self.offload,
            "flash_attention": self.flash_attention,
            "eval_batch": self.eval_batch,
            "phys_batch": self.physical_batch,
            "context_checkpoints": self.context_checkpoints,
            "k_cache_quant": self.k_cache_quant,
            "v_cache_quant": self.v_cache_quant,
            "offload_kv": self.offload_kv,
        }
        if self.cpu_threads:
            out["cpu_threads"] = self.cpu_threads
        return out

    def describe(self) -> dict:
        out = {
            "model_key": self.model_key,
            "context": self.context,
            "parallel": self.parallel,
            "offload": self.offload,
            "flash_attention": self.flash_attention,
            "eval_batch": self.eval_batch,
            "k_cache_quant": self.k_cache_quant,
            "v_cache_quant": self.v_cache_quant,
            "offload_kv": self.offload_kv,
            "vram_budget_gib": round(self.vram_budget_bytes / GIB, 2),
            "notes": self.notes,
        }
        if self.curve:
            out["memory"] = self.curve.describe()
            out["projected_total_gib"] = round(
                self.curve.total_for(self.context, self.k_cache_quant) / GIB, 2
            )
        return out


def recommend(
    home: Path,
    model_key: str,
    *,
    vram_bytes: int | None = None,
    budget_bytes: int | None = None,
    agent_prompt_tokens: int = 38000,
    context: int | None = None,
    prefer_q8: bool = False,
    offload_kv: bool | None = None,
    cpu_threads: int | None = None,
) -> Recommendation:
    """Подбирает параметры загрузки.

    Контекст берётся не меньше agent_prompt_tokens с запасом 10%: это то,
    на чём падает агент, если поставить «сколько влезет по памяти».

    `budget_bytes` — явный бюджет памяти под модель. Нужен, когда расчётный
    бюджет (доля VRAM минус резерв под рабочий стол) выходит слишком
    пессимистичным: оценщик LM Studio систематически завышает расход.
    """
    notes: list[str] = []

    if budget_bytes:
        budget = int(budget_bytes)
        notes.append(
            f"Бюджет задан вручную: {round(budget / GIB, 2)} ГиБ. "
            f"Проверь после загрузки, что генерация не провалилась."
        )
    elif vram_bytes:
        budget = int(vram_bytes * BUDGET_FRACTION) - DESKTOP_RESERVE_BYTES
    else:
        budget = 0
        notes.append("Бюджет VRAM неизвестен — контекст подобран только под промпт агента.")

    curve = measure_memory_curve(home, model_key)
    if curve is None:
        notes.append(
            "Кривую памяти снять не удалось (нет lms CLI или сервер не отвечает). "
            "Контекст подобран по размеру промпта агента, без проверки по памяти."
        )

    floor_ctx = int(agent_prompt_tokens * 1.10)

    # Выбор кванта KV: сначала пробуем более точный q8_0, откатываемся на q4_0,
    # если по бюджету не влезает. На 8 ГБ с 9B это происходит почти всегда.
    kq = "q8_0" if prefer_q8 else "q4_0"

    if context is not None:
        chosen = context
        if curve and budget > 0:
            fits = curve.max_context(budget, kq)
            if fits < chosen:
                notes.append(
                    f"Заданный контекст {chosen} не влезает в бюджет: по памяти "
                    f"доступно {fits} токенов. Возможен выход за VRAM."
                )
    elif curve and budget > 0:
        fits_q8 = curve.max_context(budget, "q8_0")
        fits_q4 = curve.max_context(budget, "q4_0")

        if prefer_q8 or fits_q8 >= floor_ctx:
            kq = "q8_0"
            chosen = min(fits_q8, 131072)
            notes.append(f"q8_0 KV влезает: до {fits_q8} токенов.")
        elif fits_q4 >= floor_ctx:
            kq = "q4_0"
            chosen = min(fits_q4, 131072)
            notes.append(
                f"q8_0 KV влезает только до {fits_q8} токенов (нужно минимум {floor_ctx}), "
                f"взял q4_0 — он вдвое компактнее и даёт до {fits_q4}."
            )
        else:
            kq = "q4_0"
            chosen = floor_ctx
            notes.append(
                f"ВНИМАНИЕ: по проекции не влезает даже с q4_0 — доступно {fits_q4} токенов, "
                f"а промпт агента требует {floor_ctx}. Контекст выставлен по промпту. "
                f"Проекция считается оценщиком LM Studio, который завышает расход, поэтому "
                f"на живом железе это часто работает (проверь `weds bench` после загрузки). "
                f"Если работает — бюджет можно задать вручную: "
                f"`weds tune --budget-gib <сколько реально есть> --apply`. "
                f"Если генерация провалилась — нужна модель меньше, либо освободить VRAM."
            )
        chosen = max(floor_ctx, chosen)
    else:
        chosen = max(floor_ctx, 32768)
        if not curve:
            notes.append(
                "Кривую памяти снять не удалось, контекст подобран только под промпт агента."
            )

    # Округляем ВВЕРХ до кратного 1024: округление вниз могло бы опустить
    # контекст ниже порога, при котором влезает промпт агента.
    chosen = -(-chosen // 1024) * 1024

    if curve and budget > 0:
        notes.append(
            "Оценщик LM Studio завышает расход: на 8-гиговой карте проекция показывала "
            "«не влезает» при 49152, а модель на этом контексте выдавала 52 tok/s. "
            "Поэтому проекция — это предупреждение, а не приговор: сверяйся с `weds bench`."
        )

    if offload_kv is not False:
        notes.append(
            "KV-кеш держится в видеопамяти (offloadKVCacheToGpu). Замерено на 9B Q4_K_M: "
            "генерация 34 → 47 tok/s. Плата — VRAM растёт вместе с контекстом "
            "(около 13 КиБ на токен у гибридных архитектур). Если после загрузки генерация "
            "провалилась — верни кеш в RAM: --kv-in-ram."
        )

    rec = Recommendation(
        model_key=model_key,
        context=chosen,
        vram_budget_bytes=budget if budget > 0 else 0,
        curve=curve,
        k_cache_quant=kq,
        v_cache_quant=kq,
        offload_kv=offload_kv if offload_kv is not None else True,
        agent_prompt_tokens=agent_prompt_tokens,
        cpu_threads=cpu_threads,
        notes=notes,
    )
    return rec


def apply(home: Path, rec: Recommendation, *, model_key: str | None = None) -> dict:
    """Пишет конфиг. Возвращает отчёт с путями и бэкапом."""
    key = model_key or rec.model_key
    cfg_path = lmstudio.find_config_by_key(home, key)
    if cfg_path is None:
        return {
            "ok": False,
            "error": f"конфиг для модели '{key}' не найден",
            "hint": "Модель должна быть проиндексирована в LM Studio. "
                    "Проверь: weds models",
        }

    backup = lmstudio.write_config(cfg_path, load=rec.load_fields(), backup=True)
    return {
        "ok": True,
        "config_path": str(cfg_path),
        "backup_path": str(backup) if backup else None,
        "applied": rec.load_fields(),
        "note": "Перезагрузи модель, чтобы параметры применились: "
                "lms unload --all && lms load <key> -y",
    }


def restore(config_path: Path) -> dict:
    """Возвращает конфиг из бэкапа, созданного apply()."""
    backup = Path(str(config_path) + lmstudio.BACKUP_SUFFIX)
    if not backup.is_file():
        return {"ok": False, "error": f"бэкап не найден: {backup}"}
    shutil.copy2(backup, config_path)
    return {"ok": True, "restored_from": str(backup), "config_path": str(config_path)}
