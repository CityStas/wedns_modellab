"""Датчики: какое железо, сколько памяти свободно, сколько съел процесс.

Зачем это отдельным модулем, а не строчкой в бенчмарке: замер скорости без
снимка памяти невоспроизводим. Одна и та же конфигурация на этой машине даёт
9.87 tok/s при 1.9 ГБ свободных и 17.90 tok/s при 7.7 ГБ - измерено 2026-09-23
на Ternary-Bonsai-2-27B, ctx 65536, -ngl 99. Разницу объясняет не скорость
железа, а давление на память, и без этих цифр рядом с tok/s любой подбор
конфига оптимизирует по плавающей цели.

Главная метрика, которой нет в других инструментах:

    host_ratio = рабочий_набор_процесса / размер_файла_модели

Около 1.0 - движок держит в системной памяти полную копию весов, хотя все слои
уже лежат на видеокарте (дефолтный mmap в llama.cpp). Это и есть причина
просевшей скорости на машине с 16 ГБ. Около 0.15 - копия отпущена
(--load-mode none), память свободна.

Зависимостей нет: ctypes + winreg + стандартная библиотека.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

IS_WINDOWS = sys.platform == "win32"
MIB = 1024 * 1024

# Строка вида: `load_tensors:        CUDA0 model buffer size =  5395.33 MiB`
# Устройство и имя буфера идут ДВУМЯ отдельными токенами, и это принципиально:
# именно по устройству делится бюджет VRAM и системной памяти. Раньше здесь
# стоял шаблон на одно слово, он брал последний токен («model»), и все буферы
# складывались в один мешок - CUDA0 в итоге давал 0 MiB.
_DEV_BUFFER_RE = re.compile(
    r"(?:([A-Za-z][\w.]*)\s+)?(\w+)\s+buffer size\s*=\s*([\d.]+)\s*MiB")

# Строка вида: `|   - CUDA0 (RTX 2060 SUPER) |  8191 = 7000 + (7097 =  5395 +    1301 +     400) +       -5905 |`
# Это собственный бюджет движка, посчитанный ДО загрузки. Единственный
# источник VRAM, который работает, когда NVML заблокирован: и total, и free,
# и раскладка model/context/compute по каждому устройству.
_BREAKDOWN_RE = re.compile(
    r"memory_breakdown_print:.*?\|\s*-\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$",
    re.M)
_DEV_FREE_RE = re.compile(
    r"using device (\S+).*?-\s*(\d+)\s*MiB free")

# `common_fit_params: failed to fit params` печатается ВСЕГДА, когда -ngl задан
# руками: движок сообщает, что не будет подгонять параметры, потому что их
# зафиксировал пользователь. Это не ошибка, и показывать её красным в UI -
# значит приучать игнорировать красное.
_BENIGN_RE = re.compile(
    r"common_fit_params: failed to fit params|"
    r"cannot meet free memory target|"
    r"projected to use .* device memory vs\.|"
    r"trying to reproduce them with -fit off", re.I)


# --------------------------------------------------------------------------
# системная память
# --------------------------------------------------------------------------

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = ([("cb", wt.DWORD)]
                + [(n, ctypes.c_size_t) for n in
                   ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                    "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
                    "KernelNonpaged", "PageSize")]
                + [("HandleCount", wt.DWORD), ("ProcessCount", wt.DWORD),
                   ("ThreadCount", wt.DWORD)])


def system_memory() -> dict:
    """Свободная физика и запас коммита.

    ullAvailPhys - сколько реально можно выделить сейчас.
    ullAvailPageFile - запас коммита; когда он доходит до нуля, машина
    замирает, потому что начинают падать сами выделения. Это и есть
    объективный признак «фриза», а не процент загрузки.
    """
    if not IS_WINDOWS:
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            return {"total_mb": os.sysconf("SC_PHYS_PAGES") * page // MIB,
                    "avail_mb": os.sysconf("SC_AVPHYS_PAGES") * page // MIB,
                    "commit_limit_mb": None, "commit_avail_mb": None,
                    "load_pct": None, "system_cache_mb": None}
        except (ValueError, OSError, AttributeError):
            return {}
    m = _MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return {}
    out = {"total_mb": m.ullTotalPhys // MIB, "avail_mb": m.ullAvailPhys // MIB,
           "commit_limit_mb": m.ullTotalPageFile // MIB,
           "commit_avail_mb": m.ullAvailPageFile // MIB,
           "load_pct": m.dwMemoryLoad, "system_cache_mb": None}
    pi = _PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        out["system_cache_mb"] = pi.SystemCache * pi.PageSize // MIB
    return out


def process_memory(pid: int) -> dict | None:
    """Резидентная и закоммиченная память одного процесса.

    WorkingSetSize - сколько физически занимает сейчас (сюда попадают и
    страницы отображённого файла модели - именно они и создают host_ratio).
    PagefileUsage - коммит, он больше и не весь резидентный.
    """
    if not IS_WINDOWS:
        return _proc_mem_posix(pid)
    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        return {"pid": pid,
                "ws_mb": pmc.WorkingSetSize // MIB,
                "commit_mb": pmc.PagefileUsage // MIB,
                "peak_ws_mb": pmc.PeakWorkingSetSize // MIB,
                "peak_commit_mb": pmc.PeakPagefileUsage // MIB}
    finally:
        k32.CloseHandle(h)


def _proc_mem_posix(pid: int) -> dict | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return None
    def kb(key: str) -> int:
        m = re.search(rf"^{key}:\s+(\d+) kB", txt, re.M)
        return int(m.group(1)) // 1024 if m else 0
    return {"pid": pid, "ws_mb": kb("VmRSS"), "commit_mb": kb("VmSize"),
            "peak_ws_mb": kb("VmHWM"), "peak_commit_mb": 0}


def trim_working_set(pid: int) -> bool | None:
    """Отпустить резидентные страницы процесса, не трогая видеокарту.

    Веса уже в VRAM, поэтому выбрасывается только чистая файловая копия GGUF -
    её всегда можно дочитать с диска. Альтернатива флагу --load-mode none для
    уже запущенного сервера: на живом процессе освобождало 2.4 -> 8.2 ГБ и
    поднимало decode с 7.25 до 15.57 tok/s.
    """
    if not IS_WINDOWS:
        return None
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_SET_QUOTA = 0x0100
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_SET_QUOTA, False, pid)
    if not h:
        return None
    try:
        return bool(ctypes.windll.psapi.EmptyWorkingSet(h))
    finally:
        k32.CloseHandle(h)


# --------------------------------------------------------------------------
# железо
# --------------------------------------------------------------------------

def cpu_info() -> dict:
    out: dict = {"logical": os.cpu_count(),
                 "physical": None, "name": None, "source": None}
    if IS_WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                out["name"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
                out["source"] = "registry"
        except (OSError, ImportError):
            pass
    if not out["name"]:
        import platform
        out["name"] = platform.processor() or platform.machine()
        out["source"] = "platform"
    out["physical"] = _physical_cores()
    return out


def _physical_cores() -> int | None:
    """Число физических ядер.

    Через `GetLogicalProcessorInformationEx`, а не через powershell: powershell
    на каждый вызов - это отдельный процесс на 300-500 мс, а приложение
    определяет железо при каждом холодном старте. Заодно исчезает зависимость
    от того, доступен ли powershell в песочнице.
    """
    if IS_WINDOWS:
        try:
            k32 = ctypes.windll.kernel32
            length = wt.DWORD(0)
            # Первый вызов с NULL - только чтобы узнать длину буфера.
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
            if not length.value:
                return None
            buf = ctypes.create_string_buffer(length.value)
            if not k32.GetLogicalProcessorInformationEx(0, buf,
                                                        ctypes.byref(length)):
                return None
            cores, off = 0, 0
            while off + 8 <= length.value:
                rel = ctypes.c_uint32.from_buffer(buf, off).value
                size = ctypes.c_uint32.from_buffer(buf, off + 4).value
                if size <= 0:
                    break
                if rel == 0:  # RelationProcessorCore
                    cores += 1
                off += size
            if cores:
                return cores
        except (OSError, AttributeError, ValueError):
            pass
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property "
                 "NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=20)
            val = out.stdout.strip()
            return int(val) if val.isdigit() else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        return len({int(re.search(r"core id\s+:\s+(\d+)", l).group(1))
                    for l in open("/proc/cpuinfo", encoding="utf-8")
                    if re.search(r"core id\s+:", l)})
    except (OSError, AttributeError):
        return None


def gpu_info(explicit_vram_gib: float | None = None) -> dict:
    """Видеокарта и её память.

    Порядок важен: nvidia-smi часто отдаёт «Failed to initialize NVML» внутри
    песочницы и на части драйверов, поэтому есть откат на реестр Windows, а
    последним - ручной бюджет. Честный источник попадает в поле source, чтобы
    UI мог пометить цифру как измеренную или как заданную руками.
    """
    out: dict = {"name": None, "vram_total_mb": None, "vram_free_mb": None,
                 "source": None, "notes": []}
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            proc = subprocess.run(
                [exe, "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=25)
            line = (proc.stdout or "").strip().splitlines()
            if line and proc.returncode == 0:
                parts = [p.strip() for p in line[0].split(",")]
                if len(parts) >= 3:
                    out["name"] = parts[0]
                    out["vram_total_mb"] = int(float(parts[1]))
                    out["vram_free_mb"] = int(float(parts[2]))
                    out["source"] = "nvidia-smi"
            else:
                # Молчать здесь нельзя: «source: registry» без объяснения
                # выглядит как норма, хотя на деле NVML недоступен и
                # свободной VRAM мы не знаем.
                err = (proc.stderr or "").strip().splitlines()
                out["notes"].append(
                    "nvidia-smi вернул " + str(proc.returncode)
                    + (f": {err[0]}" if err else ""))
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            out["notes"].append(f"nvidia-smi не сработал: {e}")
    else:
        out["notes"].append("nvidia-smi не найден в PATH")

    if out["vram_total_mb"] is None:
        name, mb = _vram_from_registry()
        if mb:
            out["name"] = out["name"] or name
            out["vram_total_mb"] = mb
            out["source"] = "registry"

    if out["vram_total_mb"] is None and explicit_vram_gib:
        out["vram_total_mb"] = int(explicit_vram_gib * 1024)
        out["source"] = "задано вручную"
    if out["vram_total_mb"] is None:
        out["notes"].append("VRAM не определена, передай бюджет вручную")
    return out


def _vram_from_registry() -> tuple[str | None, int | None]:
    """HardwareInformation.qwMemorySize. В реестре лежит DWORD-обрезка на 4 ГБ,
    поэтому читаем сырые байты и, если пришло ровно 4 ГБ, пробуем QWORD."""
    if not IS_WINDOWS:
        return None, None
    try:
        import winreg
    except ImportError:
        return None, None
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    best_name, best_mb = None, None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as k:
                        try:
                            name = winreg.QueryValueEx(k, "DriverDesc")[0]
                        except OSError:
                            name = None
                        mb = None
                        try:
                            raw, _ = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")
                            if isinstance(raw, (bytes, bytearray)) and len(raw) >= 8:
                                mb = int.from_bytes(raw[:8], "little") // MIB
                            elif isinstance(raw, int):
                                mb = raw // MIB
                        except OSError:
                            pass
                        if not mb or mb < 512:
                            continue
                        if best_mb is None or mb > best_mb:
                            best_name, best_mb = name, mb
                except OSError:
                    continue
    except OSError:
        return None, None
    return best_name, best_mb


# --------------------------------------------------------------------------
# модель и лог сервера
# --------------------------------------------------------------------------

def model_file(path: str) -> dict:
    """Размер файла модели - знаменатель для host_ratio."""
    try:
        st = os.stat(path)
    except OSError:
        return {"path": path, "exists": False}
    return {"path": path, "exists": True, "size_mb": st.st_size // MIB,
            "size_bytes": st.st_size, "mtime": int(st.st_mtime)}


def host_ratio(model_size_mb: int | None, proc_ws_mb: int | None) -> float | None:
    """Отношение рабочего набора сервера к размеру файла модели.

    Смысл имеет для моделей от гигабайта: у мелких (0.8B, 516 МБ) в рабочий
    набор входит ещё и контекст CUDA, поэтому отношение уходит выше единицы
    (замерено 1.65) и говорит уже не о копии весов, а о накладных расходах
    рантайма. Читать так: ~1.0 и выше на большой модели - копия весов в
    системной памяти есть; ~0.15 - копия отпущена.
    """
    if not model_size_mb or proc_ws_mb is None:
        return None
    return round(proc_ws_mb / model_size_mb, 3)


def _parse_breakdown(txt: str) -> dict:
    """Разобрать таблицу `common_memory_breakdown_print` по устройствам.

    Формат колонок: `total = free + (self = model + context + compute) + unaccounted`.
    У хоста колонок total/free нет, поэтому число значений в строке разное:
    семь у видеокарты, четыре у Host. Различаем по количеству, а не по имени
    устройства: имя приходит из драйвера и может быть любым.
    """
    devices: dict = {}
    for name, vals in _BREAKDOWN_RE.findall(txt):
        nums = [int(x) for x in re.findall(r"-?\d+", vals)]
        dev = name.split("(")[0].strip()
        if len(nums) == 7:
            devices[dev] = {"total_mb": nums[0], "free_mb": nums[1],
                            "self_mb": nums[2], "model_mb": nums[3],
                            "context_mb": nums[4], "compute_mb": nums[5],
                            "unaccounted_mb": nums[6]}
        elif len(nums) == 4:
            devices[dev] = {"total_mb": None, "free_mb": None,
                            "self_mb": nums[0], "model_mb": nums[1],
                            "context_mb": nums[2], "compute_mb": nums[3],
                            "unaccounted_mb": None}
    return devices


def log_buffers(log_path: str) -> dict:
    """Разобрать лог llama-server: сколько именно памяти ушло куда.

    Это единственный доступный путь измерить VRAM, когда NVML заблокирован
    (в песочнице nvidia-smi отдаёт «Failed to initialize NVML»): движок сам
    печатает и размеры буферов, и проектный бюджет по устройствам.

    Два источника, оба нужны:
      * `buffer size = N MiB` - факт, что реально выделено, с разбивкой
        CUDA0 / CPU_Mapped / CUDA_Host;
      * `memory breakdown` - бюджет движка до загрузки: total, free и
        раскладка model/context/compute по каждому устройству.
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
    except OSError:
        return {}
    out: dict = {"log_bytes": len(txt)}

    m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", txt)
    if m:
        out["layers_gpu"], out["layers_total"] = int(m.group(1)), int(m.group(2))
        out["all_layers_on_gpu"] = out["layers_gpu"] == out["layers_total"]

    bufs: dict[str, float] = {}
    for dev, buf, val in _DEV_BUFFER_RE.findall(txt):
        key = f"{dev}/{buf}" if dev else buf
        bufs[key] = max(bufs.get(key, 0.0), float(val))
    if bufs:
        out["buffers_mib"] = bufs
        # Делим не по имени буфера, а по устройству: именно это отвечает на
        # вопрос «сколько ушло на карту, а сколько осталось в системной памяти».
        out["vram_mib"] = round(sum(v for k, v in bufs.items()
                                    if k.split("/")[0].startswith("CUDA")
                                    and "Host" not in k.split("/")[0]), 1)
        out["host_mib"] = round(sum(v for k, v in bufs.items()
                                    if not (k.split("/")[0].startswith("CUDA")
                                            and "Host" not in k.split("/")[0])), 1)

    devices = _parse_breakdown(txt)
    if devices:
        out["devices"] = devices
        gpu = next((v for k, v in devices.items()
                    if k.startswith("CUDA")), None)
        if gpu:
            out["vram_total_mb"] = gpu["total_mb"]
            out["vram_free_mb"] = gpu["free_mb"]
            out["projected_vram_mb"] = gpu["self_mb"]
    elif "model loaded" in txt and "buffers_mib" not in out:
        # Модель загрузилась, а строк про буферы нет - значит лог снят на
        # слишком низком уровне детализации. Без подсказки это выглядит как
        # «движок не сообщил», хотя причина известна и устранима.
        out["vram_hint"] = ("движок не печатал учёт памяти: нужен -lv 4, "
                            "на -lv 3 строк buffer size нет")

    for pat, key in ((r"CUDA0 KV buffer size\s*=\s*([\d.]+)", "kv_mib"),
                     (r"CUDA0 compute buffer size\s*=\s*([\d.]+)", "compute_mib"),
                     (r"CPU_Mapped model buffer size\s*=\s*([\d.]+)", "cpu_mapped_mib"),
                     (r"CUDA_Host model buffer size\s*=\s*([\d.]+)", "cuda_host_mib"),
                     (r"CUDA0 model buffer size\s*=\s*([\d.]+)", "model_mib"),
                     (r"graph splits\s*=\s*(\d+)", "graph_splits"),
                     (r"n_ctx_seq\s*=\s*(\d+)", "n_ctx")):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

    # Где именно оказались веса на стороне хоста, зависит от режима загрузки,
    # и разница принципиальная:
    #   mmap -> `CPU_Mapped model buffer size` = весь файл модели в системной
    #           памяти, отображённый с диска (5.67 ГБ на этой модели);
    #   none -> `CUDA_Host model buffer size` = только небольшой буфер,
    #           которым CUDA backend принимает тензоры (265 МБ). Полной копии
    #           весов в системной памяти нет - ровно ради этого режим и нужен.
    # Без этой развилки «хост 350 МБ» выглядит как странность, хотя это и есть
    # доказательство, что копия отпущена.
    if out.get("cuda_host_mib") and not out.get("cpu_mapped_mib"):
        out["host_weights_mib"] = out["cuda_host_mib"]
        out["host_weights_kind"] = "CUDA_Host (буфер приёма, не копия весов)"

    if "vram_free_mb" not in out:
        m = _DEV_FREE_RE.search(txt)
        if m:
            out["vram_free_mb"] = int(m.group(2))
            out.setdefault("vram_total_mb", None)

    m = re.search(r"load_mode\s*=\s*(\w+)", txt) or re.search(r"load-mode\s+(\w+)", txt)
    if m:
        out["load_mode"] = m.group(1)
        # Связка load_mode и host_ratio - то, ради чего всё это считается.
        # mmap: движок держит в системной памяти копию весов целиком.
        out["host_copy_expected"] = m.group(1) in ("mmap", "auto", "mlock")
    out["cpu_assigned_layers"] = len(re.findall(r"assigned to device CPU", txt))
    if out["cpu_assigned_layers"]:
        out["note"] = "часть слоёв ушла на CPU - жди graph splits и потерю скорости"
    if out.get("graph_splits", 0) > 2:
        out["note"] = (f"graph splits = {out['graph_splits']}: граф рвётся "
                       "между CPU и GPU, каждый токен платит за пересылку")

    hard, soft = [], []
    for line in txt.splitlines():
        if not re.search(r"failed|out of memory|cannot allocate|falling back|"
                         r"error|abort", line, re.I):
            continue
        (soft if _BENIGN_RE.search(line) else hard).append(line)
    if hard:
        out["errors"] = hard[-5:]
    if soft:
        out["warnings"] = soft[-5:]
    return out


@dataclass
class Snapshot:
    """Один снимок состояния машины."""
    t: float = 0.0
    avail_mb: int = 0
    commit_avail_mb: int | None = None
    proc_ws_mb: int | None = None
    proc_commit_mb: int | None = None
    vram_free_mb: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def snapshot(pid: int | None = None) -> Snapshot:
    sm = system_memory()
    sn = Snapshot(avail_mb=sm.get("avail_mb", 0),
                  commit_avail_mb=sm.get("commit_avail_mb"))
    if pid:
        pm = process_memory(pid) or {}
        sn.proc_ws_mb = pm.get("ws_mb")
        sn.proc_commit_mb = pm.get("commit_mb")
    return sn


# --------------------------------------------------------------------------
# проверка до старта и поиск виновников
# --------------------------------------------------------------------------

def _process_name(pid: int) -> str | None:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wt.DWORD(len(buf))
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
    except (OSError, AttributeError):
        pass
    finally:
        k32.CloseHandle(h)
    return None


def top_consumers(n: int = 6, min_mb: int = 200) -> list[dict]:
    """Кто держит память. Нужно, когда прогон отказывается стартовать.

    Отказ без объяснения бесполезен: «свободно 800 МБ» не подсказывает, что
    делать. Список процессов с рабочими наборами отвечает на вопрос «что
    закрыть». Через EnumProcesses, а не через tasklist: вывод tasklist
    приходит в кодировке консоли и разбирается ненадёжно, а внешний процесс
    на каждый вызов - лишняя нагрузка на и без того загруженную машину.
    """
    if not IS_WINDOWS:
        return []
    psapi = ctypes.windll.psapi
    cap = 2048
    arr = (ctypes.c_uint32 * cap)()
    needed = ctypes.c_uint32()
    if not psapi.EnumProcesses(arr, ctypes.sizeof(arr), ctypes.byref(needed)):
        return []
    out: list[dict] = []
    for i in range(min(needed.value // 4, cap)):
        pid = int(arr[i])
        if not pid:
            continue
        pm = process_memory(pid)
        if not pm or pm["ws_mb"] < min_mb:
            continue
        out.append({"pid": pid, "ws_mb": pm["ws_mb"],
                    "name": _process_name(pid) or "?"})
    out.sort(key=lambda r: -r["ws_mb"])
    return out[:n]


def preflight(floor_mb: int = 1500, need_mb: int = 0) -> dict:
    """Можно ли вообще начинать прогон.

    Сторож ловит перерасход во время работы, но у него нет ответа на случай,
    когда памяти нет УЖЕ ДО старта: тогда он срабатывает на первом же замере,
    и это выглядит как поломка приложения, хотя на машине просто занято.
    Проверка до старта даёт внятный отказ и список виновников.

    Проверено на этой машине: при живом чужом llama-server на 5875 МБ
    свободной памяти оставалось 800-1700 МБ, и любая попытка что-то запустить
    упиралась в порог. Правильное поведение - не пытаться.
    """
    sm = system_memory()
    avail = sm.get("avail_mb") or 0
    total = sm.get("total_mb") or 0
    out: dict = {"ok": True, "avail_mb": avail, "total_mb": total,
                 "floor_mb": floor_mb, "need_mb": need_mb,
                 "commit_avail_mb": sm.get("commit_avail_mb"),
                 "top": top_consumers()}
    if avail < floor_mb:
        out["ok"] = False
        out["reason"] = (f"свободно {avail} МБ, а порог безопасности "
                         f"{floor_mb} МБ: прогон не начинаю, иначе машина "
                         f"уйдёт в своп")
    elif need_mb and avail - need_mb < floor_mb:
        out["ok"] = False
        out["reason"] = (f"свободно {avail} МБ, модели нужно около {need_mb} МБ, "
                         f"после загрузки останется меньше порога {floor_mb} МБ")
    if not out["ok"] and out["top"]:
        who = ", ".join(f"{r['name']} ({r['ws_mb']} МБ)" for r in out["top"][:4])
        out["holders"] = who
        out["reason"] += f". Держат: {who}"
    return out
