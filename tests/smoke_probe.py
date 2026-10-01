"""Смоук modellab.probe: датчики должны отвечать на этой машине, а не падать.

Запуск: python tests/smoke_probe.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modellab import probe  # noqa: E402


def show(title, obj):
    print(f"== {title} ==")
    print(json.dumps(obj, ensure_ascii=False, default=str))


show("system_memory", probe.system_memory())
show("cpu_info", probe.cpu_info())
show("gpu_info(auto)", probe.gpu_info())
show("gpu_info(manual 8 GiB)", probe.gpu_info(8.0))
show("process_memory(self)", probe.process_memory(os.getpid()))
show("snapshot(self)", probe.snapshot(os.getpid()).to_dict())
print("== trim_working_set(self) ==")
print(probe.trim_working_set(os.getpid()))
show("process_memory(bad pid)", probe.process_memory(999999))

print("== model_file ==")
found = 0
for pat in (r"E:\Programs\LocalLLM\Models\*.gguf",
            r"E:\Programs\LocalLLM\bonsai\**\*.gguf",
            r"E:\Programs\LocalLLM\**\*.gguf"):
    for p in glob.glob(pat, recursive=True):
        show(os.path.basename(p), probe.model_file(p))
        found += 1
        if found >= 8:
            break
    if found >= 8:
        break
if not found:
    show("model_file(нет файла)", probe.model_file(r"E:\nope.gguf"))

show("host_ratio(5670, 6072)", probe.host_ratio(5670, 6072))
show("host_ratio(5670, 881)", probe.host_ratio(5670, 881))
show("host_ratio(None, 100)", probe.host_ratio(None, 100))

print("== log_buffers ==")
logs = sorted(glob.glob(r"E:\Programs\LocalLLM\bonsai\logs\*.log"),
              key=os.path.getmtime, reverse=True)
if not logs:
    show("log_buffers(нет файла)", probe.log_buffers(r"E:\nope.log"))
for lg in logs[:8]:
    show(os.path.basename(lg), probe.log_buffers(lg))

# --- проверки, а не просто печать ----------------------------------------
fails = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("== assertions ==")
sm = probe.system_memory()
check("system_memory: есть avail", bool(sm.get("avail_mb")), f"{sm.get('avail_mb')} MB")
check("system_memory: есть commit", sm.get("commit_avail_mb") is not None)
cpu = probe.cpu_info()
check("cpu: имя непустое", bool(cpu.get("name")), str(cpu.get("name")))
check("cpu: логические >= физических", (cpu.get("logical") or 0) >= (cpu.get("physical") or 0))
gpu = probe.gpu_info()
check("gpu: total определён", bool(gpu.get("vram_total_mb")), f"{gpu.get('vram_total_mb')} MB")
check("gpu: причина отсутствия free объяснена",
      gpu.get("vram_free_mb") is not None or bool(gpu.get("notes")),
      str(gpu.get("notes")))
check("gpu: ручной бюджет подхватывается",
      probe.gpu_info(6.0)["vram_total_mb"] is not None)
check("process_memory: свой pid читается", (probe.process_memory(os.getpid()) or {}).get("ws_mb") is not None)
check("process_memory: чужой pid -> None", probe.process_memory(999999) is None)
check("trim_working_set: свой процесс", probe.trim_working_set(os.getpid()) is True)

mf = probe.model_file(r"E:\Programs\LocalLLM\Models\ternary\Ternary-Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-PTQ1_0.gguf")
check("model_file: файл есть", mf.get("exists") is True)
check("model_file: размер ~5.6 ГБ", 5000 < (mf.get("size_mb") or 0) < 6500, f"{mf.get('size_mb')} MB")
check("model_file: нет файла -> exists False", probe.model_file(r"E:\nope.gguf")["exists"] is False)
check("host_ratio: полная копия ~1.0", (probe.host_ratio(5670, 6072) or 0) > 0.9)
check("host_ratio: отпущено < 0.2", (probe.host_ratio(5670, 881) or 9) < 0.2)
check("host_ratio: без модели -> None", probe.host_ratio(None, 100) is None)

lg64 = r"E:\Programs\LocalLLM\bonsai\logs\server-64k.log"
if os.path.exists(lg64):
    b = probe.log_buffers(lg64)
    check("log_buffers: CUDA0 отделён от хоста", (b.get("vram_mib") or 0) > 5000,
          f"vram={b.get('vram_mib')} host={b.get('host_mib')}")
    check("log_buffers: 65/65 на GPU", b.get("all_layers_on_gpu") is True)
    check("log_buffers: бюджет устройств разобран", "CUDA0" in (b.get("devices") or {}))
    check("log_buffers: free VRAM из лога", bool(b.get("vram_free_mb")),
          f"{b.get('vram_free_mb')} MB")
    check("log_buffers: n_ctx", b.get("n_ctx") == 65536, str(b.get("n_ctx")))
    check("log_buffers: load_mode", b.get("load_mode") == "mmap", str(b.get("load_mode")))
    check("log_buffers: host_copy_expected согласован с load_mode",
          b.get("host_copy_expected") is True)
    check("log_buffers: безобидный fit-варнинг не попал в errors",
          "errors" not in b and bool(b.get("warnings")))
check("log_buffers: нет файла -> {}", probe.log_buffers(r"E:\nope.log") == {})

print()
print(f"ИТОГ: {'всё зелёное' if not fails else 'провалено: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
