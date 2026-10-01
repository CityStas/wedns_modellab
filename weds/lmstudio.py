"""Специфика LM Studio / Bionic: home, CLI, per-model конфиги, логи.

Проверено на Bionic 1.0.9 (LM Studio 0.4.x). Пути для настоящего LM Studio
могут отличаться в части имени каталога приложения — поэтому app-каталог
ищется, а не хардкодится.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_ROOT_NAME = "user-concrete-model-default-config"
BACKUP_SUFFIX = ".weds-backup"

# Ключи load-конфига, которые проверены на живом движке.
LOAD_KEYS = {
    "context": "llm.load.contextLength",
    "parallel": "llm.load.numParallelSessions",
    "offload": "llm.load.llama.acceleration.offloadRatio",
    "cpu_threads": "llm.load.llama.cpuThreadPoolSize",
    "flash_attention": "llm.load.llama.flashAttention",
    "eval_batch": "llm.load.llama.evalBatchSize",
    "phys_batch": "llm.load.llama.physicalBatchSize",
    "context_checkpoints": "llm.load.llama.contextCheckpoints",
    "k_cache_quant": "llm.load.llama.kCacheQuantizationType",
    "v_cache_quant": "llm.load.llama.vCacheQuantizationType",
    "auto_fit": "llm.load.llama.autoFit",
    # Держать KV-кеш в видеопамяти, а не в системной RAM. Заметно ускоряет
    # генерацию, но занимает VRAM пропорционально контексту.
    "offload_kv": "llm.load.offloadKVCacheToGpu",
}

OP_KEYS = {
    "temperature": "llm.prediction.temperature",
    "top_p": "llm.prediction.topPSampling",
    "top_k": "llm.prediction.topKSampling",
    "min_p": "llm.prediction.minPSampling",
    "repeat_penalty": "llm.prediction.repeatPenalty",
    "reasoning_budget": "llm.prediction.reasoning.budgetTokens",
    "enable_thinking": "llm.prediction.reasoning.enableThinking",
}


# --------------------------------------------------------------------------- #
# Поиск home и CLI
# --------------------------------------------------------------------------- #

def find_home() -> Path | None:
    """Каталог LM Studio. Учитывает перенос через ~/.lmstudio-home-pointer."""
    import os

    env = os.environ.get("LMSTUDIO_HOME")
    if env and Path(env).is_dir():
        return Path(env)

    pointer = Path.home() / ".lmstudio-home-pointer"
    if pointer.is_file():
        try:
            target = Path(pointer.read_text(encoding="utf-8").strip())
            if target.is_dir():
                return target
        except OSError:
            pass

    default = Path.home() / ".lmstudio"
    return default if default.is_dir() else None


def find_app_dir(home: Path) -> Path | None:
    """Каталог приложения внутри home (bionic / lm-studio / ...)."""
    apps = home / "apps"
    if apps.is_dir():
        candidates = sorted(
            (d for d in apps.iterdir() if d.is_dir() and (d / "settings.json").is_file()),
            key=lambda d: (d / "settings.json").stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    if (home / "settings.json").is_file():
        return home
    return None


def find_lms(home: Path) -> Path | None:
    for name in ("lms.exe", "lms"):
        candidate = home / "bin" / name
        if candidate.is_file():
            return candidate
    return None


def find_models_root(home: Path) -> Path | None:
    """Каталог с моделями (downloadsFolder из settings.json)."""
    app = find_app_dir(home)
    if app:
        try:
            settings = json.loads((app / "settings.json").read_text(encoding="utf-8"))
            folder = settings.get("downloadsFolder")
            if folder and Path(folder).is_dir():
                return Path(folder)
        except (OSError, json.JSONDecodeError):
            pass
    models = home / "models"
    return models if models.is_dir() else None


def find_config_root(home: Path) -> Path | None:
    app = find_app_dir(home)
    if app:
        root = app / ".internal" / CONFIG_ROOT_NAME
        if root.is_dir():
            return root
    root = home / ".internal" / CONFIG_ROOT_NAME
    return root if root.is_dir() else None


def find_log_dir(home: Path) -> Path | None:
    app = find_app_dir(home)
    for base in (app / "server-logs" if app else None, home / "server-logs"):
        if base and base.is_dir():
            return base
    return None


# --------------------------------------------------------------------------- #
# CLI-обёртка
# --------------------------------------------------------------------------- #

def run_lms(home: Path, args: list[str], timeout: float = 600.0) -> tuple[int, str]:
    lms = find_lms(home)
    if not lms:
        return 127, "lms CLI not found"
    try:
        proc = subprocess.run(
            [str(lms), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"lms {' '.join(args)} timed out after {timeout}s"
    except OSError as e:
        return 1, str(e)


def parse_ps(output: str) -> list[dict]:
    """Разбирает вывод `lms ps`."""
    rows = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].upper() == "IDENTIFIER":
            continue
        if parts[0].lower().startswith(("to load", "lms ")):
            continue
        rows.append(
            {
                "identifier": parts[0],
                "model": parts[1],
                "status": parts[2],
                "size": parts[3],
                "context": parts[4] if len(parts) > 4 else None,
                "parallel": parts[5] if len(parts) > 5 else None,
            }
        )
    return rows


def load_model(home: Path, model_key: str, *, context: int | None = None,
               gpu: str | None = None, parallel: int | None = None,
               ttl: int | None = None, timeout: float = 900.0) -> tuple[int, str]:
    args = ["load", model_key, "-y"]
    if context:
        args += ["-c", str(context)]
    if gpu:
        args += ["--gpu", gpu]
    if parallel:
        args += ["--parallel", str(parallel)]
    if ttl:
        args += ["--ttl", str(ttl)]
    return run_lms(home, args, timeout=timeout)


def unload_all(home: Path, timeout: float = 120.0) -> tuple[int, str]:
    return run_lms(home, ["unload", "--all"], timeout=timeout)


# --------------------------------------------------------------------------- #
# Per-model конфиг
# --------------------------------------------------------------------------- #

def config_path_for(home: Path, gguf_path: Path) -> Path | None:
    """Путь конфига для GGUF: зеркало структуры downloadsFolder."""
    root = find_config_root(home)
    models_root = find_models_root(home)
    if not root or not models_root:
        return None
    try:
        rel = Path(gguf_path).resolve().relative_to(Path(models_root).resolve())
    except ValueError:
        return None
    return root / rel.parent / (rel.name + ".json")


def find_config_by_key(home: Path, model_key: str) -> Path | None:
    """Ищет конфиг по ключу модели (имени gguf без расширения)."""
    root = find_config_root(home)
    if not root:
        return None
    needle = model_key.lower()
    for cfg in root.rglob("*.gguf.json"):
        stem = cfg.name[: -len(".gguf.json")].lower()
        if stem == needle:
            return cfg
    # Мягкое совпадение: ключ LM Studio иногда отбрасывает суффикс кванта.
    for cfg in root.rglob("*.gguf.json"):
        stem = cfg.name[: -len(".gguf.json")].lower()
        if stem.startswith(needle) or needle.startswith(stem):
            return cfg
    return None


@dataclass
class ModelConfig:
    path: Path
    data: dict = field(default_factory=dict)

    @property
    def load_fields(self) -> list[dict]:
        return (self.data.get("load") or {}).get("fields") or []

    @property
    def operation_fields(self) -> list[dict]:
        return (self.data.get("operation") or {}).get("fields") or []

    def get_load(self, name: str):
        key = LOAD_KEYS.get(name, name)
        for f in self.load_fields:
            if f.get("key") == key:
                return f.get("value")
        return None

    def get_load_simple(self, name: str):
        """Значение без обёртки {checked, value}."""
        v = self.get_load(name)
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        return v

    def get_op(self, name: str):
        key = OP_KEYS.get(name, name)
        for f in self.operation_fields:
            if f.get("key") == key:
                return f.get("value")
        return None

    def as_dict(self) -> dict:
        out = {}
        for name in LOAD_KEYS:
            out[name] = self.get_load_simple(name)
        return out


def read_config(path: Path) -> ModelConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelConfig(path=Path(path), data=data)


def _set_field(fields: list[dict], key: str, value, wrapped: bool) -> None:
    for f in fields:
        if f.get("key") == key:
            if wrapped and isinstance(f.get("value"), dict):
                f["value"]["value"] = value
            elif wrapped:
                f["value"] = {"checked": True, "value": value}
            else:
                f["value"] = value
            return
    fields.append({"key": key, "value": {"checked": True, "value": value} if wrapped else value})


# Поля, у которых значение обёрнуто в {checked, value}
WRAPPED_LOAD = {"k_cache_quant", "v_cache_quant"}


def write_config(path: Path, *, load: dict | None = None, operation: dict | None = None,
                 backup: bool = True) -> Path | None:
    """Пишет конфиг, сохраняя бэкап. Возвращает путь бэкапа, если он создан."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"preset": "", "operation": {"fields": []}, "load": {"fields": []}}
    data.setdefault("operation", {}).setdefault("fields", [])
    data.setdefault("load", {}).setdefault("fields", [])

    backup_path = None
    if backup and path.is_file():
        candidate = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        if candidate.exists():
            # Бэкап уже есть — значит это состояние ДО первой правки. Не трогаем:
            # иначе повторный --apply затрёт оригинал промежуточным состоянием,
            # и restore откатит не туда, куда пользователь рассчитывает.
            backup_path = candidate
        else:
            shutil.copy2(path, candidate)
            backup_path = candidate

    for name, value in (load or {}).items():
        key = LOAD_KEYS.get(name, name)
        _set_field(data["load"]["fields"], key, value, wrapped=name in WRAPPED_LOAD)

    for name, value in (operation or {}).items():
        key = OP_KEYS.get(name, name)
        _set_field(data["operation"]["fields"], key, value, wrapped=name in ("reasoning_budget",))

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return backup_path


# --------------------------------------------------------------------------- #
# Логи и оценки движка
# --------------------------------------------------------------------------- #

_CTX_ERR = re.compile(
    r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)"
)
_LOAD_LINE = re.compile(r"load_model: initializing, n_slots = (\d+), n_ctx_slot = (\d+)")
# MULTILINE обязателен: без него '$' матчится только в конце всего текста,
# а не в конце каждой строки, и оценки памяти не находятся.
_ESTIMATE = re.compile(r"Estimated model usage for ([^:]+): (\{.*\})\s*$", re.MULTILINE)


def latest_log(home: Path) -> Path | None:
    log_dir = find_log_dir(home)
    if not log_dir:
        return None
    files = sorted(log_dir.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_log(home: Path, max_bytes: int = 4_000_000) -> str:
    path = latest_log(home)
    if not path:
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", "replace")


@dataclass
class LogFindings:
    context_errors: list[dict] = field(default_factory=list)
    loads: list[dict] = field(default_factory=list)
    estimates: list[dict] = field(default_factory=list)
    log_path: str | None = None

    @property
    def latest_estimate(self) -> dict | None:
        return self.estimates[-1] if self.estimates else None


def analyze_log(home: Path) -> LogFindings:
    """Достаёт из лога движка самое полезное: ошибки контекста, реальные
    параметры загрузки и оценки памяти."""
    findings = LogFindings()
    path = latest_log(home)
    findings.log_path = str(path) if path else None
    text = read_log(home)

    for m in _CTX_ERR.finditer(text):
        findings.context_errors.append(
            {"prompt_tokens": int(m.group(1)), "n_ctx": int(m.group(2))}
        )

    for m in _LOAD_LINE.finditer(text):
        findings.loads.append({"n_slots": int(m.group(1)), "n_ctx": int(m.group(2))})

    for m in _ESTIMATE.finditer(text):
        try:
            payload = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        payload["model"] = m.group(1).strip()
        findings.estimates.append(payload)

    return findings


def memory_model(findings: LogFindings) -> dict | None:
    """Из последней оценки выводит вес модели и цену контекста в байтах на токен.

    Это единственный надёжный источник цифр: ручной расчёт по метаданным GGUF
    для гибридных архитектур врёт в разы.
    """
    best = None
    for est in findings.estimates:
        ctx_bytes = est.get("estimatedContextUsageBytes") or 0
        if ctx_bytes > 0 and est.get("estimationIsAccurate"):
            best = est
    if not best:
        return None

    weights = best.get("estimatedModelVramUsageBytes") or best.get("estimatedModelUsageBytes")
    ctx_bytes = best.get("estimatedContextUsageBytes")
    return {
        "model": best.get("model"),
        "weights_bytes": weights,
        "context_bytes": ctx_bytes,
        "total_bytes": best.get("estimatedVramUsageBytes") or best.get("estimatedUsageBytes"),
        "note": "context_bytes — это полный размер KV для контекста того запроса; "
                "цену на токен считай от contextLength, который был в конфиге",
    }
