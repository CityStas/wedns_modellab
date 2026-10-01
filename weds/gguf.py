"""Парсер метаданных GGUF.

Читает заголовок GGUF и отдаёт словарь метаданных. Нужен, чтобы понимать
архитектуру модели (число слоёв, голов, длину контекста) без запуска сервера.

ВАЖНО: расчёт размера KV-кеша по метаданным для гибридных архитектур
(attention + SSM, например qwen35 / Qwen3-Next) даёт ошибку в разы.
Реальные цифры брать из логов движка (см. lmstudio.read_estimates),
а `estimate_kv_bytes` использовать только как грубую прикидку.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

_MAGIC = b"GGUF"

# type_id -> (struct format char, size in bytes)
_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}

# Ключи, которые нас интересуют, в порядке вывода.
INTERESTING = [
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.parameter_count",
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.sliding_window",
    "full_attention_interval",
    "rope.freq_base",
    "rope.dimension_count",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.inner_size",
    "ssm.conv_kernel",
    "ssm.time_step_rank",
    "vocab_size",
]


@dataclass
class GgufInfo:
    path: Path
    version: int = 0
    tensor_count: int = 0
    metadata: dict = field(default_factory=dict)
    size_bytes: int = 0

    @property
    def arch(self) -> str:
        return self.metadata.get("general.architecture", "?")

    @property
    def name(self) -> str:
        return self.metadata.get("general.name", self.path.stem)

    def get(self, key: str, default=None):
        """Достаёт ключ с подстановкой архитектуры: get('block_count')."""
        return self.metadata.get(f"{self.arch}.{key}", default)

    @property
    def is_hybrid(self) -> bool:
        """Гибрид attention+SSM: есть SSM-ветка или прореженный full attention."""
        return (
            self.get("ssm.state_size") is not None
            or (self.get("full_attention_interval") or 1) > 1
        )

    @property
    def full_attention_layers(self) -> int:
        """Сколько слоёв держат настоящий KV-кеш."""
        n = self.get("block_count") or 0
        interval = self.get("full_attention_interval") or 1
        return max(1, n // interval) if interval > 1 else n

    def estimate_kv_bytes(self, context: int, bytes_per_element: int = 2) -> int:
        """Грубая прикидка KV-кеша. Для гибридных архитектур врёт в разы —
        брать из логов движка."""
        layers = self.full_attention_layers
        kv_heads = self.get("attention.head_count_kv") or self.get("attention.head_count") or 0
        k_len = self.get("attention.key_length") or (
            (self.get("embedding_length") or 0) // max(1, self.get("attention.head_count") or 1)
        )
        v_len = self.get("attention.value_length") or k_len
        per_token = layers * kv_heads * (k_len + v_len) * bytes_per_element
        return per_token * context

    def describe(self) -> dict:
        out = {"path": str(self.path), "size_bytes": self.size_bytes, "arch": self.arch}
        for key in INTERESTING:
            full = f"{self.arch}.{key}"
            if full in self.metadata:
                out[key] = self.metadata[full]
        out["_hybrid"] = self.is_hybrid
        out["_full_attention_layers"] = self.full_attention_layers
        return out


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, type_id: int):
    if type_id == 8:  # string
        return _read_string(f)
    if type_id == 9:  # array
        (elem_type,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if elem_type == 8:
            return [_read_string(f) for _ in range(n)]
        fmt, size = _TYPES[elem_type]
        return list(struct.unpack(f"<{n}{fmt}", f.read(n * size)))
    fmt, size = _TYPES[type_id]
    return struct.unpack("<" + fmt, f.read(size))[0]


def parse(path: str | Path, max_pairs: int | None = None) -> GgufInfo:
    """Читает заголовок GGUF. Бросает ValueError, если это не GGUF."""
    path = Path(path)
    info = GgufInfo(path=path)
    try:
        info.size_bytes = path.stat().st_size
    except OSError:
        pass

    with open(path, "rb") as f:
        if f.read(4) != _MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        (info.version,) = struct.unpack("<I", f.read(4))
        (info.tensor_count,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        limit = n_kv if max_pairs is None else min(n_kv, max_pairs)
        for _ in range(limit):
            key = _read_string(f)
            (type_id,) = struct.unpack("<I", f.read(4))
            info.metadata[key] = _read_value(f, type_id)

    return info


def find_ggufs(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() == ".gguf" else []
    return sorted(p for p in root.rglob("*.gguf") if p.is_file())
