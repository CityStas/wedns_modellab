"""Бенчмарки: генерация, префилл, кэш префикса.

Отличие от checks.py: здесь не «годно/негодно», а чистые числа.
Гонять по одному прогону за раз — серия тяжёлых тестов подряд сама по себе
вешает систему на слабой видеокарте.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checks import build_prompt, calibrate_chars_per_token, new_nonce
from .servers import LocalServer


@dataclass
class BenchRow:
    name: str
    tokens: int
    seconds: float
    tps: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "tokens": self.tokens,
                "seconds": round(self.seconds, 2), "tps": round(self.tps, 1), **self.extra}


def bench_generation(srv: LocalServer, model: str, tokens: int = 128, runs: int = 2) -> BenchRow:
    """Скорость генерации. Греется одним коротким прогоном, затем усредняется."""
    srv.chat(model, [{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.0)

    best = None
    for _ in range(max(1, runs)):
        res = srv.chat(
            model,
            [{"role": "user", "content": "Count from 1 to 500, one number per line."}],
            max_tokens=tokens,
            temperature=0.0,
        )
        if not res.ok:
            return BenchRow("generation", 0, res.elapsed, 0.0, {"error": res.error})
        if best is None or res.gen_tps > best.gen_tps:
            best = res

    return BenchRow(
        "generation",
        best.completion_tokens,
        best.elapsed,
        best.gen_tps,
        {"reasoning_tokens": best.reasoning_tokens, "runs": runs},
    )


def bench_prefill(srv: LocalServer, model: str, target_tokens: int = 8000) -> BenchRow:
    cpt = calibrate_chars_per_token(srv, model)
    prompt = build_prompt(cpt, target_tokens, nonce=new_nonce())
    res = srv.chat(
        model,
        [{"role": "user", "content": prompt + "\nReply with one word: done"}],
        max_tokens=1,
        temperature=0.0,
    )
    if not res.ok:
        return BenchRow("prefill", 0, res.elapsed, 0.0,
                        {"error": res.error, "error_type": res.error_type})
    return BenchRow("prefill", res.prompt_tokens, res.elapsed, res.prefill_tps)


def bench_prefix_cache(srv: LocalServer, model: str, target_tokens: int = 6000, turns: int = 3) -> list[BenchRow]:
    """Три хода с общим префиксом и разными хвостами.

    Первый — холодный, остальные должны быть в разы быстрее, если кэш работает.
    """
    cpt = calibrate_chars_per_token(srv, model)
    prefix = build_prompt(cpt, target_tokens, nonce=new_nonce())
    rows: list[BenchRow] = []
    for i in range(max(2, turns)):
        res = srv.chat(
            model,
            [{"role": "user", "content": f"{prefix}\n\nReply with one word: turn{i}"}],
            max_tokens=4,
            temperature=0.0,
        )
        if not res.ok:
            rows.append(BenchRow(f"turn{i + 1}", 0, res.elapsed, 0.0, {"error": res.error}))
            break
        rows.append(
            BenchRow(
                f"turn{i + 1}",
                res.prompt_tokens,
                res.elapsed,
                res.prefill_tps,
                {"cold" if i == 0 else "warm": True},
            )
        )
    return rows


def run(srv: LocalServer, model: str, *, gen_tokens: int = 128, prefill_tokens: int = 8000,
        prefix_tokens: int = 6000, skip: tuple[str, ...] = ()) -> dict:
    out: dict = {"model": model, "server": srv.base_url, "kind": srv.kind}
    if "gen" not in skip:
        out["generation"] = bench_generation(srv, model, gen_tokens).to_dict()
    if "prefill" not in skip:
        out["prefill"] = bench_prefill(srv, model, prefill_tokens).to_dict()
    if "prefix" not in skip:
        rows = bench_prefix_cache(srv, model, prefix_tokens)
        out["prefix_cache"] = [r.to_dict() for r in rows]
        if len(rows) >= 2 and rows[1].seconds > 0:
            out["prefix_cache_speedup"] = round(rows[0].seconds / rows[1].seconds, 1)
    return out
