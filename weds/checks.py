"""Набор проверок готовности локальной модели к агентской работе.

Каждая проверка отвечает на конкретный вопрос и, если что-то не так, отдаёт
готовый фикс. Именно этот набор ловит то, что ломается на практике:
не влезает системный промпт агента, модель не умеет tool-call, генерация
обвалилась из-за переполнения VRAM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .servers import ChatResult, LocalServer, ModelInfo

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

# Размер системного промпта агента с описаниями инструментов.
# 38000 — замерено на WorkBuddy: запрос падал с "37955 tokens exceeds 32768".
AGENT_PROMPT_DEFAULTS = {
    "workbuddy": 38000,
    "claude-code": 24000,
    "codex": 16000,
    "cursor": 20000,
    "generic": 16000,
}

TOOL_PROBE = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute directory path"}},
                "required": ["path"],
            },
        },
    }
]

TOOL_PROMPT = "How many files are in C:\\Windows? Use the list_dir tool."


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    detail: dict = field(default_factory=dict)
    fix: str | None = None

    def to_dict(self) -> dict:
        out = {"check": self.name, "status": self.status, "summary": self.summary}
        if self.detail:
            out["detail"] = self.detail
        if self.fix:
            out["fix"] = self.fix
        return out


@dataclass
class Thresholds:
    min_gen_tps: float = 12.0
    warn_gen_tps: float = 25.0
    min_prefill_tps: float = 200.0
    warn_prefill_tps: float = 500.0
    min_prefix_speedup: float = 3.0
    max_reasoning_tokens: int = 1500
    agent_prompt_tokens: int = 38000
    context_headroom: float = 1.10  # запас 10% над промптом агента


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

FILLER = (
    "You are a coding agent. Available tools: read_file, write_file, run_command, "
    "search_files. Always prefer reading a file before editing it. "
)


def calibrate_chars_per_token(srv: LocalServer, model: str) -> float:
    """Один короткий запрос, чтобы узнать реальное соотношение символов и токенов.
    Без калибровки промпт нужного размера не построить."""
    probe = FILLER * 40
    res = srv.chat(model, [{"role": "user", "content": probe}], max_tokens=1)
    if res.ok and res.prompt_tokens > 0:
        return max(1.0, len(probe) / res.prompt_tokens)
    return 4.0


def build_prompt(chars_per_token: float, target_tokens: int, nonce: str | None = None) -> str:
    """Строит промпт примерно нужного размера.

    nonce ставится В НАЧАЛО: кэш префикса в llama.cpp матчится от первых
    токенов, поэтому уникальный префикс гарантирует холодный префилл.
    Без nonce разные проверки попадают в кэш друг друга и замеры врут.
    """
    target_chars = int(target_tokens * chars_per_token)
    reps = max(1, target_chars // len(FILLER))
    body = FILLER * reps
    if nonce:
        return f"[probe {nonce}] {body}"
    return body


def new_nonce() -> str:
    import secrets
    return secrets.token_hex(8)


def pick_model(models: list[ModelInfo], wanted: str | None) -> ModelInfo | None:
    if not models:
        return None
    if wanted:
        for m in models:
            if m.id == wanted:
                return m
        needle = wanted.lower()
        for m in models:
            if needle in m.id.lower():
                return m
        return None
    loaded = [m for m in models if m.loaded]
    return (loaded or models)[0]


# --------------------------------------------------------------------------- #
# Проверки
# --------------------------------------------------------------------------- #

def check_server(srv: LocalServer) -> CheckResult:
    ok = srv.probe()
    return CheckResult(
        name="server",
        status=PASS if ok else FAIL,
        summary=f"{srv.kind} на {srv.base_url} " + ("отвечает" if ok else "недоступен"),
        detail={"kind": srv.kind, "base_url": srv.base_url},
        fix=None if ok else (
            "Сервер не слушает порт. Запусти LM Studio / Ollama / llama.cpp и включи "
            "локальный HTTP-сервер в настройках."
        ),
    )


def check_model_present(srv: LocalServer, model: str) -> CheckResult:
    models = srv.list_models()
    found = pick_model(models, model)
    if not found:
        return CheckResult(
            name="model",
            status=FAIL,
            summary=f"модель '{model}' не найдена на сервере",
            detail={"available": [m.id for m in models]},
            fix="Проверь id модели: он должен совпадать с тем, что отдаёт сервер, "
                "а не с именем файла на диске.",
        )
    return CheckResult(
        name="model",
        status=PASS,
        summary=f"{found.id} ({found.state}"
        + (f", ctx {found.context}" if found.context else "")
        + ")",
        detail=found.describe(),
    )


def check_context_fit(
    srv: LocalServer, model: str, th: Thresholds, chars_per_token: float
) -> CheckResult:
    """Главная проверка: влезает ли системный промпт агента в контекст.

    Именно она ловит ошибку 'request (37955 tokens) exceeds the available
    context size (32768 tokens)', которая в UI выглядит как 'model error'.
    """
    target = th.agent_prompt_tokens
    prompt = build_prompt(chars_per_token, target, nonce=new_nonce())
    res = srv.chat(model, [{"role": "user", "content": prompt}], max_tokens=1)

    if res.is_context_overflow:
        m = res.raw.get("error", {}) if isinstance(res.raw, dict) else {}
        need = m.get("n_prompt_tokens") or target
        have = m.get("n_ctx") or (srv.context_length(model) or 0)
        recommended = int(need * th.context_headroom)
        return CheckResult(
            name="context_fit",
            status=FAIL,
            summary=f"промпт агента ({need} токенов) НЕ влезает в контекст ({have})",
            detail={"agent_prompt_tokens": need, "n_ctx": have, "recommended_context": recommended},
            fix=f"Поднять contextLength до {recommended} в per-model конфиге "
                f"(weds tune --model {model} --context {recommended}) "
                f"и maxInputTokens в конфиге агента.",
        )

    if not res.ok:
        return CheckResult(
            name="context_fit",
            status=WARN,
            summary=f"проверка не выполнена: {res.error_type}",
            detail={"error": res.error},
        )

    actual = res.prompt_tokens
    n_ctx = srv.context_length(model)
    detail = {"agent_prompt_tokens": actual, "n_ctx": n_ctx}

    if n_ctx and actual > n_ctx * 0.9:
        return CheckResult(
            name="context_fit",
            status=WARN,
            summary=f"влезает, но впритык: {actual} из {n_ctx} токенов",
            detail=detail,
            fix=f"Осталось меньше 10% запаса на историю диалога. Подними контекст "
                f"до {int(actual * th.context_headroom)}.",
        )

    return CheckResult(
        name="context_fit",
        status=PASS,
        summary=f"промпт агента {actual} токенов влезает в {n_ctx or '?'}",
        detail=detail,
    )


def check_tool_calling(srv: LocalServer, model: str) -> CheckResult:
    res = srv.chat(
        model,
        [{"role": "user", "content": TOOL_PROMPT}],
        tools=TOOL_PROBE,
        max_tokens=512,
    )
    if not res.ok:
        return CheckResult(
            name="tool_calling",
            status=FAIL,
            summary=f"запрос с tools упал: {res.error_type}",
            detail={"error": res.error},
            fix="Сервер не принял поле tools. Проверь версию движка и шаблон чата модели.",
        )
    if not res.tool_calls:
        return CheckResult(
            name="tool_calling",
            status=FAIL,
            summary="модель не вернула tool_calls",
            detail={"finish_reason": res.finish_reason, "content": res.content[:200]},
            fix="Агентская петля работать не будет. Либо модель не умеет tool-call, "
                "либо выбран неподходящий шаблон чата. Попробуй другую модель.",
        )

    call = res.tool_calls[0]
    fn = call.get("function") or {}
    args_raw = fn.get("arguments") or ""
    try:
        parsed = json.loads(args_raw) if args_raw else {}
        json_ok = True
    except json.JSONDecodeError:
        parsed, json_ok = None, False

    detail = {
        "tool": fn.get("name"),
        "arguments_raw": args_raw[:300],
        "arguments_parsed": parsed,
    }
    if not json_ok:
        return CheckResult(
            name="tool_calling",
            status=WARN,
            summary="tool_calls есть, но arguments — невалидный JSON",
            detail=detail,
            fix="Аргументы инструментов будут ломаться. Причина обычно в квантовании "
                "KV-кеша (q4_0) — попробуй q8_0 или F16.",
        )

    return CheckResult(
        name="tool_calling",
        status=PASS,
        summary=f"tool-call работает: {fn.get('name')}({json.dumps(parsed, ensure_ascii=False)})",
        detail=detail,
    )


def check_gen_speed(srv: LocalServer, model: str, th: Thresholds, tokens: int = 128) -> CheckResult:
    res = srv.chat(
        model,
        [{"role": "user", "content": "Count from 1 to 500, one number per line."}],
        max_tokens=tokens,
        temperature=0.0,
    )
    if not res.ok:
        return CheckResult(
            name="gen_speed",
            status=WARN,
            summary=f"не измерено: {res.error_type}",
            detail={"error": res.error},
        )
    tps = res.gen_tps
    detail = {
        "tokens": res.completion_tokens,
        "seconds": round(res.elapsed, 2),
        "tps": round(tps, 1),
        "reasoning_tokens": res.reasoning_tokens,
    }
    if tps < th.min_gen_tps:
        return CheckResult(
            name="gen_speed",
            status=FAIL,
            summary=f"{tps:.1f} tok/s — неприемлемо медленно",
            detail=detail,
            fix="Частая причина — часть слоёв осталась на CPU. Поставь полный оффлоад "
                "(offloadRatio = max) или уменьши модель. Проверь также, не переполнен "
                "ли VRAM: тогда генерация обваливается, а префилл остаётся быстрым.",
        )
    if tps < th.warn_gen_tps:
        return CheckResult(
            name="gen_speed",
            status=WARN,
            summary=f"{tps:.1f} tok/s — рабочий минимум, но небыстро",
            detail=detail,
            fix="Полный оффлоад и q4_0/q8_0 KV-кеш дают заметный прирост.",
        )
    return CheckResult(name="gen_speed", status=PASS, summary=f"{tps:.1f} tok/s", detail=detail)


def check_prefill_speed(
    srv: LocalServer, model: str, th: Thresholds, chars_per_token: float, tokens: int = 8000
) -> CheckResult:
    prompt = build_prompt(chars_per_token, tokens, nonce=new_nonce())
    res = srv.chat(model, [{"role": "user", "content": prompt + "\nReply with one word: done"}],
                   max_tokens=1, temperature=0.0)
    if not res.ok:
        status = FAIL if res.is_context_overflow else WARN
        return CheckResult(
            name="prefill_speed",
            status=status,
            summary=f"не измерено: {res.error_type}",
            detail={"error": res.error},
        )
    tps = res.prefill_tps
    detail = {"tokens": res.prompt_tokens, "seconds": round(res.elapsed, 2), "tps": round(tps, 1)}
    if tps < th.min_prefill_tps:
        return CheckResult(
            name="prefill_speed",
            status=WARN,
            summary=f"{tps:.0f} tok/s — холодный старт будет долгим",
            detail=detail,
            fix="Промпт агента ~38k токенов: при такой скорости холодный префилл займёт "
                "минуты. Спасает кэш префикса — он платится один раз за диалог.",
        )
    return CheckResult(
        name="prefill_speed", status=PASS, summary=f"{tps:.0f} tok/s", detail=detail
    )


def check_prefix_cache(
    srv: LocalServer, model: str, th: Thresholds, chars_per_token: float, tokens: int = 6000
) -> CheckResult:
    """Платится ли префилл агента каждый ход или кэшируется.

    Если кэш не работает, локальная модель в агентской петле неюзабельна.
    """
    # Один nonce на все ходы: префикс должен совпадать, иначе кэш не проверить.
    prefix = build_prompt(chars_per_token, tokens, nonce=new_nonce())
    times = []
    for tail in ("alpha", "beta"):
        res = srv.chat(
            model,
            [{"role": "user", "content": f"{prefix}\n\nReply with one word: {tail}"}],
            max_tokens=4,
            temperature=0.0,
        )
        if not res.ok:
            return CheckResult(
                name="prefix_cache",
                status=WARN,
                summary=f"не проверено: {res.error_type}",
                detail={"error": res.error},
            )
        times.append(res.elapsed)

    cold, warm = times[0], times[1]
    speedup = cold / warm if warm > 0 else 0.0
    detail = {"cold_seconds": round(cold, 2), "warm_seconds": round(warm, 2),
              "speedup": round(speedup, 1)}

    if speedup < th.min_prefix_speedup:
        return CheckResult(
            name="prefix_cache",
            status=FAIL,
            summary=f"кэш префикса не работает (ускорение {speedup:.1f}x)",
            detail=detail,
            fix="Каждый ход будет заново префиллить промпт агента — это десятки секунд "
                "на ход. Включи llm.load.llama.contextCheckpoints, держи "
                "numParallelSessions = 1 и не меняй промпт агента между ходами.",
        )
    return CheckResult(
        name="prefix_cache",
        status=PASS,
        summary=f"кэш работает, повтор быстрее в {speedup:.0f}x",
        detail=detail,
    )


def check_reasoning_overhead(srv: LocalServer, model: str, th: Thresholds) -> CheckResult:
    res = srv.chat(
        model,
        [{"role": "user", "content": "How much is 17*23? Reply with the number only."}],
        max_tokens=1024,
        temperature=0.0,
    )
    if not res.ok:
        return CheckResult(
            name="reasoning_overhead",
            status=WARN,
            summary=f"не измерено: {res.error_type}",
            detail={"error": res.error},
        )
    rt = res.reasoning_tokens
    detail = {"reasoning_tokens": rt, "completion_tokens": res.completion_tokens,
              "answer": (res.content or "").strip()[:80], "seconds": round(res.elapsed, 2)}

    if rt == 0:
        return CheckResult(
            name="reasoning_overhead", status=PASS,
            summary="thinking не тратит токены (либо отключён)", detail=detail,
        )
    if rt > th.max_reasoning_tokens:
        return CheckResult(
            name="reasoning_overhead",
            status=WARN,
            summary=f"{rt} reasoning-токенов на тривиальный вопрос",
            detail=detail,
            fix="В агентской петле это прямая потеря времени на каждом шаге. "
                "Ограничь llm.prediction.reasoning.budgetTokens (например, 1024) "
                "или отключи thinking.",
        )
    return CheckResult(
        name="reasoning_overhead", status=PASS,
        summary=f"{rt} reasoning-токенов — в пределах нормы", detail=detail,
    )


def check_context_exact(srv: LocalServer, model: str) -> CheckResult:
    """Сверяет контекст, который реально применился, с ожидаемым."""
    info = pick_model(srv.list_models(), model)
    if not info or info.context is None:
        return CheckResult(
            name="context_exact", status=SKIP,
            summary="сервер не отдаёт загруженный контекст",
        )
    if not info.loaded:
        return CheckResult(
            name="context_exact", status=WARN,
            summary=f"модель не загружена (JIT-загрузка при первом запросе)",
            detail={"context": info.context},
            fix="Первый запрос из агента может упасть по таймауту, пока модель грузится. "
                "Прогрей её заранее: weds load --model " + info.id,
        )
    return CheckResult(
        name="context_exact", status=PASS,
        summary=f"загруженный контекст {info.context}",
        detail={"context": info.context, "max_context": info.max_context,
                "parallel": "см. lms ps"},
    )


# --------------------------------------------------------------------------- #
# Прогон всего набора
# --------------------------------------------------------------------------- #

def run_all(
    srv: LocalServer,
    model: str,
    th: Thresholds,
    *,
    skip_slow: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    r = check_server(srv)
    results.append(r)
    if r.status == FAIL:
        return results

    # Приводим имя к каноническому id сервера: пользователь может передать
    # подстроку ('ornith'), а дальше нужен точный id — иначе поиск контекста
    # в кэше сервера промахивается и context_fit теряет число.
    try:
        info = pick_model(srv.list_models(), model)
        if info is not None:
            model = info.id
    except Exception:
        pass

    r = check_model_present(srv, model)
    results.append(r)
    if r.status == FAIL:
        return results

    results.append(check_context_exact(srv, model))

    try:
        cpt = calibrate_chars_per_token(srv, model)
    except Exception:
        cpt = 4.0

    results.append(check_context_fit(srv, model, th, cpt))
    results.append(check_tool_calling(srv, model))
    results.append(check_reasoning_overhead(srv, model, th))

    if not skip_slow:
        results.append(check_prefix_cache(srv, model, th, cpt))
        results.append(check_prefill_speed(srv, model, th, cpt))
    results.append(check_gen_speed(srv, model, th))

    return results


def summarize(results: list[CheckResult]) -> dict:
    counts = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "pass": counts[PASS],
        "warn": counts[WARN],
        "fail": counts[FAIL],
        "skip": counts[SKIP],
        "verdict": "FAIL" if counts[FAIL] else ("WARN" if counts[WARN] else "PASS"),
    }
