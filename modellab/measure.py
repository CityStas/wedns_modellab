"""Замер скорости: потоковый, с таймстампом на каждый токен и снимком памяти.

Главный подводный камень, ради которого этот модуль существует отдельно.

llama-server в конце ответа печатает собственные `timings`, и они выглядят
как готовый ответ на вопрос «сколько токенов в секунду». Для сравнения
конфигураций они бесполезны: первый токен после префилла стоит секунды
(захват CUDA-графа), поэтому на короткой выборке агрегат измеряет в основном
этот один токен. Реальный случай: агрегат дал 2.72 tok/s там, где честная
устойчивая скорость - 18 tok/s. Разница в 6.6 раза, и вся она в методе замера.

Поэтому: SSE читается чанк за чанком, время пишется на каждый, первые SKIP
токенов выбрасываются, скорость считается по интервалу между последним и
(SKIP+1)-м токеном. Ряд мгновенных значений сохраняется - он же идёт в UI
как график, и по нему видно, ровный это ход или провалы.

Второй момент: замер без снимка памяти невоспроизводим. Та же конфигурация
на этой машине даёт 9.87 tok/s при 1.9 ГБ свободных и 17.90 tok/s при 7.7 ГБ.
Поэтому к каждому результату прикладываются свободная память, рабочий набор
процесса и host_ratio - отношение рабочего набора к размеру файла модели.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import probe

SKIP_TOKENS = 8


@dataclass
class StreamResult:
    ok: bool = False
    err: str = ""
    chunks: int = 0
    skip: int = SKIP_TOKENS
    ttft_s: float | None = None
    t_first_s: float | None = None
    t_last_s: float | None = None
    tps_steady: float | None = None
    tps_total: float | None = None
    ms_per_token: float | None = None
    tps_series: list[float] = field(default_factory=list)
    jitter: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prefill_tps: float | None = None
    server_timings: dict = field(default_factory=dict)
    stop_reason: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def prompt_filler(chars: int, seed: str = "x") -> str:
    """Заполнитель промпта. Одинаковый по длине, разный по тексту.

    `seed` обязателен для повторов: у llama-server включён кеш префикса, и
    второй прогон с тем же текстом читает промпт из кеша. Скорость префилла
    тогда получается фантастической (замерено 1352 tok/s против честных 230
    по llama-bench pp512), а decode почти не страдает - именно поэтому ошибка
    не бросается в глаза, если смотреть только на генерацию.
    """
    if chars <= 0:
        return "ok"
    word = f"{seed}{chars} "
    return (word * (chars // len(word) + 1))[:chars]


def stream_chat(base_url: str, model: str, prompt: str,
                max_tokens: int = 128, skip: int = SKIP_TOKENS,
                timeout: float = 1800.0, temperature: float = 0.0,
                extra_body: dict | None = None) -> StreamResult:
    """Потоковый замер одного ответа.

    `temperature=0` по умолчанию намеренно: замер должен быть воспроизводимым,
    а с температурой модели длина ответа гуляет и вместе с ней - знаменатель.
    """
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "temperature": temperature,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    if extra_body:
        body.update(extra_body)
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})

    res = StreamResult(skip=skip)
    marks: list[float] = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except ValueError:
                    continue
                if d.get("usage"):
                    res.prompt_tokens = d["usage"].get("prompt_tokens")
                    res.completion_tokens = d["usage"].get("completion_tokens")
                if d.get("timings"):
                    res.server_timings = d["timings"]
                choice = (d.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    res.stop_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    marks.append(time.time() - t0)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read(400).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            detail = ""
        res.err = f"HTTP {e.code}: {detail}"
        return res
    except (urllib.error.URLError, OSError, ValueError) as e:
        res.err = str(e)[:300]
        return res

    res.chunks = len(marks)
    if not marks:
        res.err = "сервер не отдал ни одного токена"
        return res
    res.ok = True
    res.ttft_s = round(marks[0], 3)
    res.t_first_s = res.ttft_s
    res.t_last_s = round(marks[-1], 3)
    res.tps_total = round(len(marks) / marks[-1], 2) if marks[-1] > 0 else None

    if len(marks) > skip + 1:
        window = marks[skip:]
        per_token = [(window[i] - window[i - 1]) for i in range(1, len(window))]
        span = window[-1] - window[0]
        res.ms_per_token = round(span * 1000 / len(per_token), 1)
        res.tps_steady = round(len(per_token) / span, 2) if span > 0 else None
        res.tps_series = [round(1 / dt, 2) for dt in per_token if dt > 0]
        if res.tps_series:
            avg = sum(res.tps_series) / len(res.tps_series)
            var = sum((x - avg) ** 2 for x in res.tps_series) / len(res.tps_series)
            res.jitter = round(var ** 0.5 / avg * 100, 1) if avg else None
    else:
        res.err = (f"слишком короткий ответ: {len(marks)} токенов, "
                   f"для устойчивой скорости нужно больше {skip + 1}")

    if res.prompt_tokens and res.ttft_s:
        res.prefill_tps = round(res.prompt_tokens / res.ttft_s, 1)
    return res


def measure(base_url: str, model: str, prompt_chars: int = 200,
            max_tokens: int = 128, skip: int = SKIP_TOKENS,
            watchdog=None, srv=None, model_path: str | None = None,
            repeats: int = 1) -> dict:
    """Замер с памятью: скорость + снимок до/после + низшая точка за прогон.

    `repeats > 1` усредняет: на этой машине разброс между прогонами доходил до
    10%, и одиночный замер легко принять за разницу между конфигами.
    """
    model_size = None
    if model_path:
        mf = probe.model_file(model_path)
        model_size = mf.get("size_mb") if mf.get("exists") else None

    runs: list[StreamResult] = []
    before = probe.snapshot(srv.pid if srv else None)
    for i in range(max(1, repeats)):
        # Свой seed на каждый прогон: иначе повтор читает промпт из кеша
        # префикса и префилл перестаёт быть измерением.
        r = stream_chat(base_url, model, prompt_filler(prompt_chars, seed=f"r{i}"),
                        max_tokens=max_tokens, skip=skip)
        runs.append(r)
        if not r.ok:
            break
    after = probe.snapshot(srv.pid if srv else None)

    good = [r for r in runs if r.ok and r.tps_steady is not None]
    out: dict = {"ok": bool(good), "runs": [r.to_dict() for r in runs],
                 "prompt_chars": prompt_chars, "max_tokens": max_tokens,
                 "repeats": len(runs), "model": model_path}
    if not good:
        out["err"] = runs[-1].err if runs else "нет прогонов"
        return out

    tps = [r.tps_steady for r in good]
    out["tps_steady"] = round(sum(tps) / len(tps), 2)
    out["tps_steady_spread"] = round(max(tps) - min(tps), 2) if len(tps) > 1 else None
    out["tps_total"] = good[-1].tps_total
    out["ttft_s"] = good[-1].ttft_s
    # Префилл берём из ПЕРВОГО прогона: он единственный гарантированно не
    # читает промпт из кеша префикса. Decode усредняется по всем прогонам -
    # на него кеш не влияет.
    first = good[0]
    out["prefill_tps"] = first.prefill_tps
    out["prefill_run"] = 0
    out["ttft_first_s"] = first.ttft_s
    out["ms_per_token"] = good[-1].ms_per_token
    out["jitter_pct"] = good[-1].jitter
    out["tps_series"] = good[-1].tps_series
    out["prompt_tokens"] = first.prompt_tokens
    out["completion_tokens"] = good[-1].completion_tokens
    out["stop_reason"] = good[-1].stop_reason
    out["server_timings"] = first.server_timings

    # Сервер считает скорость сам - показываем оба числа рядом, чтобы расхождение
    # с агрегатом было видно, а не спрятано.
    st = first.server_timings or {}
    if st.get("predicted_per_second"):
        out["server_tps"] = round(float(st["predicted_per_second"]), 2)
    if st.get("prompt_per_second"):
        out["server_prefill_tps"] = round(float(st["prompt_per_second"]), 1)
    # Доказательство, что промпт не пришёл из кеша. Если тут не ноль, цифра
    # префилла недействительна, и это должно быть видно, а не подразумеваться.
    if st.get("prompt_n"):
        out["prompt_n"] = st["prompt_n"]
    if st.get("prompt_ms") and first.prompt_tokens:
        out["prefill_from_cache"] = False
    # Доказательство, что промпт не пришёл из кеша. Ноль - это тоже ответ:
    # если поле отсутствует, непонятно, не было попаданий или их не считали.
    for r in good:
        t = r.server_timings or {}
        if t.get("cache_n") is not None:
            out["cache_hits"] = t["cache_n"]
            break
    out.setdefault("cache_hits", 0)

    out["mem"] = {"before": before.to_dict(), "after": after.to_dict()}
    if srv and srv.pid:
        pm = probe.process_memory(srv.pid) or {}
        out["mem"]["proc_ws_mb"] = pm.get("ws_mb")
        out["mem"]["proc_peak_ws_mb"] = pm.get("peak_ws_mb")
        out["host_ratio"] = probe.host_ratio(model_size, pm.get("ws_mb"))
        out["model_size_mb"] = model_size
    if watchdog:
        rep = watchdog.report()
        out["mem"]["watchdog"] = rep
        out["mem"]["min_avail_mb"] = rep.get("min_avail_mb")
        out["mem"]["min_commit_avail_mb"] = rep.get("min_commit_avail_mb")
    if srv and srv.log_path:
        buf = probe.log_buffers(srv.log_path)
        if buf:
            out["engine"] = {k: buf[k] for k in
                             ("load_mode", "graph_splits", "n_ctx",
                              "all_layers_on_gpu", "vram_mib", "host_mib",
                              "kv_mib", "model_mib", "cpu_mapped_mib",
                              "vram_free_mb", "projected_vram_mb")
                             if k in buf}
    return out


def prefill_curve(base_url: str, model: str, sizes=(200, 2000, 8000, 20000),
                  max_tokens: int = 8, watchdog=None) -> list[dict]:
    """Кривая префилла: как падает скорость обработки промпта с его длиной.

    Нужна, потому что «модель быстрая» и «модель пригодна для работы с
    контекстом» - разные утверждения. На Ternary-Bonsai 27B при ctx 98304
    обработка промпта падала со 116 до 21 tok/s: карта упиралась в 96%, и
    Windows начинал вытеснять память.
    """
    out: list[dict] = []
    for n in sizes:
        r = stream_chat(base_url, model, prompt_filler(n), max_tokens=max_tokens,
                        skip=0)
        row = {"prompt_chars": n, "ok": r.ok, "err": r.err,
               "ttft_s": r.ttft_s, "prompt_tokens": r.prompt_tokens,
               "prefill_tps": r.prefill_tps}
        st = r.server_timings or {}
        if st.get("prompt_per_second"):
            row["server_prefill_tps"] = round(float(st["prompt_per_second"]), 1)
        if watchdog:
            row["min_avail_mb"] = watchdog.report().get("min_avail_mb")
        out.append(row)
    return out
