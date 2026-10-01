"""Подбор конфигурации: сколько контекста влезает и в каком режиме загрузки.

На слабом железе узкое место - не скорость счёта, а память, поэтому подбор
сводится к одному вопросу: какой максимальный контекст влезает на карту
целиком, чтобы не пришлось выкидывать слои на CPU. Перебор тут плох: одна
загрузка модели стоит ~50 с, и сетка по ctx (4k/8k/16k/32k/64k/96k) - это
пять минут ради одной цифры.

Что делается вместо перебора.

Расход VRAM от контекста линеен и складывается из двух слагаемых, которые
движок печатает сам: KV-кеш и буфер вычислений. Значит, достаточно ДВУХ
загрузок на разных ctx, чтобы получить прямую, и потолок по контексту
считается аналитически, а не угадывается:

    ctx_max = ctx_a + (доступно - buf_a) / наклон

Проверено на Ternary-Bonsai-2-27B (KV q4_0, flash-attn on), замер 2026-09-23.
Движок печатает бюджет тремя слагаемыми - веса, контекст, вычисления - и
«контекст» у этой модели включает не только KV, но и рекуррентное состояние
(~150 МБ, постоянно), поэтому в наклон оно не попадает и уходит в intercept:

    ctx   4096 -> контекст  221 + вычисления 126 = 347
    ctx  32768 -> контекст  725 + вычисления 240 = 965
    ctx  49152 -> контекст 1013 + вычисления 320 = 1333

Прямая по первым двум точкам: 260 + 0.02155*ctx, то есть наклон 22.1 МБ на
1024 токена. Третья точка в подгонке не участвовала и проверяет её:
предсказание 5395 + 260 + 0.02155*49152 = 6714 МБ против фактических 6729 МБ -
ошибка 0.2 %. На ctx 65536, который в подгонке тоже не участвовал, прямая даёт
7067 МБ против фактических 7107 МБ (KV 1152 + рекуррент 150 + вычисления 410):
ошибка 0.6 %, и она влезает в резерв 256 МБ.

Отсюда же граница применимости: «контекст» из лога нельзя брать как KV -
рекуррентное состояние сидит в нём же, и прямая, построенная только по
колонке контекста, дала бы завышенный наклон.

Резерв свободной VRAM взят 256 МБ, а не 1024 МБ, как в собственном целевом
показателе движка: движок печатает «cannot meet free memory target of 1024
MiB», но при этом загружается и работает - его проекция переоценивает расход
примерно на 100 МБ (7097 МБ проекции против 7000 МБ свободных дали рабочие
17.9 tok/s). Слишком большой резерв просто съедал бы контекст зря.

Модель-двойник той же архитектуры используется только для проверки формы:
что набор флагов вообще принимается этим билдом, что слои уходят на карту и
граф не рвётся. Величины с двойника не переносятся - у него другое число
слоёв и голов, поэтому его потолок по контексту ничего не говорит о цели.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import llamasrv, measure, probe

PROFILE_FILE = Path(__file__).resolve().parent / "profiles.json"
VRAM_RESERVE_MIB = 256
CTX_STEP = 2048
MIN_CTX = 2048


# --------------------------------------------------------------------------
# профили
# --------------------------------------------------------------------------

def hardware_fingerprint(sysinfo: dict) -> str:
    """Ключ «железо» для профилей. Всё, что влияет на потолок по памяти."""
    gpu = sysinfo.get("gpu") or {}
    cpu = sysinfo.get("cpu") or {}
    mem = sysinfo.get("mem") or {}
    parts = [gpu.get("name") or "?", str(gpu.get("vram_total_mb") or "?"),
             cpu.get("name") or "?", str(mem.get("total_mb") or "?")]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def profile_key(model_info: dict, sysinfo: dict) -> str:
    """Ключ профиля: модель + железо.

    Размер модели и число слоёв входят, потому что от них зависит и расход
    VRAM, и потолок по контексту; путь и дата файла - нет, иначе профиль
    терялся бы при переименовании каталога.
    """
    parts = [hardware_fingerprint(sysinfo),
             str(model_info.get("size_mb") or "?"),
             str(model_info.get("layers") or "?"),
             str(model_info.get("arch") or "?")]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load_profiles(path: Path = PROFILE_FILE) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_profile(key: str, data: dict, path: Path = PROFILE_FILE) -> None:
    """Дописать профиль, не затерев остальные.

    Запись через временный файл: если процесс упадёт посреди записи, файл
    профилей не превратится в обрезанный JSON, из-за которого потеряются
    все остальные профили.
    """
    prof = load_profiles(path)
    prof[key] = data
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def profile_headroom(prof: dict) -> float | None:
    """Запас VRAM, записанный в профиль (или пересчитанный из прямой)."""
    if prof.get("headroom_mib") is not None:
        return prof["headroom_mib"]
    vm = prof.get("vram_model") or {}
    ctx = (prof.get("cfg") or {}).get("ctx")
    try:
        slope = float(vm["slope_mib_per_token"])
        inter = float(vm["intercept_mib"])
        model = float(vm["model_mib"])
        base = float(vm.get("vram_free_mib") or vm.get("vram_total_mib"))
        return round(base - model - inter - slope * float(ctx), 1)
    except (KeyError, TypeError, ValueError):
        return None


def profile_reuse_check(prof: dict,
                        free_now_mib: float | None) -> tuple[bool | None, str]:
    """Можно ли применить сохранённый профиль, не перемеряя.

    Профиль хранит не только контекст, но и обстоятельства, при которых он был
    выбран: сколько VRAM было свободно и какой запас оставался. Свободная VRAM
    на этой машине плавает на сотни мегабайт от одного лишь рабочего стола и
    браузера (замер: движок на загрузке видел 7000 МБ, nvidia-smi через две
    минуты - 7488 МБ). Поэтому «профиль говорит 49152» само по себе ещё не
    значит, что сегодня 49152 влезет, и молча применять его нельзя.

    Сравнивается РАЗНОСТЬ: насколько сейчас свободнее/теснее, чем было на
    момент замера. Так систематический сдвиг между источниками (движок считает
    свободное сам, nvidia-smi - по-своему) сокращается, и остаётся то, что
    действительно изменилось. Если карта стала теснее на величину больше
    записанного запаса - профиль не применяется, нужен новый замер.

    Возвращает (True, пояснение) / (False, причина) / (None, почему нельзя
    проверить). None - это не «всё хорошо»: без числа проверка не состоялась,
    и вызывающий обязан сказать об этом вслух.
    """
    ctx = (prof.get("cfg") or {}).get("ctx")
    stored_free = (prof.get("vram_model") or {}).get("vram_free_mib")
    head = profile_headroom(prof)
    if not ctx or not stored_free or head is None:
        return None, "в профиле нет данных о свободной VRAM на момент замера"
    if free_now_mib is None:
        return None, (f"на ctx {ctx} запас был {head} МБ при {stored_free} МБ "
                      f"свободных, но сейчас свободную VRAM прочитать не удалось")
    shortfall = float(stored_free) - float(free_now_mib)
    if shortfall > head:
        return False, (f"на ctx {ctx} нужно {float(stored_free) - head:.0f} МБ "
                       f"при {stored_free} МБ свободных (запас {head} МБ), а "
                       f"сейчас свободно {float(free_now_mib):.0f} МБ: нехватка "
                       f"{shortfall:.0f} МБ больше запаса")
    return True, (f"на ctx {ctx} запас {head} МБ, свободной VRAM сейчас "
                  f"{float(free_now_mib):.0f} МБ против {stored_free} МБ на "
                  f"момент замера")


# --------------------------------------------------------------------------
# линейная модель VRAM
# --------------------------------------------------------------------------

@dataclass
class VramModel:
    """Расход VRAM, зависящий от контекста: buf(ctx) = intercept + slope*ctx."""
    ctx_a: int = 0
    buf_a: float = 0.0
    ctx_b: int = 0
    buf_b: float = 0.0
    slope_mib_per_token: float = 0.0
    intercept_mib: float = 0.0
    model_mib: float = 0.0
    vram_free_mib: float | None = None
    vram_total_mib: float | None = None
    load_mode: str = ""

    @property
    def slope_mib_per_1k(self) -> float:
        return round(self.slope_mib_per_token * 1024, 1)

    def predict(self, ctx: int) -> float:
        return self.intercept_mib + self.slope_mib_per_token * ctx

    def max_ctx(self, reserve_mib: int = VRAM_RESERVE_MIB) -> int | None:
        """Наибольший контекст, при котором всё влезает на карту.

        Считается от СВОБОДНОЙ памяти, а не от полной: часть карты занимает
        рабочий стол, и на этой машине это около 1.2 ГБ (8191 МБ всего против
        7000 МБ свободных в момент старта движка).
        """
        base = self.vram_free_mib if self.vram_free_mib else self.vram_total_mib
        if not base or self.slope_mib_per_token <= 0:
            return None
        room = base - self.model_mib - reserve_mib
        if room <= self.predict(MIN_CTX):
            return None
        ctx = (room - self.intercept_mib) / self.slope_mib_per_token
        return int(ctx // CTX_STEP * CTX_STEP)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["slope_mib_per_1k"] = self.slope_mib_per_1k
        return d

    def headroom_mib(self, ctx: int) -> float | None:
        """Сколько VRAM останется свободной, если загрузиться на этом ctx.

        Нужен, чтобы «влезает» было числом, а не вердиктом. Разница между
        «влезает с 271 МБ в запасе» и «влезает с 1024 МБ в запасе» - это
        разница между сервером, который падает, когда пользователь откроет
        браузер, и сервером, который этого не замечает.
        """
        base = self.vram_free_mib if self.vram_free_mib else self.vram_total_mib
        if not base:
            return None
        return round(base - self.model_mib - self.predict(ctx), 1)



def fit_vram_model(probes: list[dict], load_mode: str = "") -> VramModel | None:
    """Построить прямую по двум точкам (ctx, расход на контекст).

    Расход на контекст = всё, что движок выделил сверх весов: KV-кеш,
    рекуррентное состояние и буфер вычислений. Именно эта часть растёт с
    контекстом; веса от него не зависят.
    """
    pts = []
    for p in probes:
        b = p.get("buf") or {}
        dev = (b.get("devices") or {}).get("CUDA0") or {}
        model_mib = dev.get("model_mb") or b.get("model_mib") or 0
        self_mib = dev.get("self_mb") or 0
        ctx = b.get("n_ctx") or p.get("ctx")
        if not (ctx and self_mib and model_mib):
            continue
        pts.append((int(ctx), float(self_mib - model_mib),
                    float(model_mib), b.get("vram_free_mb"),
                    b.get("vram_total_mb")))
    if len(pts) < 2:
        return None
    (ca, ba, model_mib, free_mib, total_mib), (cb, bb, _, _, _) = pts[0], pts[1]
    if cb == ca:
        return None
    slope = (bb - ba) / (cb - ca)
    # Режим загрузки берём из разбора лога: в params его может не быть вовсе
    # (подбор не обязан его задавать), а пустая строка в отчёте выглядела бы
    # как «режим неизвестен» там, где движок его напечатал.
    lm = load_mode or next(
        ((p.get("buf") or {}).get("load_mode") for p in probes
         if (p.get("buf") or {}).get("load_mode")), "")
    return VramModel(ctx_a=ca, buf_a=ba, ctx_b=cb, buf_b=bb,
                     slope_mib_per_token=slope, intercept_mib=ba - slope * ca,
                     model_mib=model_mib, vram_free_mib=free_mib,
                     vram_total_mib=total_mib, load_mode=lm)



# --------------------------------------------------------------------------
# двойник
# --------------------------------------------------------------------------

def find_proxy(model_info: dict, models: list[dict],
               max_mb: int = 1500) -> dict | None:
    """Наименьшая модель той же архитектуры, но не та же самая.

    Смысл - не в переносе цифр, а в проверке формы на дешёвой модели: набор
    флагов принят билдом, слои ушли на карту, граф не порвался. Это ловит
    несовместимость флагов за 40 с на модели в 500 МБ вместо 50 с на модели
    в 5.7 ГБ, которая при этом может не влезть.
    """
    cands = [m for m in models
             if m.get("arch") == model_info.get("arch")
             and m.get("path") != model_info.get("path")
             and (m.get("size_mb") or 0) <= max_mb]
    if not cands:
        return None
    return min(cands, key=lambda m: m.get("size_mb") or 0)


# --------------------------------------------------------------------------
# выгрузка в launcher
# --------------------------------------------------------------------------

_CMD_HEADER = r"""@echo off
rem ---------------------------------------------------------------
rem  Configuration picked by modellab {stamp}.
rem
rem  {why}
rem
rem  Measured: decode {tps} tok/s, prefill {pp} tok/s, TTFT {ttft} s,
rem  host_ratio {hr}, minimum free RAM {minram} MB.
rem
rem  WHY THE CONTEXT IS THE NUMBER TO WATCH: it is the largest context
rem  that still fits on the card with {reserve} MB to spare, computed
rem  from a line fitted on two real loads - not guessed from a table.
rem  Raise it and the engine starts paging VRAM out through WDDM, which
rem  is what kills llama-server mid-answer on this machine; lower it and
rem  you pay context for nothing.
rem
rem  -lv is deliberately NOT set here. Verbosity 4 is what makes the
rem  engine print its VRAM budget while measuring, but it floods the
rem  console during normal use. Run the server from modellab if you
rem  need the budget lines.
rem
rem  WARNING: this file is generated; manual edits are lost the next
rem  time the picker runs. The previous version is kept as {bak}.
rem
rem  NOTE: this file must keep CRLF line endings and ASCII-only text.
rem  cmd.exe mis-parses LF-only batch files, and non-ASCII bytes in a
rem  cp866/cp1251 console corrupt the parsing. The header is written in
rem  English for that reason - it is not decoration.
rem ---------------------------------------------------------------
setlocal
set "LAB_BIN={bin}"
set "LAB_EXE={exe_name}"
set "LAB_MODEL={model}"
set "LAB_CTX={ctx}"
set "LAB_PORT={port}"
set "LAB_LOG={log}"
if not exist "%LAB_BIN%\%LAB_EXE%" (
  echo [modellab] server not found: %LAB_BIN%\%LAB_EXE%
  pause
  exit /b 1
)
if not exist "%LAB_MODEL%" (
  echo [modellab] model not found: %LAB_MODEL%
  pause
  exit /b 1
)
if not exist "%LAB_LOG%" mkdir "%LAB_LOG%"
echo [modellab] model = %LAB_MODEL%
echo [modellab] ctx   = %LAB_CTX%   ngl={ngl}   load-mode={load_mode}   kv={kv}
echo [modellab] host  = 127.0.0.1:%LAB_PORT%
echo.
"%LAB_BIN%\%LAB_EXE%" ^
{server}
"""


def export_cmd(cfg: llamasrv.ServerConfig, path: str, result: dict | None = None,
               why: str = "", log_dir: str | None = None) -> dict:
    """Записать конфиг в .cmd рядом с прежним, прежний - в .bak.

    Отдельная функция, а не «просто write_text», из-за двух вещей, на которых
    этот проект уже спотыкался: cmd.exe разбирает bat-файл по байтовому
    смещению (нужны CRLF и ASCII), и перезапись рабочего лаунчера без копии
    лишает пользователя возможности откатиться.

    Заголовок - по-английски, и это не вкусовщина. В проекте уже есть рабочие
    лаунчеры (start-bonsai.cmd), и там прямо записано правило: файл остаётся
    CRLF и ASCII-only, потому что не-ASCII байты в консоли cp866/cp1251 ломают
    разбор. Первая версия этой функции писала заголовок по-русски и тут же
    заменяла 323 символа на «?» - то есть выдавала файл, который сам себе
    противоречил: комментарий обещал ASCII, а внутри лежала каша.

    Флаги печатаются парами «опция + значение» в одной строке. Вариант
    «по одному токену на строку» тоже работает, но значение на отдельной
    строке с продолжением-кареткой - это лишняя точка отказа и нечитаемый
    файл, который пользователь должен суметь проверить глазами.

    Кавычек вокруг подстановки команды быть не должно. В первой версии
    шаблон заканчивался на «"{server}"», и файл получал строку
    `"  -m "%LAB_MODEL%" ^` плюс одинокую кавычку в конце: cmd.exe разбирал
    `"  -m "` как имя команды, и лаунчер не запускался ни при каких
    настройках. Файл при этом создавался, был непустой и выглядел правдоподобно
    - поэтому проверка на чётность кавычек в каждой строке есть в тестах.
    """
    p = Path(path)
    # Каталог создаём сами: путь вводится руками в UI, и падать на
    # FileNotFoundError после того, как копия прежнего файла уже сделана,
    # значит оставить пользователя с .bak и без нового лаунчера.
    p.parent.mkdir(parents=True, exist_ok=True)
    bak = p.with_suffix(p.suffix + ".bak")
    if p.exists():
        shutil.copy2(p, bak)
    result = result or {}
    # -c вынесен в переменную %LAB_CTX% и добавлен в шаблон ниже; -lv здесь
    # лишний (см. заголовок файла).
    skip_next = {"-m", "--alias", "--host", "--port", "-c", "-lv"}
    argv = cfg.to_argv()[1:]
    groups: list[str] = []
    i = 0
    while i < len(argv):
        v = argv[i]
        if v in skip_next:
            i += 2
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if v.startswith("-") and nxt is not None and not nxt.startswith("-"):
            groups.append(f"{v} {nxt}")
            i += 2
        else:
            groups.append(v)
            i += 1
    # Каретка ставится МЕЖДУ строками, а не в конце каждой: последняя строка
    # блока не должна заканчиваться на «^» - за ней уже ничего нет, и cmd
    # будет ждать продолжения до конца файла.
    body = " ^\n".join(f"  {g}" for g in groups)

    text = _CMD_HEADER.format(
        stamp=time.strftime("%Y-%m-%d %H:%M"), why=why or "Picked automatically.",
        tps=result.get("tps_steady", "?"), pp=result.get("prefill_tps", "?"),
        ttft=result.get("ttft_s", "?"), hr=result.get("host_ratio", "?"),
        minram=(result.get("mem") or {}).get("min_avail_mb", "?"),
        reserve=VRAM_RESERVE_MIB,
        bak=bak.name, bin=str(Path(cfg.exe).parent), exe_name=Path(cfg.exe).name,
        model=cfg.model,

        ctx=cfg.ctx, port=cfg.port, ngl=cfg.ngl, load_mode=cfg.load_mode,
        kv=cfg.cache_type_k, log=log_dir or str(Path(path).parent / "logs"),
        server=f'  -m "%LAB_MODEL%" ^\n'
               f'  --alias {cfg.alias} ^\n'
               f'  --host 127.0.0.1 --port %LAB_PORT% ^\n'
               f'  -c %LAB_CTX% ^\n'
               f'  --log-file "%LAB_LOG%\\server-live.log" ^\n'
               f'{body}')
    # Перевод в CRLF и запрет не-ASCII: иначе cmd.exe ломается.
    data = text.replace("\r\n", "\n").replace("\n", "\r\n")
    bad = [c for c in data if ord(c) > 127]
    data = data.encode("ascii", "replace").decode("ascii")
    p.write_bytes(data.encode("ascii"))
    out = {"path": str(p), "backup": str(bak) if bak.exists() else None,
           "bytes": len(data), "crlf": data.count("\r\n"),
           "non_ascii": len(bad), "ctx_flag": "-c %LAB_CTX%" in data,
           # По списку групп, а не поиском подстроки в файле: заголовок сам
           # упоминает «-lv» в объяснении, почему его здесь нет.
           "verbose_flag": any(g.startswith("-lv") for g in groups),
           "flags": groups}

    if bad:
        # Молча испортить файл нельзя: именно так и появился заголовок из «?».
        out["note"] = (f"в тексте было {len(bad)} не-ASCII символов, они "
                       f"заменены на '?': {''.join(sorted(set(bad)))[:40]}")
    return out



# --------------------------------------------------------------------------
# сам подбор
# --------------------------------------------------------------------------

def _logbuf(srv: llamasrv.LlamaServer) -> dict:
    return probe.log_buffers(srv.log_path) if srv and srv.log_path else {}


def _probe_ctx(lab, job, cfg_kwargs: dict, ctx: int) -> dict:
    """Одна загрузка на заданном контексте. Возвращает разбор лога и память."""
    job.say(f"проба ctx {ctx}: загружаю")
    lab._start_sync(job, {**cfg_kwargs, "ctx": ctx})
    srv = lab.srv
    buf = _logbuf(srv)
    wd = lab.wd
    row = {"ctx": ctx, "buf": buf, "mem": wd.report() if wd else {},
           "ok": buf.get("all_layers_on_gpu") is True}
    if buf.get("vram_hint"):
        job.say("  движок не отдал бюджет VRAM: " + buf["vram_hint"])
    dev = (buf.get("devices") or {}).get("CUDA0") or {}
    if dev:
        job.say(f"  веса {dev.get('model_mb')} + контекст {dev.get('context_mb')} "
                f"+ вычисления {dev.get('compute_mb')} = {dev.get('self_mb')} МБ "
                f"при свободных {dev.get('free_mb')} МБ")
    job.say(f"  слоёв на GPU {buf.get('layers_gpu')}/{buf.get('layers_total')}, "
            f"graph splits {buf.get('graph_splits')}, "
            f"load-mode {buf.get('load_mode')}")
    lab.stop_now(job)
    return row


def ab(lab, job, params: dict) -> dict:
    """Перемежающееся сравнение конфигураций. Единственный честный способ.

    Почему не «замерить A, замерить B и сравнить».

    На этой машине одна и та же конфигурация в соседних прогонах дала 8.12 и
    15.38 tok/s - разброс в 1.9 раза. Он берётся не из модели, а из окружения:
    рабочий стол, браузер, обновление кеша ОС, тепловой режим карты. На таком
    фоне однократное сравнение двух конфигураций измеряет шум, а не эффект,
    и именно так рождаются «оптимизации», которые ничего не оптимизируют.

    Поэтому прогоны перемежаются: A, B, A, B, ... Каждая пара видит примерно
    одно и то же состояние машины, и сравнивать надо разности внутри пары, а
    не средние по конфигурациям. Знак разности должен быть одинаковым во всех
    парах - тогда эффект реален; если знак скачет, значит различие меньше шума,
    и это тоже результат, а не неудача.
    """
    variants = params.get("variants") or []
    if len(variants) < 2:
        raise ValueError("нужно минимум два варианта для сравнения")
    repeats = int(params.get("repeats", 3))
    prompt_chars = int(params.get("prompt_chars", 400))
    max_tokens = int(params.get("max_tokens", 96))
    model = params.get("model") or lab.last_config.get("model")
    if not model:
        raise ValueError("не задана модель")

    rows: list[dict] = []
    for rep in range(repeats):
        for v in variants:
            name = v.get("name") or v.get("load_mode") or f"v{len(rows)}"
            over = {k: val for k, val in v.items() if k != "name"}
            job.say(f"повтор {rep + 1}/{repeats}, вариант {name}: поднимаю")
            lab._start_sync(job, {**over, "model": model})
            srv = lab.srv
            m = measure.measure(srv.base_url, srv.cfg.alias,
                                prompt_chars=prompt_chars, max_tokens=max_tokens,
                                watchdog=lab.wd, srv=srv, model_path=model,
                                repeats=1)
            buf = lab.engine_now()
            dev = (buf.get("devices") or {}).get("CUDA0") or {}
            # VRAM и разбивку буферов пишем в строку обязательно: сравнение
            # KV-кеша или режима загрузки - это в первую очередь вопрос
            # памяти, и без этих чисел A/B отвечает только на половину
            # вопроса (скорость), причём на ту половину, где как раз шум.
            row = {"rep": rep, "variant": name, "cfg": over,
                   "tps": m.get("tps_steady"), "prefill": m.get("prefill_tps"),
                   "ws_mb": (m.get("mem") or {}).get("proc_ws_mb"),
                   "host_ratio": m.get("host_ratio"),
                   "min_avail_mb": (m.get("mem") or {}).get("min_avail_mb"),
                   "graph_splits": buf.get("graph_splits"),
                   "load_mode": buf.get("load_mode"),
                   "vram_mib": buf.get("vram_mib"),
                   "context_mb": dev.get("context_mb"),
                   "self_mb": dev.get("self_mb"),
                   "cache_n": m.get("cache_hits"),
                   "ok": m.get("ok"), "err": m.get("err", "")}

            rows.append(row)
            job.say(f"  {name}: {row['tps']} tok/s, "
                    f"рабочий набор {row['ws_mb']} МБ, "
                    f"минимум RAM {row['min_avail_mb']} МБ")
            lab.stop_now(job)

    names = []
    for v in variants:
        n = v.get("name") or v.get("load_mode") or "?"
        if n not in names:
            names.append(n)
    summary: list[dict] = []
    for n in names:
        vals = [r["tps"] for r in rows if r["variant"] == n and r.get("tps")]
        if not vals:
            summary.append({"variant": n, "n": 0})
            continue
        vram = [r["vram_mib"] for r in rows
                if r["variant"] == n and r.get("vram_mib")]
        ctxb = [r["context_mb"] for r in rows
                if r["variant"] == n and r.get("context_mb") is not None]
        summary.append({"variant": n, "n": len(vals),
                        "mean_tps": round(sum(vals) / len(vals), 2),
                        "min_tps": min(vals), "max_tps": max(vals),
                        "spread": round(max(vals) - min(vals), 2),
                        "mean_vram_mib": (round(sum(vram) / len(vram), 1)
                                          if vram else None),
                        "mean_context_mb": (round(sum(ctxb) / len(ctxb), 1)
                                            if ctxb else None),
                        "load_mode": next((r["load_mode"] for r in rows
                                           if r["variant"] == n
                                           and r.get("load_mode")), None)})


    verdict: dict = {}
    if len(names) >= 2:
        a, b = names[0], names[1]
        deltas = []
        for rep in range(repeats):
            ra = next((r for r in rows if r["variant"] == a and r["rep"] == rep), None)
            rb = next((r for r in rows if r["variant"] == b and r["rep"] == rep), None)
            if ra and rb and ra.get("tps") and rb.get("tps"):
                deltas.append(round(rb["tps"] - ra["tps"], 2))
        if deltas:
            same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
            mean_d = sum(deltas) / len(deltas)
            verdict = {"a": a, "b": b, "deltas": deltas,
                       "mean_delta": round(mean_d, 2),
                       "same_sign": same_sign,
                       "conclusion": (
                           f"{b} быстрее {a} на {abs(mean_d):.2f} tok/s "
                           f"во всех {len(deltas)} парах - эффект реальный"
                           if same_sign and abs(mean_d) > 0.5 else
                           f"различие между {a} и {b} меньше разброса "
                           f"(знак меняется: {deltas}) - на этом числе "
                           f"прогонов эффект не доказан")}
            job.say("вывод: " + verdict["conclusion"])

    return {"model": model, "repeats": repeats, "rows": rows,
            "summary": summary, "verdict": verdict}


def run(lab, job, params: dict) -> dict:
    """Полный подбор. Вызывается из Lab.search как тело задачи."""
    model = params.get("model")
    if not model:
        raise ValueError("не задана модель")
    # Нормализация до всего остального: путь приезжает то с прямыми слэшами
    # (Git Bash переписывает аргументы), то с обратными, а profile_key и
    # запись в профиль должны видеть одну и ту же строку. config_for
    # нормализует сам, но winner и профиль собираются здесь.
    model = os.path.normpath(model)
    models = lab.models()
    info = lab.model_info(model) or {"path": model}
    sysinfo = lab.system()
    key = profile_key(info, sysinfo)

    report: dict = {"model": model, "model_info": info, "key": key,
                    "hardware": hardware_fingerprint(sysinfo), "steps": [],
                    "proxy": None, "vram_model": None, "winner": None,
                    "profile_hit": None}

    if not params.get("force"):
        prof = load_profiles().get(key)
        if prof and prof.get("cfg"):
            job.say(f"профиль найден: {key} от {prof.get('stamp')}, "
                    f"ctx {prof['cfg'].get('ctx')}, "
                    f"{prof.get('result', {}).get('tps_steady')} tok/s")
            report["profile_hit"] = prof
            if params.get("reuse", True):
                free_now = (sysinfo.get("gpu") or {}).get("vram_free_mb")
                ok, why = profile_reuse_check(prof, free_now)
                report["reuse_check"] = {"ok": ok, "why": why,
                                         "free_now_mib": free_now}
                if ok is False:
                    job.say("профиль найден, но применять его сейчас нельзя: "
                            + why)
                    job.say("меряю заново - контекст выбирается под текущую "
                            "карту, а не под вчерашнюю")
                else:
                    job.say(("профиль применяю: " if ok else
                             "профиль применяю без проверки: ") + why)
                    report["winner"] = prof["cfg"]
                    report["result"] = prof.get("result")
                    job.say("подбор не нужен "
                            "(force=true чтобы перемерить принудительно)")
                    return report


    ctx_min = int(params.get("ctx_min", 4096))
    ctx_mid = int(params.get("ctx_mid", 32768))
    ctx_train = int(info.get("ctx_train") or params.get("ctx_max", 65536))
    ngl = int(params.get("ngl", 99))
    base = {"model": model, "ngl": ngl,
            "cache_type_k": params.get("cache_type_k", "q4_0"),
            "cache_type_v": params.get("cache_type_v", "q4_0"),
            "flash_attn": params.get("flash_attn", "on")}

    # -- 1. двойник: проверка формы на дешёвой модели --------------------
    if params.get("use_proxy", True):
        proxy = find_proxy(info, models)
        if proxy:
            job.say(f"двойник: {proxy['name']} ({proxy['size_mb']} МБ, "
                    f"{proxy.get('arch')}, {proxy.get('layers')} слоёв) - "
                    f"проверяю форму флагов, величины с него не переношу")
            row = _probe_ctx(lab, job, {**base, "model": proxy["path"]},
                             min(ctx_min, 4096))
            row["model"] = proxy["name"]
            report["proxy"] = row
            if not row["ok"]:
                job.say("  ВНИМАНИЕ: даже на двойнике не все слои ушли на карту")
        else:
            job.say("двойник той же архитектуры не найден - пропускаю")

    # -- 2. две пробы на цели: прямая расхода VRAM от контекста ----------
    probes = []
    for ctx in (ctx_min, ctx_mid):
        if ctx > ctx_train:
            continue
        row = _probe_ctx(lab, job, base, ctx)
        row["step"] = f"probe@{ctx}"
        probes.append(row)
        report["steps"].append(row)
        if not row["ok"]:
            job.say(f"  контекст {ctx} не влез целиком - модель уходит на CPU, "
                    f"дальше подбор бессмысленен")
            break

    vm = fit_vram_model(probes, load_mode=base.get("load_mode", ""))
    if vm is None:
        job.say("не удалось построить прямую расхода VRAM: движок не отдал "
                "бюджет по устройствам (нужен -lv 4)")
        report["error"] = "нет данных о бюджете VRAM"
        return report
    report["vram_model"] = vm.to_dict()
    job.say(f"прямая: расход на контекст = {vm.intercept_mib:.0f} + "
            f"{vm.slope_mib_per_token:.4f}*ctx МБ "
            f"(наклон {vm.slope_mib_per_1k} МБ на 1024 токена)")
    ctx_ceiling = vm.max_ctx()
    if ctx_ceiling is None:
        job.say("прямая не даёт потолка: наклон нулевой или свободной VRAM мало")
        report["error"] = "потолок по контексту не определён"
        return report
    ctx_fit = min(ctx_ceiling, ctx_train)
    job.say(f"расчётный потолок: {ctx_ceiling} токенов "
            f"(предел модели {ctx_train}) -> беру {ctx_fit}")
    hr = vm.headroom_mib(ctx_fit)
    if hr is not None:
        job.say(f"  на {ctx_fit} в запасе остаётся {hr} МБ VRAM "
                f"(резерв {VRAM_RESERVE_MIB} МБ)")
    report["ctx_ceiling"] = ctx_ceiling
    report["ctx_fit"] = ctx_fit
    report["headroom_mib"] = hr

    # -- 3. подтверждение и замер на выбранном контексте -----------------
    winner = None
    attempt_ctx = ctx_fit
    for attempt in range(int(params.get("attempts", 3))):
        row = _probe_ctx(lab, job, base, attempt_ctx)
        row["step"] = f"confirm@{attempt_ctx}"
        report["steps"].append(row)
        if not row["ok"]:
            job.say(f"  ctx {attempt_ctx} не влез - шагаю вниз на четверть")
            attempt_ctx = max(MIN_CTX, int(attempt_ctx * 0.75 // CTX_STEP * CTX_STEP))
            continue
        lab._start_sync(job, {**base, "ctx": attempt_ctx})
        srv = lab.srv
        m = measure.measure(srv.base_url, srv.cfg.alias,
                            prompt_chars=int(params.get("prompt_chars", 400)),
                            max_tokens=int(params.get("max_tokens", 128)),
                            watchdog=lab.wd, srv=srv, model_path=model,
                            repeats=int(params.get("repeats", 2)))
        m["engine"] = lab.engine_now()
        row["measure"] = m
        winner = {"ctx": attempt_ctx, **base}
        report["winner"] = winner
        report["result"] = m
        job.say(f"  ctx {attempt_ctx}: {m.get('tps_steady')} tok/s "
                f"(префилл {m.get('prefill_tps')}), host_ratio {m.get('host_ratio')}")
        lab.stop_now(job)
        if m.get("ok"):
            break
        job.say(f"  замер не удался: {m.get('err')}")
        attempt_ctx = max(MIN_CTX, int(attempt_ctx * 0.75 // CTX_STEP * CTX_STEP))

    if not report.get("winner"):
        job.say("подбор не дал рабочей конфигурации")
        return report

    # -- 4. профиль и выгрузка -------------------------------------------
    if report.get("result", {}).get("ok"):
        save_profile(key, {"stamp": time.strftime("%Y-%m-%d %H:%M"),
                           "cfg": report["winner"], "result": report["result"],
                           "vram_model": report["vram_model"],
                           "headroom_mib": report.get("headroom_mib"),
                           "ctx_ceiling": report.get("ctx_ceiling"),
                           "model": info, "hardware": report["hardware"]})
        job.say(f"профиль сохранён: {key}")

    out = params.get("export")
    if out:
        cfg = llamasrv.config_for(model, **{k: v for k, v in report["winner"].items()
                                           if k != "model"})
        why = (f"Context {report['winner']['ctx']} picked from a fitted VRAM "
               f"line: {report['vram_model']['slope_mib_per_1k']} MiB per 1024 "
               f"tokens, ceiling {report.get('ctx_ceiling')} against "
               f"{report['vram_model'].get('vram_free_mib')} MiB free VRAM, "
               f"leaving {report.get('headroom_mib')} MiB headroom.")
        info2 = export_cmd(cfg, out, report.get("result"), why,
                           log_dir=str(Path(out).parent / "logs"))
        report["export"] = info2
        job.say(f"launcher записан: {info2['path']} "
                f"({info2['bytes']} байт, CRLF {info2['crlf']}, "
                f"не-ASCII {info2['non_ascii']}, копия {info2['backup']})")
    return report
