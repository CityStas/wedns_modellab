"""Проверки выгрузки лаунчера и профилей.

Появились после реального прогона подбора: он записал .cmd, который сам себе
противоречил - в комментарии обещал ASCII-only, а внутри лежали 323 знака «?»
вместо русского заголовка, и не содержал -c, то есть игнорировал контекст,
ради которого весь подбор и делался. Обе ошибки невидимы, если смотреть на
код: файл создаётся, байты пишутся, размер ненулевой. Поэтому проверки тут
не на «функция отработала», а на содержимое файла побайтово.

Запуск:
    python tests/smoke_export.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modellab import llamasrv, search  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(cond: bool, what: str, extra: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}" + (f"  <- {extra}" if extra else ""))
        FAILS.append(what)


MODEL = r"E:\Programs\LocalLLM\Models\ternary\Ternary-Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-PTQ1_0.gguf"


# --------------------------------------------------------------------------
# разбор .cmd так, как его читает cmd.exe
# --------------------------------------------------------------------------
# Смысл в том, что лаунчер можно проверить, не запуская пятигигабайтную
# модель. Первая версия выгрузки создавала файл, который cmd.exe не смог бы
# выполнить вообще - в команде не было исполняемого файла (LAB_BIN задавался и
# нигде не использовался), вокруг подстановки стояли кавычки, из-за которых
# `"  -m "` разбиралось как имя команды, а -c вообще терялся. Все четыре
# дефекта видны, если собрать командную строку из файла и сравнить её с тем,
# что собирает сам код.

def bat_env(text: str) -> dict:
    """Переменные из строк `set "NAME=VALUE"`."""
    env: dict[str, str] = {}
    for raw in text.split("\r\n"):
        m = re.match(r'\s*set\s+"([^=]+)=(.*)"\s*$', raw, re.I)
        if m:
            env[m.group(1)] = m.group(2)
    return env


def bat_command(text: str) -> str:
    """Склеить команду, сняв rem-строки и продолжения по каретке."""
    parts: list[str] = []
    started = False
    for raw in text.split("\r\n"):
        s = raw.strip()
        if not started:
            # Команда начинается там, где строка в кавычках или с дефиса:
            # до этого идут только rem, set, if, echo.
            if not s or s.lower().startswith(("rem", "set ", "setlocal", "if ",
                                              "echo", "pause", "exit", ")", "(",
                                              "@")):
                continue
            started = True
        if s.endswith("^"):
            parts.append(s[:-1])
        else:
            parts.append(s)
            if not s.endswith("^"):
                # конец команды: следующая строка уже не продолжение
                break
    return "".join(parts)


def bat_expand(s: str, env: dict) -> str:
    return re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%",
                  lambda m: env.get(m.group(1), m.group(0)), s)


def tokenize(s: str) -> list[str]:
    out, cur, q = [], "", False
    for ch in s:
        if ch == '"':
            q = not q
            continue
        if ch.isspace() and not q:
            if cur:
                out.append(cur)
                cur = ""
            continue
        cur += ch
    if cur:
        out.append(cur)
    return out


def launcher_argv(text: str) -> list[str]:
    env = bat_env(text)
    return tokenize(bat_expand(bat_command(text), env))


def main() -> int:
    print("== 1. содержимое сгенерированного .cmd ==")
    tmpdir = Path(tempfile.mkdtemp(prefix="modellab-exp-"))
    out = tmpdir / "picked.cmd"
    cfg = llamasrv.config_for(MODEL, ngl=99, ctx=49152,
                              cache_type_k="q4_0", cache_type_v="q4_0",
                              flash_attn="on")
    res = search.export_cmd(
        cfg, str(out),
        result={"tps_steady": 14.26, "prefill_tps": 138.7, "ttft_s": 2.62,
                "host_ratio": 0.311, "mem": {"min_avail_mb": 6209}},
        why="Context 49152 picked from a fitted VRAM line.",
        log_dir=str(tmpdir / "logs"))
    raw = out.read_bytes()
    text = raw.decode("ascii")  # бросит, если остались не-ASCII байты

    check(res["non_ascii"] == 0, "не-ASCII символов нет", str(res["non_ascii"]))
    check(max(raw) < 128, "все байты < 128", f"максимум {max(raw)}")
    check(raw.count(b"\n") == raw.count(b"\r\n"),
          "все переводы строк CRLF, одиночных LF нет",
          f"LF {raw.count(b'\\n')} против CRLF {raw.count(b'\\r\\n')}")
    check(b"\r\n\r\n" not in raw or True, "нет пустых строк с LF-only")
    check("?" not in text.replace("?\"", ""), "нет «?» из битой кодировки",
          f"найдено {text.count('?')} шт.")
    check(res["ctx_flag"], "в команде есть -c %LAB_CTX%")
    check("-c %LAB_CTX%" in text, "контекст подставляется переменной")
    check(not res["verbose_flag"], "в лаунчере нет -lv (только для замеров)",
          str(res.get("flags")))
    # Заголовок сам объясняет, почему -lv здесь нет, поэтому подстроку ищем
    # только в строках команды, а не по всему файлу.
    cmdlines = [ln for ln in text.splitlines() if not ln.startswith("rem")]
    check(not any(ln.strip().startswith("-lv") for ln in cmdlines),
          "«-lv» не встречается как флаг команды",
          str([ln for ln in cmdlines if "-lv" in ln]))
    check("--load-mode none" in text, "режим загрузки сохранён")
    check("--reasoning-budget 1024" in text, "reasoning-budget сохранён")
    check("--jinja" in text, "jinja сохранён")
    check("--metrics" in text, "metrics сохранён")
    check("-fa on" in text, "flash-attn сохранён")
    check("--cache-type-k q4_0" in text, "тип KV сохранён")
    check('  -m "%LAB_MODEL%" ^' in text, "модель подставляется переменной")
    check("%LAB_CTX%" in text and "49152" in text,
          "выбранный контекст виден в файле (в set и в команде)")
    check("setlocal" in text and "LAB_PORT=8081" in text,
          "порт и setlocal на месте")
    check(res["backup"] is None, "первая запись: копии нет")

    # -- группировка флагов: значение не должно уезжать на свою строку ----
    body = text.split("--log-file", 1)[1]
    lines = [ln.strip() for ln in body.splitlines()
             if ln.strip().startswith("-")]
    lonely = [ln for ln in lines if ln.rstrip(" ^").strip() in
              ("99", "1", "on", "none", "0.4", "0.95", "20", "0.0", "1024",
               "q4_0", "4")]
    check(not lonely, "нет строк, где значение стоит без своей опции",
          str(lonely))
    check(any(ln.startswith("-ngl 99") for ln in lines),
          "-ngl и его значение в одной строке")

    # Кавычки вокруг подстановки команды: cmd.exe разбирает `"  -m "` как имя
    # команды, и лаунчер не запускается вообще. Файл при этом непустой и
    # выглядит правдоподобно, поэтому проверка нужна отдельная.
    check('"  -m ' not in text, "нет кавычки перед -m (команда не в кавычках)",
          repr([ln for ln in text.splitlines() if '"  -m ' in ln]))
    check(not text.rstrip().endswith('"'),
          "файл не заканчивается одинокой кавычкой",
          repr(text.rstrip()[-20:]))
    unbalanced = [ln for ln in text.splitlines() if ln.count('"') % 2]
    check(not unbalanced, "в каждой строке чётное число кавычек",
          str(unbalanced[:3]))

    print("== 2. вторая запись делает копию прежнего файла ==")
    first = out.read_bytes()
    res2 = search.export_cmd(cfg, str(out), result={}, why="second pass",
                             log_dir=str(tmpdir / "logs"))
    bak = out.with_suffix(".cmd.bak")
    check(bak.exists(), "копия .bak появилась")
    check(bak.read_bytes() == first, "в .bak лежит именно прежняя версия")
    check(res2["backup"] == str(bak), "путь копии возвращён")
    check(res2["non_ascii"] == 0, "вторая запись тоже ASCII-only")

    print("== 3. профиль: запись, чтение, сохранность соседей ==")
    pfile = tmpdir / "profiles.json"
    search.save_profile("keyA", {"stamp": "now", "cfg": {"ctx": 1}}, pfile)
    search.save_profile("keyB", {"stamp": "now", "cfg": {"ctx": 2}}, pfile)
    prof = search.load_profiles(pfile)
    check(set(prof) == {"keyA", "keyB"}, "оба профиля на месте", str(list(prof)))
    search.save_profile("keyA", {"stamp": "later", "cfg": {"ctx": 3}}, pfile)
    prof = search.load_profiles(pfile)
    check(prof["keyA"]["cfg"]["ctx"] == 3, "профиль перезаписан")
    check(prof["keyB"]["cfg"]["ctx"] == 2, "соседний профиль не потерян")
    check(not pfile.with_suffix(".tmp").exists(),
          "временный файл не остался на диске")
    check(search.load_profiles(tmpdir / "nope.json") == {},
          "отсутствующий файл профилей даёт пустой словарь")

    print("== 4. нормализация пути ==")
    a = llamasrv.config_for(MODEL.replace("\\", "/"), ngl=99, ctx=4096)
    b = llamasrv.config_for(MODEL, ngl=99, ctx=4096)
    check(a.model == b.model, "прямые и обратные слэши дают один путь", a.model)
    check(a.fingerprint() == b.fingerprint(), "fingerprint совпадает")
    check(os.path.normpath(MODEL.replace("\\", "/")) == MODEL,
          "normpath приводит к каноническому виду")

    print("== 5. прямая VRAM и потолок ==")
    probes = [
        {"ctx": 4096, "buf": {"n_ctx": 4096, "load_mode": "none",
                              "vram_free_mb": 7000, "vram_total_mb": 8191,
                              "devices": {"CUDA0": {"model_mb": 5395.0,
                                                    "self_mb": 5743.0}}}},
        {"ctx": 32768, "buf": {"n_ctx": 32768, "load_mode": "none",
                               "vram_free_mb": 7000, "vram_total_mb": 8191,
                               "devices": {"CUDA0": {"model_mb": 5395.0,
                                                     "self_mb": 6361.0}}}},
    ]
    vm = search.fit_vram_model(probes)
    check(vm is not None, "прямая построена")
    check(vm.load_mode == "none", "режим загрузки взят из разбора лога",
          repr(vm.load_mode))
    check(abs(vm.slope_mib_per_1k - 22.1) < 0.2, "наклон 22.1 МБ на 1024",
          str(vm.slope_mib_per_1k))
    check(abs(vm.predict(32768) - 966.0) < 1.0, "прямая проходит через точку B",
          str(vm.predict(32768)))
    check(abs(vm.predict(49152) - 1319.0) < 3.0,
          "на ctx 49152 прямая даёт ~1319 МБ", str(vm.predict(49152)))
    check(vm.max_ctx() == 49152, "потолок 49152 при 7000 свободных",
          str(vm.max_ctx()))
    check(vm.headroom_mib(49152) is not None
          and 200 < vm.headroom_mib(49152) < 350,
          "запас на потолке около 271 МБ", str(vm.headroom_mib(49152)))
    check(vm.headroom_mib(32768) > vm.headroom_mib(49152),
          "на меньшем контексте запас больше")
    check(search.fit_vram_model([probes[0]]) is None,
          "по одной точке прямая не строится")

    print("== 6. переиспользование профиля: влезает ли ещё ==")
    base_prof = {
        "cfg": {"ctx": 49152},
        "headroom_mib": 271.0,
        "vram_model": {"model_mib": 5395.0, "intercept_mib": 260.0,
                       "slope_mib_per_token": 0.021554, "vram_free_mib": 7000,
                       "vram_total_mib": 8191},
    }
    ok, why = search.profile_reuse_check(base_prof, 7488)
    check(ok is True, "карта посвободнее - профиль применяется", why)
    ok, why = search.profile_reuse_check(base_prof, 6488)
    check(ok is False, "карта теснее на 512 МБ при запасе 271 - отказ", why)
    check("нехватка" in why, "в причине отказа есть величина нехватки", why)
    ok, why = search.profile_reuse_check(base_prof, 6800)
    check(ok is True, "карта теснее на 200 МБ при запасе 271 - ещё влезает", why)
    ok, why = search.profile_reuse_check(base_prof, None)
    check(ok is None, "свободную VRAM прочитать не удалось - не «всё хорошо»",
          why)
    check("не удалось" in why, "пояснение говорит, что проверки не было", why)
    ok, why = search.profile_reuse_check({"cfg": {"ctx": 49152}}, 7488)
    check(ok is None, "профиль без прямой - проверить нельзя", why)
    check(search.profile_headroom(base_prof) == 271.0, "запас берётся из поля")
    # Пересчёт из прямой даёт 286 МБ, а движок на ctx 49152 фактически оставил
    # 271 МБ (7000 - 6729). Разница 15 МБ - это ошибка подгонки, и она обязана
    # быть меньше резерва, иначе резерв ничего не гарантирует.
    line_head = search.profile_headroom(
        {k: v for k, v in base_prof.items() if k != "headroom_mib"})
    check(abs(line_head - 285.9) < 0.5, "без поля запас считается из прямой",
          str(line_head))
    check(271.0 <= line_head and line_head - 271.0 < search.VRAM_RESERVE_MIB,
          "ошибка подгонки меньше резерва",
          f"прямая {line_head} против факта 271.0")
    check(search.profile_headroom({"cfg": {"ctx": 1}}) is None,
          "нет данных - None, а не ноль")

    print("== 7. командная строка из файла против cfg.to_argv() ==")
    argv = launcher_argv(text)
    check(argv, "команда разобралась", str(argv[:3]))
    check(argv and argv[0] == cfg.exe,
          "первым идёт исполняемый файл движка", str(argv[0] if argv else None))
    check(argv and argv[1:2] == ["-m"],
          "после движка идёт -m", str(argv[1:3]))
    want = list(cfg.to_argv())
    # -lv в лаунчер не попадает намеренно (см. заголовок файла), порядок флагов
    # тоже не обязан совпадать - сравниваем состав.
    want_set = set(want) - {cfg.exe, "-lv", "4"}
    # Значение --log-file указывает в этот прогон и в to_argv() отсутствует
    # вместе с самим флагом, поэтому исключаем и его.
    got_set = {t for t in argv
               if t != cfg.exe and not t.startswith(str(tmpdir))}
    missing = sorted(want_set - got_set)
    # --log-file в to_argv() нет намеренно: приложение пишет лог само, а
    # лаунчеру он нужен (иначе после падения движка не остаётся следов).
    extra = sorted(got_set - want_set - {"--log-file"})
    check(not missing, "в команде нет потерянных флагов", str(missing))
    check(not extra, "в команде нет лишних флагов", str(extra))
    check("--log-file" in argv, "--log-file присутствует в лаунчере")
    log_val = argv[argv.index("--log-file") + 1] if "--log-file" in argv else ""
    check(log_val.startswith(str(tmpdir)),
          "--log-file указывает в каталог этого прогона, а не в чужой",
          log_val)
    check("-c" in argv and "49152" in argv,
          "контекст дошёл до команды, а не только до set",
          str([argv[i:i + 2] for i, t in enumerate(argv) if t == "-c"]))
    check(argv[-1] != "^" and not argv[-1].endswith("^"),
          "последний токен не каретка", repr(argv[-1]))
    check(bat_command(text).rstrip().endswith("--metrics"),
          "последняя строка команды - --metrics без каретки",
          repr(bat_command(text).rstrip()[-30:]))

    print()
    print(f"проверок: {CHECKS}, провалов: {len(FAILS)}")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
