# HOW: запуск локальных моделей через Ollama

Дата: 2026-09-20 · Ollama **0.34.2** · Windows 11 · RTX 2060 SUPER 8 ГБ (GPU-оффлоад частичный)

Полная техническая документация по установке и хранилищу — `E:\Programs\LocalLLM\ollama\README.md`.
Этот файл — короткий «как запустить и куда нажимать» для трёх потребителей: WorkBuddy, Cursor, pi.

---

## 0. Карта: что где лежит

| Что | Путь |
|---|---|
| Бинарник сервера | `E:\Programs\LocalLLM\ollama\ollama.exe` |
| Лончер (двойной клик) | `E:\Programs\LocalLLM\ollama\start-ollama.cmd` |
| Автозапуск при входе | `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Ollama.cmd` |
| Хранилище (манифесты + блобы) | `E:\Programs\LocalLLM\cache\.ollama\models` |
| Исходные `.gguf` (33 ГБ) | `E:\Programs\LocalLLM\Models` |
| `C:\Users\tio\.ollama` | junction → `E:\Programs\LocalLLM\cache\.ollama` (на `C:` не пишется ничего) |
| Modelfiles (только справка!) | `E:\Programs\LocalLLM\ollama\Modelfiles\` |
| Регистрация GGUF без копирования | `E:\Programs\LocalLLM\ollama\register-gguf.py` |
| Конфиг моделей WorkBuddy | `C:\Users\tio\.workbuddy-ai\models.json` |
| Конфиг моделей pi | `C:\Users\tio\.pi\agent\models.json` |
| Конфиг Cline (внутри Cursor) | `C:\Users\tio\.cline\data\settings\providers.json` |

`E:\Programs\LocalLLM\ollama` уже прописан в `PATH` (HKCU) — команда `ollama` работает в любом новом терминале.
Автозапуск **не настроен** (папка Startup пустая) — после перезагрузки сервер надо поднять руками, см. §2, шаг 1.

---

## 1. Настройки: у всех ли моделей то же, что у `Ornith-1.5-9B-CRACK`?

**Короткий ответ: у всех восьми — да по `temperature 0.4`, но `num_ctx` разный, а `q4_0` — это вообще не настройка модели.**

Три разных уровня, не путать:

| Настройка | Значение | Где живёт | Область действия |
|---|---|---|---|
| `temperature` | `0.4` | params-блоб модели | у всех 8 моделей одинаковая, но **записана отдельно по группам** |
| `num_ctx` | `98304` (9B) / `32768` (мелкие) | params-блоб модели | по группам |
| KV-кеш `q4_0` вместо `q8_0` | `q4_0` | переменная окружения `OLLAMA_KV_CACHE_TYPE` | **сервер целиком, все модели сразу** |
| Flash attention | `1` | `OLLAMA_FLASH_ATTENTION` | сервер целиком (обязателен для квантованного KV) |

Фактическое состояние (проверено `POST /api/show` по каждой модели):

| Модель | num_ctx | temperature | capabilities | VRAM при загрузке |
|---|---|---|---|---|
| `ornith-1.5-9b-crack` | 98304 | 0.4 | tools, thinking, completion | ~6.4 ГБ |
| `ornith-1.5-9b` | 98304 | 0.4 | tools, thinking, completion | ~6.4 ГБ |
| `qwen3.5-9b` | 98304 | 0.4 | tools, thinking, completion | ~6.4 ГБ |
| `qwen3.5-9b-deepseek-v4-flash` | 98304 | 0.4 | tools, thinking, completion | ~6.4 ГБ |
| `qwen3.5-9b-uncensored-hauhaucs` | 98304 | 0.4 | tools, thinking, completion | ~6.4 ГБ |
| `omnicoder-9b` | 98304 | 0.4 | tools, thinking, completion | ~6.4 ГБ |
| `qwen3.5-0.8b` | 32768 | 0.4 | tools, thinking, completion | 1185 МиБ |
| `qwen2-0.5b-instruct` | 32768 | 0.4 | **только completion** | 999 МиБ |

> У `qwen2-0.5b-instruct` нет ни tools, ни thinking. В агентских клиентах (pi, Cline, Cursor Agent) он бесполезен — только чат.

---

### 1.1. Где менять `q4_0` / flash attention / прочее серверное

Всё это — переменные окружения пользователя. Реестр: `HKEY_CURRENT_USER\Environment`.

Текущий набор (проверено чтением реестра):

```
OLLAMA_MODELS           = E:\Programs\LocalLLM\cache\.ollama\models
OLLAMA_HOST             = 127.0.0.1:11434
OLLAMA_CONTEXT_LENGTH   = 98304
OLLAMA_KV_CACHE_TYPE    = q4_0        <- вот он, «q4_0 вместо q8_0»
OLLAMA_FLASH_ATTENTION  = 1
OLLAMA_KEEP_ALIVE       = 60m
OLLAMA_NUM_PARALLEL     = 1
OLLAMA_MAX_LOADED_MODELS= 1
```

**Поменять руками — два способа.**

GUI: `Win+R` → `sysdm.cpl` → вкладка «Дополнительно» → «Переменные среды…» → блок **«Переменные среды пользователя»** (не системные!) → выбрать → «Изменить».

Командная строка (обычный cmd/PowerShell, права админа не нужны):

```bat
setx OLLAMA_KV_CACHE_TYPE q8_0
```

⚠️ **Никогда не делай `setx PATH ...`** — `setx` режет значение до 1024 символов и калечит `PATH`.

⚠️ После правки **обязателен перезапуск сервера** (закрыть окно с `ollama serve` / снять процесс и запустить заново). Уже запущенный сервер переменные не перечитывает.

⚠️ `OLLAMA_CONTEXT_LENGTH=98304` у нас **фактически не работает**: у всех моделей `num_ctx` запечён в манифест, а манифест перебивает переменную. Это видно в `/api/ps` → `context_length`. Переменная нужна только как страховка для моделей без запечённого `num_ctx`.

Проверить, что сервер поднялся с нужным KV-кешем, напрямую нельзя (ни `/api/ps`, ни лог этого не печатают). Косвенный признак — `size` в `/api/ps`: с `q4_0` 9B-модель на 98304 даёт ~6.4 ГБ, с `f16` было бы заметно больше.

### 1.2. Где менять `num_ctx` и `temperature` (per-model)

Они запечены в **params-блоб** — обычный JSON-файл в хранилище:

| Блоб | Размер | Содержимое | На какие модели влияет |
|---|---|---|---|
| `blobs\sha256-ab17cf4e4e242037dd102025052cdda3c1599670831c71d3ed35dde1384a276f` | 35 Б | `{"num_ctx":98304,"temperature":0.4}` | **все шесть 9B сразу** |
| `blobs\sha256-7096934481df62e78543bc64ff04590532e8279038d78049099e4a7ac5694f49` | 35 Б | `{"num_ctx":32768,"temperature":0.4}` | `qwen3.5-0.8b`, `qwen2-0.5b-instruct` |

Полный путь: `E:\Programs\LocalLLM\cache\.ollama\models\blobs\<имя файла>`.

**Ключевой момент:** имя блоба — это sha256 его содержимого. Просто «открыть и переписать» файл нельзя, Ollama ищет блоб строго по `sha256-<digest>` из манифеста. Порядок правки:

1. Записать новое содержимое: компактный JSON, без пробелов и без перевода строки на конце (порядок ключей и размер не важны — Ollama читает JSON, а `size` считается из фактического файла).
2. Посчитать sha256 нового содержимого.
3. Положить его как новый файл `blobs\sha256-<новый digest>`.
4. В манифестах `manifests\registry.ollama.ai\library\<имя модели>\latest` заменить в `layers[]` элемент с `"mediaType":"application/vnd.ollama.image.params"`: поля `digest` и `size`.
5. Перезапустить сервер.
6. Проверить: `curl -s http://127.0.0.1:11434/api/show -d "{\"model\":\"qwen3.5-9b\"}"` → поле `parameters` должно показать новые `num_ctx` / `temperature`.
7. Старый блоб станет «сиротой» и будет удалён при следующем старте сервера (`msg="total unused blobs removed: N"` в логе) — это нормально, не пугаться.

Чтобы поменять **одну** модель из группы: сделать для неё свой params-блоб по шагам 1–3 и поправить только её манифест. Остальные продолжат ссылаться на общий.

⚠️ **`E:\Programs\LocalLLM\ollama\Modelfiles\*.Modelfile` — это только документация параметров.** Не запускай `ollama create -f <Modelfile>`: он **копирует** GGUF в хранилище, на 8 моделях это 37 ГБ дубля. Правильный путь регистрации — `register-gguf.py` (жёсткая ссылка + манифест руками).

### 1.3. Где менять параметры со стороны клиентов

| Клиент | Файл | Что там |
|---|---|---|
| WorkBuddy | `C:\Users\tio\.workbuddy-ai\models.json` | `temperature: 0.4`, `maxInputTokens: 98304`, `maxOutputTokens: 8192`, флаги `supportsToolCall` / `supportsReasoning` |
| pi | `C:\Users\tio\.pi\agent\models.json` | `contextWindow`, `maxTokens`, `reasoning`, `samplingParams` (свободный объект, уходит в тело запроса как есть) |
| Cursor (BYOK) | UI: Settings → Models | контекст и max output задаются на стороне Cursor, файла нет |
| Cline | UI + `C:\Users\tio\.cline\data\settings\providers.json` | `baseUrl`, `model`, `apiKey` |

### 1.4. Перебить настройки на один запрос

| Что | Нативный `/api/chat` | OpenAI `/v1/chat/completions` |
|---|---|---|
| `num_ctx` | ✅ `options.num_ctx` — **проверено**: 8192 → в `/api/ps` `context_length: 8192` | ❌ игнорируется — **проверено**: остался `32768` из манифеста |
| `temperature` | ✅ `options.temperature` | ✅ поле `temperature` (стандартное для протокола, перебивает запечённое; отдельным замером не проверял) |

Вывод: через OpenAI-эндпоинт (WorkBuddy, Cursor, pi, Cline) контекст модели меняется **только** правкой манифеста.

### 1.5. Размер контекста и скорость: резать бесполезно

Замер на `qwen3.5-9b-deepseek-v4-flash`, каждое значение — с полной перезагрузкой модели:

| `num_ctx` | VRAM | генерация (прогретая) | prompt eval, 22 510 токенов |
|---|---|---|---|
| 98304 | 5.87 ГиБ | 31.9 ток/с | 943 ток/с |
| 80000 | 5.87 ГиБ | 30.6 ток/с | 908 ток/с |
| 75000 | 5.80 ГиБ | 30.3 ток/с | 910 ток/с |
| 65536 | 5.68 ГиБ | 30.4 ток/с | 909 ток/с |

**Уменьшение контекста не даёт скорости.** 98304 → 65536 освобождает всего **0.19 ГиБ** VRAM и ноль токенов в секунду — всё в пределах шума.

Причина: `qwen35` — гибридная архитектура, полное внимание только у 8 слоёв из 32 (`full_attention_interval=4`), остальные — SSM с фиксированным состоянием. KV-кеш поэтому крошечный: 33k токенов контекста стоят ~0.2 ГиБ. Режешь окно — режешь не то, что занимает память. Отдельно проверено: при 65536 раскладка становится 100% GPU, и скорость всё равно не растёт.

**Что реально влияет на скорость:**

1. **Первый ответ после загрузки — вдвое медленнее.** 14 ток/с против 30+ на втором прогоне: прогрев CUDA-графов. Поэтому `OLLAMA_KEEP_ALIVE=60m` важнее любой экономии контекста.
2. **Размер промпта — главный счёт.** 22 510 токенов промпта = 24 с. Платишь за фактические токены, а не за размер окна.
3. **Разный `num_ctx` у разных клиентов = перезагрузка на каждом переключении (~100 с).** У Cline стоит 80000, у WorkBuddy 98304 — каждое переключение между ними перезагружает модель. Контексты у клиентов стоит выровнять.

**Кто вообще может задать `num_ctx`:**

| Способ | Работает? |
|---|---|
| нативный `/api/chat` → `options.num_ctx` | ✅ да |
| OpenAI `/v1/chat/completions` → `num_ctx` в теле | ❌ игнорируется |
| OpenAI `/v1/chat/completions` → `options.num_ctx` в теле | ❌ тоже игнорируется (проверено) |

Отсюда: в Cline два разных пути к Ollama — провайдер **Ollama** (нативный, умеет задавать контекст, у тебя там 80000) и **OpenAI Compatible** (не умеет). Настройка живёт в `C:\Users\tio\.cline\data\globalState.json` → `ollamaApiOptionsCtxNum`.

---

## 2. Запуск: пошагово

### Шаг 1. Поднять сервер Ollama

**Основной способ** — двойной клик по `E:\Programs\LocalLLM\ollama\start-ollama.cmd`.
Скрипт сам выставляет все `OLLAMA_*`, печатает путь к моделям, контекст, KV-кеш и хост, а в конце оставляет окно открытым (`pause`), чтобы было видно лог и код выхода.

> Честно про проверку: сам `.cmd` проверен только по содержимому (в песочнице ассистента batch-файлы не исполняются). Живой прогон сервера делался через `ollama serve` с теми же переменными — он отработал штатно.

**Резервный способ** (проверен живым прогоном) — в любом терминале:

```
ollama serve
```

Переменные при этом берутся из реестра (`HKCU\Environment`), а там всё уже прописано — так что результат тот же.

**Автозапуск — настроен** (2026-09-20). Файл `C:\Users\tio\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\Ollama.cmd` поднимает сервер при входе в систему:

```bat
start "ollama" /min "E:\Programs\LocalLLM\ollama\ollama.exe" serve
```

Он сам выставляет все `OLLAMA_*`, запускает сервер **отвязанным** (`start`) и свёрнутым (`/min`). Важная деталь: **в нём нет `pause`** — в `start-ollama.cmd` он есть, но для автозапуска это смертельно: скрипт входа зависнет навсегда, ожидая нажатия клавиши. Поэтому для Startup используется отдельный файл, а не ярлык на `start-ollama.cmd`.

Убрать автозапуск — просто удалить этот файл. Минус: свёрнутое окно консоли будет висеть в панели задач.

> ⚠️ **Сервер, поднятый ассистентом из своей сессии, живёт только до конца этой сессии.** Его процесс висит на фоновой задаче, и когда задача закрывается — `ollama.exe` умирает вместе с ней. Диагностика: `ps -W | grep -i ollama` пусто и `curl --noproxy '*' http://127.0.0.1:11434/api/version` даёт код 7 (connection refused). **Поэтому поднимать сервер надо самому** — двойным кликом по `start-ollama.cmd` или `ollama serve` в своём терминале. Тогда он ни от чьей сессии не зависит.

⚠️ Второй сервер на тот же порт не встанет: `ollama serve` завершится с ошибкой, порт 11434 уже занят. Сначала закрыть предыдущий.

### Шаг 2. Убедиться, что сервер жив

```
curl -s http://127.0.0.1:11434/api/version
→ {"version":"0.34.2"}

ollama list          # список зарегистрированных моделей
ollama ps            # что сейчас загружено в память (пусто = ничего не загружено)
```

Если `curl` не отвечает — сервер не поднялся; смотреть окно с логом.

### Шаг 3. Первый запрос = прогрев (это долго, и это нормально)

Модель не держится в памяти постоянно. `OLLAMA_MAX_LOADED_MODELS=1` — в памяти живёт только одна модель, при переключении предыдущая выгружается.

| Действие | Время |
|---|---|
| Загрузка 9B-модели | **83–100 с** |
| Загрузка мелкой (0.8B / 0.5B) | 50–61 с |
| Ответ при уже загруженной модели | 1.5 с (горячий) / 126 с (с загрузкой) |
| Скорость генерации, 9B | 37.6 ток/с на свежей загрузке, 24–26 ток/с на повторных |
| Обработка промпта | ~615 ток/с → промпт WorkBuddy на 41k токенов ≈ 67 с |
| Держится в памяти после запроса | 60 минут (`OLLAMA_KEEP_ALIVE=60m`) |

⚠️ Пока идёт загрузка, клиент просто «думает» — это не зависание. Не жми Cancel и не шли второй запрос.

### Шаг 4. WorkBuddy

Ничего запускать не надо — WorkBuddy сам читает `C:\Users\tio\.workbuddy-ai\models.json` и подмешивает модели в свой кэш конфига. В списке моделей они выглядят как:

| ID в UI | Модель Ollama |
|---|---|
| `custom-local:ornith-1.5-9b-crack` | `ornith-1.5-9b-crack` |
| `custom-local:ornith-1.5-9b` | `ornith-1.5-9b` |
| `custom-local:qwen3.5-9b` | `qwen3.5-9b` |
| `custom-local:qwen3.5-9b-deepseek-v4-flash` | `qwen3.5-9b-deepseek-v4-flash` |
| `custom-local:qwen3.5-9b-uncensored-hauhaucs` | `qwen3.5-9b-uncensored-hauhaucs` |
| `custom-local:omnicoder-9b` | `omnicoder-9b` |
| `custom-local:qwen3.5-0.8b` | `qwen3.5-0.8b` |
| `custom-local:qwen2-0.5b-instruct` | `qwen2-0.5b-instruct` |

Перезапуск WorkBuddy не нужен — кэш `acc-product-config-v3.json` перезаписывается сам.

Первый чат: выбрать модель → отправить сообщение → **ждать до ~2 минут** (загрузка + обработка промпта).

⚠️ **Пустой ответ = не баг.** `qwen3.5-*` — reasoning-модели: Ollama кладёт «размышления» в отдельное поле `message.reasoning`, а не в `content`. Если `max_tokens` маленький, весь бюджет уходит в reasoning и `content` приходит пустым. Лечится увеличением вывода или отключением размышлений (`reasoning_effort: "none"`).

### Шаг 5. Cursor

**Вариант А — встроенный Cursor через BYOK.** `Ctrl+,` → вкладка **Models**:

1. **OpenAI API Key** — вписать любую непустую строку (`ollama`). Ключ Ollama не проверяет, но поле не должно быть пустым.
   *(Судя по `state.vscdb`, у тебя там уже что-то сохранено — можно оставить как есть.)*
2. Включить **Override OpenAI Base URL** и вписать ровно:
   ```
   http://127.0.0.1:11434/v1
   ```
   Без `/chat/completions` на конце и **без слэша в конце** — иначе Cursor склеит путь неверно и Verify упадёт.
3. Нажать **Verify**.
4. Нажать **+ Add Model** и вписать ID **ровно как в Ollama**: `ornith-1.5-9b-crack`. Список моделей Cursor не подтягивает сам, ID пишется руками.

Что важно знать про BYOK в Cursor (версия 3.14.7):

- ✅ Chat, Composer, основной цикл Agent идут на локальную модель.
- ❌ **Tab-автокомплит и «Apply from Chat» BYOK не подчиняются** — всегда на серверах Cursor.
- ⚠️ Часть внутренних подзадач Agent всё равно уходит на бэкенд Cursor (известный баг, официально не закрыт).
- ⚠️ Локальная модель 9B с контекстом 98304 на обработке промпта Cursor будет ощутимо тормозить: ~615 ток/с на промпт-эвал.
- Если после включения Override «поехали» официальные модели — выключить Override, перезапустить Cursor.

**Вариант Б — расширение Cline (уже установлено, `saoudrizwan.claude-dev` 4.1.19).** Это надёжнее: Cline полностью ходит в тот эндпоинт, который ему указали, ничего не утекает на бэкенд.

Настройки Cline → **API Provider: OpenAI Compatible**:

| Поле | Значение |
|---|---|
| Base URL | `http://127.0.0.1:11434/v1` |
| API Key | `ollama` (любая строка) |
| Model ID | `ornith-1.5-9b-crack` |
| Context Window | `98304` (для мелких — `32768`) |
| Max Output Tokens | `8192` |
| Supports Images | off |
| Supports Computer Use | off |

Сейчас в `providers.json` у Cline прописан `https://302.ai` — при переключении на локальную модель просто подменить три поля.

### Шаг 6. pi (Pi coding agent)

pi — `@earendil-works/pi-coding-agent` v0.85.1, конфиг `C:\Users\tio\.pi\agent\`.

**Правится один файл — `C:\Users\tio\.pi\agent\models.json`.** Провайдер `ollama` там **уже прописан** (2026-09-20), LM Studio оставлен рядом. Бэкап исходного: `C:\Users\tio\.pi\agent\models.json.bak-ollama-20260920-013256`. Итоговое содержимое:

```json
{
  "providers": {
    "lmstudio": { "...как было..." },
    "ollama": {
      "baseUrl": "http://127.0.0.1:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "compat": {
        "supportsDeveloperRole": false
      },
      "models": [
        { "id": "ornith-1.5-9b-crack",            "name": "Ornith 1.5 9B CRACK (ollama)",            "reasoning": true,  "input": ["text"], "contextWindow": 98304, "maxTokens": 8192 },
        { "id": "ornith-1.5-9b",                  "name": "Ornith 1.5 9B (ollama)",                  "reasoning": true,  "input": ["text"], "contextWindow": 98304, "maxTokens": 8192 },
        { "id": "qwen3.5-9b",                     "name": "Qwen3.5 9B (ollama)",                     "reasoning": true,  "input": ["text"], "contextWindow": 98304, "maxTokens": 8192 },
        { "id": "qwen3.5-9b-deepseek-v4-flash",   "name": "Qwen3.5 9B DeepSeek V4 Flash (ollama)",   "reasoning": true,  "input": ["text"], "contextWindow": 98304, "maxTokens": 8192 },
        { "id": "qwen3.5-9b-uncensored-hauhaucs", "name": "Qwen3.5 9B Uncensored HauhauCS (ollama)", "reasoning": true,  "input": ["text"], "contextWindow": 98304, "maxTokens": 8192 },
        { "id": "omnicoder-9b",                   "name": "OmniCoder 9B (ollama)",                   "reasoning": true,  "input": ["text"], "contextWindow": 98304, "maxTokens": 8192 },
        { "id": "qwen3.5-0.8b",                   "name": "Qwen3.5 0.8B (ollama)",                   "reasoning": true,  "input": ["text"], "contextWindow": 32768, "maxTokens": 4096 },
        { "id": "qwen2-0.5b-instruct",            "name": "Qwen2 0.5B Instruct (ollama)",            "reasoning": false, "input": ["text"], "contextWindow": 32768, "maxTokens": 4096 }
      ]
    }
  }
}
```

Почему именно так:

- `contextWindow` **обязательно** указывать: дефолт pi — `128000`, это больше наших 98304, и pi будет считать, что влезает больше, чем реально.
- `reasoning: false` для `qwen2-0.5b-instruct` — у неё нет ни tools, ни thinking.
- `compat.supportsDeveloperRole: false` — pi для reasoning-моделей шлёт роль `developer`; Ollama её принимает (проверено), но по документации pi для Ollama рекомендуется именно `false`. Вреда нет.
- `samplingParams` (если нужно перебить `temperature`) — свободный объект, уходит в тело запроса как есть.

**Проверка:**

```
pi --list-models ollama
```
Фактический вывод (проверено на этом конфиге):
```
provider  model                           context  max-out  thinking  images
ollama    omnicoder-9b                    98.3K    8.2K     yes       no
ollama    ornith-1.5-9b                   98.3K    8.2K     yes       no
ollama    ornith-1.5-9b-crack             98.3K    8.2K     yes       no
ollama    qwen2-0.5b-instruct             32.8K    4.1K     no        no
ollama    qwen3.5-0.8b                    32.8K    4.1K     yes       no
ollama    qwen3.5-9b                      98.3K    8.2K     yes       no
ollama    qwen3.5-9b-deepseek-v4-flash    98.3K    8.2K     yes       no
ollama    qwen3.5-9b-uncensored-hauhaucs  98.3K    8.2K     yes       no
```
`pi --list-models` без аргумента покажет ещё и две модели LM Studio (`gemma-4-12B-it-Q4_K_M`, `omnicoder-9b-q4_k_m`) — они на месте.

**Запуск:**

```
pi --provider ollama --model ornith-1.5-9b-crack
```

Файл `models.json` перечитывается при каждом открытии `/model` — перезапускать pi не нужно.
Модель по умолчанию: в пикере моделей `Ctrl+L` навести на нужную и нажать `Ctrl+S`.

**Живой прогон (проверено):**
- `pi -p --provider ollama --model qwen3.5-0.8b "Reply with exactly one word: PONG"` → `PONG`
- `pi -p --provider ollama --model ornith-1.5-9b-crack "Use the bash tool to list files…"` → модель вызвала инструмент `bash` и вернула результат. Полный цикл 1 м 56 с (включая ~100 с загрузки модели).

### Шаг 7. Любой другой OpenAI-совместимый клиент

| Параметр | Значение |
|---|---|
| Base URL | `http://127.0.0.1:11434/v1` |
| Endpoint | `http://127.0.0.1:11434/v1/chat/completions` |
| API Key | любая непустая строка |
| Model ID | имя из `ollama list`, например `ornith-1.5-9b-crack` |
| Список моделей | `GET http://127.0.0.1:11434/v1/models` |

`localhost` лучше не использовать — если клиент живёт в WSL или контейнере, нужен `127.0.0.1` или LAN-IP, а `OLLAMA_HOST` придётся выставить на `0.0.0.0:11434`.

---

## 3. Чек-лист «всё завелось»

```
[ ] ollama serve запущен (окно с логом открыто)
[ ] curl http://127.0.0.1:11434/api/version  ->  {"version":"0.34.2"}
[ ] ollama list  ->  8 моделей
[ ] в клиенте прописан base URL http://127.0.0.1:11434/v1 и ID модели
[ ] первый запрос подождали (до ~2 минут) — модель грузится в память
[ ] ollama ps  ->  видно загруженную модель и context_length 98304 (или 32768)
```

---

## 4. Грабли (коротко)

- **`ollama create -f Modelfile` копирует веса.** 8 моделей = 37 ГБ дубля. Регистрировать только через `register-gguf.py`.
- **Правка params-блоба задевает всю группу.** Один блоб `ab17cf4e…` делится на шесть 9B-моделей.
- **Смена блоба = смена его имени.** Имя блоба — это его sha256; переписывание файла на месте не сработает.
- **`num_ctx` через OpenAI-эндпоинт не перебивается.** Только правкой манифеста.
- **`OLLAMA_CONTEXT_LENGTH` бессилен**, пока `num_ctx` запечён в манифест.
- **Одна модель в памяти.** `MAX_LOADED_MODELS=1` → переключение модели = ~100 с на загрузку.
- **Один параллельный запрос.** `NUM_PARALLEL=1` → одновременные запросы встают в очередь, а не идут параллельно.
- **Чужой запрос убивает префикс-кеш.** Кеш промпта один на модель; любой посторонний запрос к той же модели сбрасывает его, и следующий агентский ход пересчитывает промпт целиком.
- **`OLLAMA_MODELS` в системном реестре устарел** — там прописан `E:\Programs\LocalLLM\OllamaModels` (такой папки нет). Не мешает, но чистится только с правами администратора.
- **Свободное место.** Удаления в Windows уходят в корзину `E:\$RECYCLE.BIN`, а `pagefile.sys` на `E:` занимает 16 ГБ. Если место «пропало» — смотреть сначала туда.
