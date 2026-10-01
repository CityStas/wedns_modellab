"""UI: встроенный http.server плюс одна статическая страница.

Почему не Electron и не браузерный рантайм: приложение измеряет свободную
память, а любой из этих вариантов держит в памяти сотни мегабайт и попадает
в тот же бюджет, который мы измеряем. На машине с 16 ГБ это уже
существенная погрешность. http.server из стандартной библиотеки плюс один
HTML-файл стоят около 15 МБ и не мешают замерам.

Сервер слушает только 127.0.0.1 и проверяет заголовок Host: иначе страница
на любом сайте могла бы обратиться к API по localhost (DNS rebinding) и
запустить llama-server на чужой конфигурации.

Запуск:
    python -m modellab.gui            # http://127.0.0.1:8090
    python -m modellab.gui --open --port 8099
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import VERSION, lab as lab_mod

STATIC = Path(__file__).resolve().parent / "static"
ALLOWED_HOSTS = re.compile(r"^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$")


class Handler(BaseHTTPRequestHandler):
    server_version = f"modellab/{VERSION}"
    protocol_version = "HTTP/1.1"
    lab: lab_mod.Lab = None  # проставляется в serve()

    # -- утилиты -----------------------------------------------------------

    def log_message(self, fmt, *args):
        pass  # консоль занята прогрессом прогонов, а не логом доступа

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg: str, code: int = 400) -> None:
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {}

    def _guard(self) -> bool:
        host = self.headers.get("Host") or ""
        if not ALLOWED_HOSTS.match(host):
            self._err("запросы принимаются только с localhost", 403)
            return False
        return True

    # -- маршруты ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        u = urllib.parse.urlparse(self.path)
        path, q = u.path, urllib.parse.parse_qs(u.query)
        try:
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if path.startswith("/static/"):
                return self._file(STATIC / path[len("/static/"):])
            if path == "/api/system":
                return self._json(self.lab.system())
            if path == "/api/models":
                return self._json(self.lab.models(refresh="refresh" in q))
            if path == "/api/state":
                return self._json(self.lab.state())
            if path == "/api/log":
                tail = int((q.get("tail") or ["200"])[0])
                return self._json({"log": self.lab.log_tail(tail)})
            if path == "/api/job":
                jid = (q.get("id") or [""])[0]
                j = self.lab.job(jid)
                return self._json(j) if j else self._err("нет такой задачи", 404)
            if path == "/api/version":
                return self._json({"version": VERSION, "lab": True})
            return self._err("нет такого маршрута", 404)
        except Exception as e:  # noqa: BLE001
            return self._err(f"{type(e).__name__}: {e}", 500)

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = urllib.parse.urlparse(self.path).path
        body = self._body()
        try:
            if path == "/api/start":
                return self._json(self.lab.start(body).to_dict())
            if path == "/api/stop":
                return self._json(self.lab.stop().to_dict())
            if path == "/api/measure":
                return self._json(self.lab.measure(body).to_dict())
            if path == "/api/search":
                return self._json(self.lab.search(body).to_dict())
            if path == "/api/ab":
                return self._json(self.lab.ab(body).to_dict())
            if path == "/api/bench":
                return self._json(self.lab.bench(body).to_dict())
            return self._err("нет такого маршрута", 404)
        except RuntimeError as e:  # занято
            return self._err(str(e), 409)
        except Exception as e:  # noqa: BLE001
            return self._err(f"{type(e).__name__}: {e}", 500)

    def _file(self, p: Path) -> None:
        try:
            data = p.read_bytes()
        except OSError:
            return self._err("файл не найден", 404)
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix in (".html", ".js", ".css"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def serve(port: int = 8090, host: str = "127.0.0.1", open_browser: bool = False,
          models_root: str | None = None) -> None:
    lab = lab_mod.Lab(**({"models_root": models_root} if models_root else {}))
    Handler.lab = lab
    # Прогрев кэшей до старта HTTP: холодное определение железа и чтение
    # заголовков всех GGUF занимают секунды, а страница опрашивает состояние
    # раз в секунду и первые тики выглядела бы зависшей.
    lab.models()
    sysinfo = lab.system()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"modellab {VERSION} - {url}")
    print(f"модели: {lab.models_root} ({len(lab.models())} шт.)")
    print(f"бинарник: {(sysinfo.get('exe') or {}).get('path')}")
    print("Ctrl+C для выхода (сервер будет остановлен)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nостанавливаю...")
    finally:
        lab.shutdown()
        httpd.shutdown()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="modellab.gui",
                                 description="Тестер-оптимизатор локальных моделей")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="открыть браузер")
    ap.add_argument("--models", default=None, help="каталог с GGUF")
    a = ap.parse_args(argv)
    serve(a.port, a.host, a.open, a.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
