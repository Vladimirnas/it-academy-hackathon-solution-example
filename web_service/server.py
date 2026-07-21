

import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.getenv("PORT", "80"))
ROOT = os.getenv("STATIC_ROOT", "/app/static")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    handler = partial(Handler, directory=ROOT)
    print(f"Клиент на порту {PORT}, статика из {ROOT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), handler).serve_forever()
