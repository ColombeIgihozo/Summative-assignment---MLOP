"""Round-robin HTTP gateway used when Docker cannot start.

Docker Desktop on this machine cannot boot a Linux engine until virtualization
is enabled. This stdlib proxy spreads Locust traffic across one or more local
API processes so the same flood test can still compare replica counts.
"""

from __future__ import annotations

import itertools
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request

BACKENDS = [item.strip() for item in os.environ.get("BACKENDS", "http://127.0.0.1:8001").split(",") if item.strip()]
PORTS = itertools.cycle(BACKENDS)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else None
        target = next(PORTS) + self.path
        headers = {key: value for key, value in self.headers.items() if key.lower() not in {"host", "content-length"}}
        request = urllib.request.Request(target, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
                status = response.status
                response_headers = response.headers
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            response_headers = exc.headers
        except Exception as exc:
            payload = str(exc).encode()
            status = 502
            response_headers = {}
        self.send_response(status)
        for key, value in response_headers.items():
            if key.lower() in {"transfer-encoding", "connection", "content-length"}:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Gateway on {port} -> {', '.join(BACKENDS)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
