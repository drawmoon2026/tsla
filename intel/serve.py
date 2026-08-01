"""局域网只读静态服务 — 手机看仪表盘.

安全边界（设计即约束）：
- 只服务 data/intel/ 目录下、后缀在白名单（.html/.json）内的文件——
  sentinel.sqlite / *.csv / .env 等一律 403，不存在的路径 404；
- 路径解析后必须仍落在 data/intel/ 之内（resolve 后前缀校验，防 ../ 穿越）；
- 隐藏文件（任一路径段以 . 开头）一律 403；
- 只读 GET/HEAD，无上传无执行；绑定 0.0.0.0 仅为同一局域网设备可达，
  不做公网映射。

用法：
    .venv/bin/python -m intel.serve            # 0.0.0.0:8765，/ 重定向 dashboard.html
    .venv/bin/python -m intel.serve --port 9000
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "intel"
DEFAULT_PORT = 8765
ALLOWED_SUFFIXES = {".html", ".json"}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class IntelHandler(BaseHTTPRequestHandler):
    server_version = "IntelServe/1.0"

    def _resolve(self, raw_path: str) -> tuple[int, Path | None]:
        """URL path → (状态码, 文件路径)。403=越权/非白名单，404=不存在，200=可服务."""
        rel = unquote(urlsplit(raw_path).path).lstrip("/")
        parts = Path(rel).parts
        if not parts or any(p.startswith(".") or p == ".." for p in parts):
            return 403, None
        target = (DATA_DIR / rel).resolve()
        if not target.is_relative_to(DATA_DIR):  # 符号链接/穿越兜底
            return 403, None
        if target.suffix.lower() not in ALLOWED_SUFFIXES:
            return 403, None
        if not target.is_file():
            return 404, None
        return 200, target

    def _respond(self, head_only: bool) -> None:
        path = urlsplit(self.path).path
        if path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/dashboard.html")
            self.end_headers()
            return
        code, target = self._resolve(self.path)
        if code != 200:
            msg = b"403 Forbidden\n" if code == 403 else b"404 Not Found\n"
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            if not head_only:
                self.wfile.write(msg)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[target.suffix.lower()])
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._respond(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond(head_only=True)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[serve] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="data/intel 局域网只读静态服务")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.bind, args.port), IntelHandler)
    print(f"[serve] {args.bind}:{args.port} → {DATA_DIR} "
          f"(白名单后缀 {sorted(ALLOWED_SUFFIXES)})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
