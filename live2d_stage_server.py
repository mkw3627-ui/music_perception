"""Serve the plugin's Live2D stage on 127.0.0.1 with a one-time token.

The host does not expose plugin static files by URL, so the stage is served by
a small stdlib-only HTTP server owned by this plugin:

* binds to 127.0.0.1 only (never the LAN),
* every path must start with a random token, so a stray tab or another local
  process cannot enumerate the folder,
* only the ``live2d/`` directory is reachable,
* the server stops itself after TTL seconds and when the plugin unloads.
"""
from __future__ import annotations

import json
import mimetypes
import secrets
import socketserver
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit

_LOCK = threading.Lock()
_SERVERS: dict[str, dict] = {}
DEFAULT_TTL_S = 1800


class _Handler(SimpleHTTPRequestHandler):
    token = ""
    root: Path = Path(".")
    state_provider = None

    def translate_path(self, path: str) -> str:
        raw = str(path or "").split("?", 1)[0].split("#", 1)[0]
        prefix = "/" + self.token
        if not raw.startswith(prefix):
            return str(self.root / "__token_required__")
        rest = urllib.parse.unquote(raw[len(prefix):]).lstrip("/") or "stage.html"
        target = (self.root / rest).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError:
            return str(self.root / "__forbidden__")
        return str(target)

    def do_GET(self):  # noqa: N802 - stdlib handler API
        raw = str(self.path or "").split("?", 1)[0]
        if raw == "/" + self.token + "/state.json":
            try:
                payload = self.state_provider() if self.state_provider else {}
            except Exception as exc:  # noqa: BLE001 - 状态失败不能中断页面
                payload = {"ok": False, "error": type(exc).__name__}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *_args):  # keep the plugin log clean
        return


def start_stage_server(live2d_dir: Path, *, ttl: int = DEFAULT_TTL_S,
                       state_provider=None) -> dict:
    live2d_dir = Path(live2d_dir).resolve()
    if not (live2d_dir / "stage.html").is_file():
        raise FileNotFoundError(f"缺少预览页：{live2d_dir / 'stage.html'}")
    with _LOCK:
        for token, info in list(_SERVERS.items()):
            if info["expires"] > time.time() and info["thread"].is_alive():
                return {"ok": True, "reused": True, "url": info["url"],
                        "expires_in": int(info["expires"] - time.time())}
            _stop(token)

        token = secrets.token_urlsafe(16)
        handler = type("StageHandler", (_Handler,), {"token": token,
                                                     "root": live2d_dir,
                                                     "state_provider": state_provider})
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}/{token}/stage.html"
        thread = threading.Thread(target=httpd.serve_forever,
                                  name="neko-live2d-stage", daemon=True)
        thread.start()
        expires = time.time() + max(60, int(ttl))
        _SERVERS[token] = {"server": httpd, "thread": thread,
                           "expires": expires, "url": url, "root": str(live2d_dir)}
        _start_expiry_watch(token)
        return {"ok": True, "reused": False, "url": url,
                "expires_in": int(expires - time.time())}


def _start_expiry_watch(token: str) -> None:
    def watch():
        while True:
            time.sleep(15)
            with _LOCK:
                info = _SERVERS.get(token)
                if info is None:
                    return
                if time.time() >= info["expires"]:
                    _stop(token)
                    return

    threading.Thread(target=watch, name="neko-live2d-stage-ttl", daemon=True).start()


def _stop(token: str) -> None:
    info = _SERVERS.pop(token, None)
    if not info:
        return
    try:
        info["server"].shutdown()
        info["server"].server_close()
    except Exception:  # noqa: BLE001 - 关闭失败不能拖垮插件
        pass


def stop_all() -> None:
    with _LOCK:
        for token in list(_SERVERS):
            _stop(token)


def status() -> dict:
    with _LOCK:
        rows = [{"url": info["url"], "expires_in": int(info["expires"] - time.time())}
                for info in _SERVERS.values()]
    return {"running": bool(rows), "servers": rows}


mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("model/vnd.moc3", ".moc3")


# ---------------------------------------------------------------------------
# Main-window bridge: a fixed, read-only endpoint the app's index.html polls.
# Only spectrum state and the two driver files are served; no filesystem walk.
# ---------------------------------------------------------------------------
BRIDGE_PORT = 48765
_BRIDGE = {"server": None, "thread": None}


_CLIENT = {"at": 0.0, "version": None, "model": 0, "applied": 0, "groove": 0,
           "lips": 0, "lseq": 0, "lport": 0, "lcues": 0, "lw": 0, "lown": 0}


class _BridgeHandler(BaseHTTPRequestHandler):
    state_provider = None
    root: Path = Path(".")

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib handler API
        split = urlsplit(self.path)
        path = split.path
        if path == "/hello":
            from urllib.parse import parse_qs
            q = parse_qs(split.query)
            def one(k, cast=int):
                try:
                    return cast(q.get(k, ["0"])[0])
                except (TypeError, ValueError):
                    return 0
            _CLIENT.update({"at": time.time(), "version": one("v"),
                            "model": one("model"), "applied": one("applied"),
                            "groove": one("groove"),
                            # 为她而唱的口型：有没有连上、第几版口型轨、多少个字
                            "lips": one("lips"), "lseq": one("lseq"),
                            "lport": one("lport"), "lcues": one("lcues"),
                            "lw": one("lw"), "lown": one("lown")})
            self._send(200, b"{\"ok\": true}", "application/json; charset=utf-8")
            return
        if path in ("/state.json", "/health"):
            try:
                payload = (self.state_provider() if path == "/state.json"
                           and self.state_provider else {"ok": True})
            except Exception as exc:  # noqa: BLE001
                payload = {"ok": False, "error": type(exc).__name__}
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path in ("/music_groove.js", "/music_visemes.js", "/bridge.js"):
            target = (self.root / path.lstrip("/")).resolve()
            try:
                target.relative_to(self.root.resolve())
                body = target.read_bytes()
            except (ValueError, OSError):
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            self._send(200, body, "application/javascript; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, *_args):
        return


def start_bridge_server(live2d_dir: Path, *, state_provider=None,
                        port: int = BRIDGE_PORT) -> dict:
    live2d_dir = Path(live2d_dir).resolve()
    with _LOCK:
        info = _BRIDGE.get("server")
        if info is not None and _BRIDGE.get("thread") and _BRIDGE["thread"].is_alive():
            return {"ok": True, "reused": True, "port": _BRIDGE["server"].server_address[1]}
        handler = type("BridgeHandler", (_BridgeHandler,), {
            "state_provider": state_provider, "root": live2d_dir})
        try:
            httpd = socketserver.ThreadingTCPServer(("127.0.0.1", int(port)), handler)
        except OSError as exc:
            return {"ok": False, "error": f"端口 {port} 被占用：{exc}"}
        httpd.daemon_threads = True
        httpd.allow_reuse_address = True
        thread = threading.Thread(target=httpd.serve_forever,
                                  name="neko-live2d-bridge", daemon=True)
        thread.start()
        _BRIDGE.update({"server": httpd, "thread": thread})
        return {"ok": True, "reused": False, "port": httpd.server_address[1],
                "url": f"http://127.0.0.1:{httpd.server_address[1]}/state.json"}


def stop_bridge_server() -> None:
    with _LOCK:
        httpd = _BRIDGE.get("server")
        _BRIDGE.update({"server": None, "thread": None})
    if httpd is not None:
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:  # noqa: BLE001
            pass


def bridge_status() -> dict:
    with _LOCK:
        httpd = _BRIDGE.get("server")
        alive = bool(httpd is not None and _BRIDGE.get("thread")
                     and _BRIDGE["thread"].is_alive())
        port = httpd.server_address[1] if httpd is not None else None
    since = round(time.time() - float(_CLIENT.get("at") or 0.0), 1)
    return {"running": alive, "port": port,
            "client": {**_CLIENT, "seen_s_ago": since if _CLIENT.get("at") else None}}
