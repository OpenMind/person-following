#!/usr/bin/env python3
"""
A tiny, dependency-free HTTP control API for the person-following system.

Runs an HTTP server (in a background thread) inside the same process as
`tracked_person_publisher_ros.py`. The server pushes commands into a Queue,
and the main loop consumes them safely.

Endpoints
---------
GET  /healthz                 -> {"ok": true}
GET  /status                  -> latest status snapshot (JSON)
POST /command                 -> {"cmd": "enroll"|"clear"|"quit"|"status"}
POST /enroll | /clear | /quit -> convenience aliases

Notes
-----
- No auth/token by design (per your request). Prefer binding to 127.0.0.1
  and using `--network host` so only the host can reach it.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Literal, Optional, Tuple

CommandName = Literal["enroll", "clear", "quit", "status"]


@dataclass(frozen=True)
class Command:
    """Command dataclass for control operations."""

    name: CommandName
    ts: float = field(default_factory=time.time)


class SharedStatus:
    """Thread-safe status snapshot that the HTTP server can return."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"ok": True, "ts": time.time()}

    def set(self, data: Dict[str, Any]) -> None:
        """Update the status data."""
        with self._lock:
            self._data = dict(data)

    def get(self) -> Dict[str, Any]:
        """Get a copy of the status data."""
        with self._lock:
            return dict(self._data)


class _CommandHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        addr: Tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        cmd_queue: "queue.Queue[Command]",
        shared_status: SharedStatus,
    ) -> None:
        super().__init__(addr, handler_cls)
        self.cmd_queue = cmd_queue
        self.shared_status = shared_status


class _Handler(BaseHTTPRequestHandler):
    server: _CommandHTTPServer  # for type checkers

    def log_message(self, fmt: str, *args: Any) -> None:
        # Silence default HTTP request logs.
        return

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        if self.path == "/status":
            self._send_json(200, self.server.shared_status.get())
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        # Convenience aliases
        if self.path in ("/enroll", "/clear", "/quit", "/status"):
            cmd = self.path.lstrip("/")
            self._enqueue(cmd)  # type: ignore[arg-type]
            return

        if self.path != "/command":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return

        data = self._read_json() or {}
        cmd = data.get("cmd") or data.get("command")
        if not isinstance(cmd, str):
            self._send_json(400, {"ok": False, "error": "missing_cmd"})
            return
        self._enqueue(cmd)

    def _enqueue(self, cmd: str) -> None:
        cmd = cmd.strip().lower()
        if cmd not in ("enroll", "clear", "quit", "status"):
            self._send_json(400, {"ok": False, "error": "invalid_cmd", "cmd": cmd})
            return

        if cmd == "status":
            self._send_json(200, self.server.shared_status.get())
            return

        try:
            self.server.cmd_queue.put_nowait(Command(name=cmd))
        except Exception as e:
            self._send_json(
                500, {"ok": False, "error": "queue_error", "detail": str(e)}
            )
            return

        self._send_json(200, {"ok": True, "queued": cmd})


class CommandServer:
    """Starts/stops the HTTP control server in a background thread."""

    def __init__(
        self,
        host: str,
        port: int,
        cmd_queue: "queue.Queue[Command]",
        shared_status: SharedStatus,
    ) -> None:
        self._host = host
        self._port = port
        self._cmd_queue = cmd_queue
        self._shared_status = shared_status

        self._httpd: Optional[_CommandHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        if self._httpd is not None:
            return

        self._httpd = _CommandHTTPServer(
            (self._host, self._port),
            _Handler,
            cmd_queue=self._cmd_queue,
            shared_status=self._shared_status,
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None
