"""HTTP server for mode control and status endpoints."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from person_following.nodes.person_follow_greet import PersonFollower


class ModeControlHTTPServer(ThreadingHTTPServer):
    """HTTP server for mode control with reference to PersonFollower node."""

    def __init__(self, addr, handler_cls, node: "PersonFollower"):
        """Initialize HTTP server with PersonFollower node reference."""
        super().__init__(addr, handler_cls)
        self.node = node


class ModeControlHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mode control endpoints."""

    server: ModeControlHTTPServer

    def log_message(self, fmt, *args):
        """Suppress default HTTP logging."""
        return

    def _send_json(self, code: int, payload: dict):
        """Send JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[dict]:
        """Read and parse JSON from request body."""
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

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        if self.path == "/status":
            node = self.server.node
            self._send_json(
                200,
                {
                    "ok": True,
                    "operation_mode": node.get_operation_mode(),
                    "state": node.state.value,
                },
            )
            return
        if self.path == "/get_mode":
            mode = self.server.node.get_operation_mode()
            self._send_json(200, {"ok": True, "operation_mode": mode})
            return
        if self.path == "/geofence":
            node = self.server.node
            geofence_status = node.geofence_manager.get_status()
            self._send_json(200, geofence_status)
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        """Handle POST requests."""
        data = self._read_json() or {}

        if self.path == "/set_mode":
            mode = data.get("mode")
            if mode not in ("greeting", "following"):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "invalid_mode",
                        "valid_modes": ["greeting", "following"],
                    },
                )
                return
            success = self.server.node.set_operation_mode(mode)
            self._send_json(200, {"ok": success, "operation_mode": mode})
            return

        if self.path == "/geofence/reset_center":
            # Reset geofence center to current robot position
            node = self.server.node
            success, center = node.geofence_manager.reset_center()
            self._send_json(
                200,
                {
                    "ok": success,
                    "center": center,
                    "message": "Geofence center reset to current position",
                },
            )
            return

        if self.path == "/geofence/enable":
            node = self.server.node
            center = node.geofence_manager.enable()
            self._send_json(
                200,
                {"ok": True, "geofence_enabled": True, "center": center},
            )
            return

        if self.path == "/geofence/disable":
            node = self.server.node
            node.geofence_manager.disable()
            self._send_json(200, {"ok": True, "geofence_enabled": False})
            return

        if self.path == "/command":
            cmd = data.get("cmd")
            if cmd == "set_mode":
                mode = data.get("mode")
                if mode not in ("greeting", "following"):
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "invalid_mode",
                            "valid_modes": ["greeting", "following"],
                        },
                    )
                    return
                success = self.server.node.set_operation_mode(mode)
                self._send_json(200, {"ok": success, "operation_mode": mode})
                return
            self._send_json(400, {"ok": False, "error": "unknown_command"})
            return

        self._send_json(404, {"ok": False, "error": "not_found"})
