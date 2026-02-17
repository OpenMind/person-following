#!/usr/bin/env python3
"""
HTTP control API for person-following system.

Endpoints
---------
GET  /healthz                 -> {"ok": true}
GET  /status                  -> latest status snapshot (JSON)
POST /command                 -> {"cmd": "..."}

Commands
--------
- enroll: Enroll nearest person as target
- clear: Clear current target (does NOT clear history)
- switch: Switch to next person NOT in history
- clear_history: Clear history (memory only)
- delete_history: Clear history AND delete file
- save_history: Force save history to file
- load_history: Force load history from file
- set_max_history: Set max history size (POST with {"cmd":"set_max_history", "size": 10})
- status: Get current status
- quit: Stop the system

Complete Command Usage
--------
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/status
curl -X POST http://127.0.0.1:8080/enroll
curl -X POST http://127.0.0.1:8080/clear
curl -X POST http://127.0.0.1:8080/quit
curl http://127.0.0.1:8080/get_mode
curl -X POST http://127.0.0.1:8080/command -H 'Content-Type: application/json' -d '{"cmd":"set_mode", "mode":"greeting"}'
curl -X POST http://127.0.0.1:8080/command -H 'Content-Type: application/json' -d '{"cmd":"set_mode", "mode":"following"}'

curl -X POST http://127.0.0.1:8080/switch
curl -X POST http://127.0.0.1:8080/greeting_ack
curl -X POST http://127.0.0.1:8080/clear_history
curl -X POST http://127.0.0.1:8080/delete_history
curl -X POST http://127.0.0.1:8080/load_history
curl -X POST http://127.0.0.1:8080/save_history
curl -X POST http://127.0.0.1:8080/command -d '{"cmd":"set_max_history", "size": 10}'
curl -X POST http://127.0.0.1:8080/command -d '{"cmd":"set_mode", "mode":"following"}'
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Literal, Optional, Tuple

OperationMode = Literal["greeting", "following"]

# All valid commands
CommandName = Literal[
    # Common commands (both modes)
    "enroll",
    "clear",
    "status",
    "quit",
    "set_mode",
    "get_mode",
    # Greeting mode only
    "switch",
    "clear_history",
    "delete_history",
    "save_history",
    "load_history",
    "set_max_history",
    "greeting_ack",
]

VALID_COMMANDS = {
    # Common
    "enroll",
    "clear",
    "status",
    "quit",
    "set_mode",
    "get_mode",
    # Greeting mode only
    "switch",
    "clear_history",
    "delete_history",
    "save_history",
    "load_history",
    "set_max_history",
    "greeting_ack",
}

# Commands only available in greeting mode
GREETING_ONLY_COMMANDS = {
    "switch",
    "clear_history",
    "delete_history",
    "save_history",
    "load_history",
    "set_max_history",
    "greeting_ack",
}


@dataclass
class Command:
    """
    Command dataclass for control operations.
    """

    name: CommandName
    ts: float = field(default_factory=time.time)
    params: Dict[str, Any] = field(default_factory=dict)  # for set_max_history


class SharedStatus:
    """
    Thread-safe status snapshot.
    """

    # Added initial_mode parameter
    def __init__(self, initial_mode: OperationMode = "greeting") -> None:
        """
        Initialize thread-safe shared status container.

        Parameters
        ----------
        initial_mode : {'greeting', 'following'}, optional
            Initial operation mode, by default 'greeting'.
        """
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "ok": True,
            "ts": time.time(),
            "operation_mode": initial_mode,
        }

    def set(self, data: Dict[str, Any]) -> None:
        """
        Update shared status data (thread-safe).

        Parameters
        ----------
        data : dict
            New status data to store.
        """
        with self._lock:
            self._data = dict(data)

    def get(self) -> Dict[str, Any]:
        """
        Get current shared status data (thread-safe).

        Returns
        -------
        dict
            Current status data.
        """
        with self._lock:
            return dict(self._data)

    # Get current mode
    def get_mode(self) -> OperationMode:
        """
        Get current operation mode (thread-safe).

        Returns
        -------
        str
            Current operation mode.
        """
        with self._lock:
            return self._data.get("operation_mode", "greeting")

    # Set mode
    def set_mode(self, mode: OperationMode) -> None:
        """
        Set operation mode (thread-safe).

        Parameters
        ----------
        mode : {'greeting', 'following'}
            New operation mode.
        """
        with self._lock:
            self._data["operation_mode"] = mode


class _CommandHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        addr: Tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        cmd_queue: "queue.Queue[Command]",
        shared_status: SharedStatus,
    ) -> None:
        """
        Initialize HTTP request handler.

        Parameters
        ----------
        request : socket
            Client socket.
        client_address : tuple
            Client address (host, port).
        server : HTTPServer
            Parent server instance.
        cmd_queue : queue.Queue
            Queue for dispatching commands.
        shared_status : SharedStatus
            Shared status container.
        """
        super().__init__(addr, handler_cls)
        self.cmd_queue = cmd_queue
        self.shared_status = shared_status


class _Handler(BaseHTTPRequestHandler):
    server: _CommandHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        """
        Suppress default HTTP logging.

        Parameters
        ----------
        fmt : str
            Format string (ignored).
        *args : Any
            Format arguments (ignored).
        """
        return

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        """
        Send JSON response to client.

        Parameters
        ----------
        code : int
            HTTP status code.
        payload : dict
            JSON-serializable response data.
        """
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        """
        Read and parse JSON from request body.

        Returns
        -------
        dict or None
            Parsed JSON data, or None if invalid/missing.
        """
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

    def do_GET(self) -> None:
        """
        Handle HTTP GET requests.

        Endpoints: /healthz, /status, /get_mode
        """
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        if self.path == "/status":
            self._send_json(200, self.server.shared_status.get())
            return
        if self.path == "/get_mode":
            mode = self.server.shared_status.get_mode()
            self._send_json(200, {"ok": True, "operation_mode": mode})
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        """
        Handle HTTP POST requests.

        Endpoints: /enroll, /clear, /switch, /greeting_ack, /clear_history,
        /delete_history, /save_history, /load_history, /quit, /command
        """
        # Read JSON body first (needed for set_max_history)
        data = self._read_json() or {}

        # Convenience aliases: /switch, /save_history, etc.
        path_cmd = self.path.lstrip("/").replace("-", "_")
        if path_cmd in VALID_COMMANDS:
            self._enqueue(path_cmd, data)
            return

        if self.path != "/command":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return

        cmd = data.get("cmd") or data.get("command")
        if not isinstance(cmd, str):
            self._send_json(400, {"ok": False, "error": "missing_cmd"})
            return
        self._enqueue(cmd, data)

    def _enqueue(self, cmd: str, data: Dict[str, Any]) -> None:
        """
        Validate and enqueue a command.

        Checks if command is allowed in current operation mode.

        Parameters
        ----------
        cmd : str
            Command name.
        data : dict
            Command parameters.
        """
        cmd = cmd.strip().lower()
        if cmd not in VALID_COMMANDS:
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "invalid_cmd",
                    "cmd": cmd,
                    "valid_commands": sorted(VALID_COMMANDS),
                },
            )
            return

        # Check if command is allowed in current mode
        current_mode = self.server.shared_status.get_mode()
        if cmd in GREETING_ONLY_COMMANDS and current_mode != "greeting":
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "command_not_available_in_mode",
                    "cmd": cmd,
                    "current_mode": current_mode,
                    "required_mode": "greeting",
                },
            )
            return

        # Handle status (immediate response, no queue)
        if cmd == "status":
            self._send_json(200, self.server.shared_status.get())
            return

        # Handle get_mode (immediate response, no queue)
        if cmd == "get_mode":
            mode = self.server.shared_status.get_mode()
            self._send_json(200, {"ok": True, "operation_mode": mode})
            return

        # Handle set_mode
        if cmd == "set_mode":
            mode = data.get("mode")
            if mode not in ("greeting", "following"):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "invalid_mode",
                        "mode": mode,
                        "valid_modes": ["greeting", "following"],
                    },
                )
                return
            try:
                self.server.cmd_queue.put_nowait(
                    Command(name=cmd, params={"mode": mode})
                )
            except Exception as e:
                self._send_json(
                    500, {"ok": False, "error": "queue_error", "detail": str(e)}
                )
                return
            self._send_json(200, {"ok": True, "queued": cmd, "mode": mode})
            return

        # Handle set_max_history (greeting mode only, already checked above)
        if cmd == "set_max_history":
            size = data.get("size")
            if size is None:
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "missing_size",
                        "usage": '{"cmd": "set_max_history", "size": 10}',
                    },
                )
                return
            try:
                size = int(size)
            except (ValueError, TypeError):
                self._send_json(
                    400, {"ok": False, "error": "invalid_size", "size": size}
                )
                return
            try:
                self.server.cmd_queue.put_nowait(
                    Command(name=cmd, params={"size": size})
                )
            except Exception as e:
                self._send_json(
                    500, {"ok": False, "error": "queue_error", "detail": str(e)}
                )
                return
            self._send_json(200, {"ok": True, "queued": cmd, "size": size})
            return

        # All other commands - simple queue (no params)
        try:
            self.server.cmd_queue.put_nowait(Command(name=cmd))
        except Exception as e:
            self._send_json(
                500, {"ok": False, "error": "queue_error", "detail": str(e)}
            )
            return
        self._send_json(200, {"ok": True, "queued": cmd})


class CommandServer:
    """
    HTTP control server.
    """

    def __init__(
        self,
        host: str,
        port: int,
        cmd_queue: "queue.Queue[Command]",
        shared_status: SharedStatus,
    ) -> None:
        """
        Initialize HTTP command server.

        Parameters
        ----------
        cmd_queue : queue.Queue
            Queue for dispatching commands to main loop.
        shared_status : SharedStatus
            Shared status container for /status endpoint.
        host : str, optional
            Bind address, by default '0.0.0.0'.
        port : int, optional
            Bind port, by default 8080.
        """
        self._host = host
        self._port = port
        self._cmd_queue = cmd_queue
        self._shared_status = shared_status
        self._httpd: Optional[_CommandHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        """
        Get server URL.

        Returns
        -------
        str
            URL string (e.g., 'http://0.0.0.0:8080').
        """
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """
        Start HTTP server in background thread.
        """
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
        """
        Stop HTTP server and wait for thread to finish.
        """
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None
