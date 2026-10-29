"""Small, dependency-light bridge for MCP servers.

The bridge deliberately implements only the JSON-RPC calls needed by the
agent: ``initialize``, ``tools/list`` and ``tools/call``.  It supports a
long-lived stdio child process as well as an HTTP JSON-RPC endpoint.  MCP is
optional; an unavailable or unconfigured server leaves the built-in tools
usable and is reported through :attr:`error`.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPConfig:
    """Connection settings for one MCP server."""

    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    timeout_seconds: float = 10.0

    @property
    def configured(self) -> bool:
        return bool(self.url or self.command)


def _safe_tool_name(name: str) -> str:
    """Return an OpenAI-compatible name while retaining the MCP name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    cleaned = cleaned.strip("_") or "tool"
    # Prefixing prevents collisions with the built-in registry tools.
    return f"mcp_{cleaned}"[:64]


def _normalise_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    result = dict(schema)
    result.setdefault("type", "object")
    result.setdefault("properties", {})
    if not isinstance(result["properties"], dict):
        result["properties"] = {}
    required = result.get("required", [])
    result["required"] = [x for x in required if isinstance(x, str)] if isinstance(required, list) else []
    return result


class MCPBridge:
    """Discover and invoke tools from one MCP server.

    The constructor never raises for connection failures.  This is important
    for local development: the Research Agent must still start when an
    optional MCP process is not installed or is temporarily offline.
    """

    def __init__(self, config: MCPConfig | None = None):
        self.config = config or MCPConfig()
        self.definitions: list[dict[str, Any]] = []
        self._tool_names: dict[str, str] = {}
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._request_id = 0
        self._lock = threading.Lock()
        self.connected = False
        self.error: str | None = None
        self._mock_results: dict[str, Any] = {}
        if self.config.configured:
            self._connect()

    @classmethod
    def from_tools(cls, tools: list[dict[str, Any]], results: dict[str, Any] | None = None) -> "MCPBridge":
        """Build an in-memory bridge, useful for offline tests and demos."""
        bridge = cls(MCPConfig())
        bridge.connected = True
        bridge._register_tools(tools)
        bridge._mock_results = dict(results or {})
        return bridge

    @property
    def available(self) -> bool:
        return self.connected and bool(self.definitions)

    def _connect(self) -> None:
        try:
            if self.config.url:
                self._rpc = self._http_rpc
            else:
                self._start_stdio()
                self._rpc = self._stdio_rpc
            self._rpc_call("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "research-agent", "version": "0.1.0"},
            })
            # MCP initialized is a notification and has no response.
            self._notify("notifications/initialized", {})
            listing = self._rpc_call("tools/list", {})
            tools = (listing.get("tools", []) if isinstance(listing, dict) else [])
            self._register_tools(tools)
            self.connected = True
            logger.info("MCP connected: %d tools discovered", len(self.definitions))
        except Exception as exc:  # optional integration must not break startup
            self.error = f"{type(exc).__name__}: {str(exc)[:500]}"
            self.connected = False
            logger.warning("MCP unavailable; continuing with built-in tools (%s)", self.error)
            self.close()

    def _register_tools(self, tools: list[dict[str, Any]]) -> None:
        self.definitions.clear()
        self._tool_names.clear()
        for item in tools:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            original = item["name"]
            exposed = _safe_tool_name(original)
            # Keep names unique even when a server publishes names differing
            # only by punctuation.
            base, suffix = exposed, 2
            while exposed in self._tool_names:
                exposed = f"{base[:59]}_{suffix}"
                suffix += 1
            self._tool_names[exposed] = original
            self.definitions.append({
                "type": "function",
                "function": {
                    "name": exposed,
                    "description": str(item.get("description") or f"MCP tool: {original}"),
                    "parameters": _normalise_schema(item.get("inputSchema", item.get("input_schema"))),
                },
            })

    def _start_stdio(self) -> None:
        command = self.config.command
        args = list(self.config.args)
        if not args:
            command_parts = shlex.split(command, posix=os.name != "nt")
            if not command_parts:
                raise ValueError("MCP_SERVER_COMMAND is empty")
            command, args = command_parts[0], command_parts[1:]
        self._process = subprocess.Popen(
            [command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_stdio, daemon=True, name="mcp-stdio-reader")
        self._reader.start()

    def _read_stdio(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Ignoring non-JSON MCP stdout line")
                    continue
                if isinstance(message, dict):
                    self._responses.put(message)
        except (OSError, ValueError):
            return

    def _write_stdio(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("MCP stdio process is not running")
        self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def _stdio_rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._write_stdio(message)
            request_id = message.get("id")
            deadline = self.config.timeout_seconds
            while deadline > 0:
                try:
                    response = self._responses.get(timeout=min(0.2, deadline))
                except queue.Empty:
                    deadline -= 0.2
                    if self._process is not None and self._process.poll() is not None:
                        raise RuntimeError("MCP stdio process exited")
                    continue
                if response.get("id") != request_id:
                    continue
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                return response.get("result", {})
            raise TimeoutError("MCP stdio request timed out")

    def _http_rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(self.config.url, json=message, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return payload.get("result", {})

    def _rpc_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        message = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        return self._rpc(message)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if hasattr(self, "_write_stdio") and not self.config.url:
            self._write_stdio(message)

    def call(self, exposed_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.connected:
            return {"error": self.error or "MCP server is not configured or unavailable"}
        original = self._tool_names.get(exposed_name)
        if not original:
            return {"error": f"Unknown MCP tool: {exposed_name}"}
        if not self.config.configured:
            result = self._mock_results.get(original, self._mock_results.get(exposed_name))
            if callable(result):
                result = result(arguments)
            return result if isinstance(result, dict) else {"result": result}
        try:
            result = self._rpc_call("tools/call", {"name": original, "arguments": arguments})
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            logger.warning("MCP tool failed: %s (%s)", exposed_name, type(exc).__name__)
            return {"error": f"MCP tool failed: {str(exc)[:1000]}"}

    def status(self) -> dict[str, Any]:
        return {"configured": self.config.configured, "connected": self.connected,
                "tool_count": len(self.definitions), "error": self.error}

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
        self._process = None


def config_from_settings(settings: Any) -> MCPConfig:
    """Read MCP values from Settings, with env fallback for older Settings."""
    command = str(getattr(settings, "mcp_server_command", "") or os.getenv("MCP_SERVER_COMMAND", "")).strip()
    url = str(getattr(settings, "mcp_server_url", "") or os.getenv("MCP_SERVER_URL", "")).strip()
    raw_args = getattr(settings, "mcp_server_args", None)
    if raw_args is None:
        raw_args = os.getenv("MCP_SERVER_ARGS", "")
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args.strip().startswith("[") else shlex.split(raw_args, posix=os.name != "nt")
        except (json.JSONDecodeError, ValueError):
            parsed = shlex.split(raw_args, posix=os.name != "nt")
    else:
        parsed = list(raw_args or [])
    timeout = float(getattr(settings, "mcp_timeout_seconds", os.getenv("MCP_TIMEOUT_SECONDS", "10")))
    return MCPConfig(command=command, args=tuple(str(x) for x in parsed), url=url, timeout_seconds=max(1.0, min(timeout, 120.0)))
