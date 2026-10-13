"""Model Context Protocol (MCP) Client for RepoPilot.

Connects to in-process or stdio MCP servers (GitHub, Filesystem, Sandbox),
aggregates their tool definitions, and enforces least-privilege scoping:
- Write tools remain disabled until explicitly authorized (Verifier stage)
- Read tools remain accessible for exploration and retrieval
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union

import yaml
from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Represents an MCP tool provided by a server."""

    name: str
    description: str
    server_name: str
    is_write: bool = False
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of invoking an MCP tool."""

    tool_name: str
    is_error: bool
    data: Any
    raw_text: str = ""


class MCPClient:
    """Universal MCP client managing server connections and tool invocation."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config_path = Path(config_path) if config_path else None
        self._servers: Dict[str, MCPServer] = {}
        self._write_tools: Set[str] = set()
        self._tool_to_server: Dict[str, str] = {}
        self._cached_tool_defs: Dict[str, ToolDefinition] = {}

        if self.config_path and self.config_path.exists():
            self._load_config(self.config_path)

    def _load_config(self, path: Path) -> None:
        """Parses server configurations and tool permissions from YAML."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            servers_cfg = data.get("servers", {})
            for s_name, s_info in servers_cfg.items():
                write_tools = s_info.get("write_tools", [])
                for wt in write_tools:
                    self._write_tools.add(wt)
        except Exception as exc:
            logger.warning("Failed to load MCP config from %s: %s", path, exc)

    def register_server(
        self,
        name: str,
        server: MCPServer,
        write_tools: Optional[List[str]] = None,
    ) -> None:
        """Registers an in-process MCPServer instance."""
        self._servers[name] = server
        if write_tools:
            for wt in write_tools:
                self._write_tools.add(wt)
        self._refresh_tools_sync(name, server)

    def mark_write_tool(self, tool_name: str) -> None:
        """Explicitly tags a tool as requiring write authorization."""
        self._write_tools.add(tool_name)
        if tool_name in self._cached_tool_defs:
            self._cached_tool_defs[tool_name].is_write = True

    def _refresh_tools_sync(self, server_name: str, server: MCPServer) -> None:
        """Caches tools exposed by the given server."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running loop: schedule task or run in background
                asyncio.create_task(self._refresh_tools_async(server_name, server))
                return
            loop.run_until_complete(self._refresh_tools_async(server_name, server))
        except RuntimeError:
            asyncio.run(self._refresh_tools_async(server_name, server))

    async def _refresh_tools_async(self, server_name: str, server: MCPServer) -> None:
        """Asynchronously discovers and registers tools from a server."""
        tools = await server.list_tools()
        for t in tools:
            name = getattr(t, "name", str(t))
            desc = getattr(t, "description", "")
            schema = getattr(t, "inputSchema", {}) or {}
            is_write = name in self._write_tools

            tool_def = ToolDefinition(
                name=name,
                description=desc,
                server_name=server_name,
                is_write=is_write,
                input_schema=schema,
            )
            self._tool_to_server[name] = server_name
            self._cached_tool_defs[name] = tool_def

    async def alist_tools(self, allow_write: bool = True) -> List[ToolDefinition]:
        """Asynchronously lists tools available to the current stage."""
        # Ensure all servers are refreshed
        for s_name, s_inst in self._servers.items():
            await self._refresh_tools_async(s_name, s_inst)

        tools: List[ToolDefinition] = []
        for t in self._cached_tool_defs.values():
            if not allow_write and t.is_write:
                continue
            tools.append(t)
        return tools

    def list_tools(self, allow_write: bool = True) -> List[ToolDefinition]:
        """Synchronously lists tools available to the current stage."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, return cached tools
                return [
                    t for t in self._cached_tool_defs.values()
                    if allow_write or not t.is_write
                ]
            return loop.run_until_complete(self.alist_tools(allow_write))
        except RuntimeError:
            return asyncio.run(self.alist_tools(allow_write))

    async def acall_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        allow_write: bool = False,
    ) -> ToolResult:
        """Asynchronously executes an MCP tool with least-privilege permission checks."""
        # Verify tool exists
        server_name = self._tool_to_server.get(tool_name)
        if not server_name or server_name not in self._servers:
            # Try a refresh
            for s_name, s_inst in self._servers.items():
                await self._refresh_tools_async(s_name, s_inst)
            server_name = self._tool_to_server.get(tool_name)

        if not server_name or server_name not in self._servers:
            raise KeyError(f"Tool '{tool_name}' not found across registered MCP servers.")

        # Permission check: Write tools stay off until the Verifier stage
        is_write = tool_name in self._write_tools or self._cached_tool_defs.get(tool_name, ToolDefinition("", "", "")).is_write
        if is_write and not allow_write:
            raise PermissionError(
                f"Permission denied: Tool '{tool_name}' is a write tool. "
                "Write tools stay disabled until the Verifier stage."
            )

        server = self._servers[server_name]
        res = await server.call_tool(tool_name, arguments)

        # Extract structured content or text
        data: Any = None
        raw_text = ""
        is_error = getattr(res, "is_error", False)

        if hasattr(res, "structured_content") and res.structured_content:
            data = res.structured_content.get("result", res.structured_content)
        elif hasattr(res, "content") and res.content:
            first_block = res.content[0]
            raw_text = getattr(first_block, "text", str(first_block))
            try:
                data = json.loads(raw_text)
            except Exception:
                data = raw_text
        else:
            data = str(res)

        return ToolResult(
            tool_name=tool_name,
            is_error=bool(is_error),
            data=data,
            raw_text=raw_text or str(data),
        )

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        allow_write: bool = False,
    ) -> ToolResult:
        """Synchronously executes an MCP tool with permission checks."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In active loop, run in a separate thread to prevent blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.acall_tool(tool_name, arguments, allow_write),
                    )
                    return future.result()
            return loop.run_until_complete(self.acall_tool(tool_name, arguments, allow_write))
        except RuntimeError:
            return asyncio.run(self.acall_tool(tool_name, arguments, allow_write))
