# src/repopilot/mcp/servers/__init__.py
from repopilot.mcp.servers.fs import FileSystemMCPServer, PermissionDeniedError
from repopilot.mcp.servers.sandbox import SandboxMCPServer

try:
    from repopilot.mcp.servers.github import GitHubMCPServer
except ImportError:  # pragma: no cover
    GitHubMCPServer = None  # type: ignore

__all__ = [
    "FileSystemMCPServer",
    "PermissionDeniedError",
    "SandboxMCPServer",
    "GitHubMCPServer",
]