# src/repopilot/mcp/servers/fs.py
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any


class PermissionDeniedError(PermissionError):
    """Raised when a tool violates path confinement or write-gating rules."""


class FileSystemMCPServer:
    """
    MCP server providing repository file operations.
    Enforces:
      - Path confinement inside `repo_root` (blocks traversal and symlink escape)
      - Least-privilege write lock (`allow_writes=False` by default)
      - Untrusted text tagging so callers know file contents come from the repo
    """

    name: str = "fs"

    def __init__(self, repo_root: str | Path, allow_writes: bool = False) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise FileNotFoundError(f"Repository root does not exist: {self.repo_root}")
        self.allow_writes = allow_writes

    def set_write_enabled(self, enabled: bool) -> None:
        self.allow_writes = bool(enabled)

    def list_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "name": "read_file",
                "description": "Read a file (or line range) relative to the repository root.",
                "parameters": {
                    "path": "str",
                    "start_line": "int | None",
                    "end_line": "int | None",
                },
                "scope": "read",
            },
            {
                "name": "list_files",
                "description": "List files under directory matching optional glob pattern.",
                "parameters": {"directory": "str", "pattern": "str"},
                "scope": "read",
            },
            {
                "name": "search_files",
                "description": "Search for a literal substring across repository files.",
                "parameters": {"query": "str", "glob": "str", "max_results": "int"},
                "scope": "read",
            },
        ]
        if self.allow_writes:
            tools.extend(
                [
                    {
                        "name": "write_file",
                        "description": "Write content to a file inside the repository.",
                        "parameters": {"path": "str", "content": "str"},
                        "scope": "write",
                    },
                    {
                        "name": "apply_edit",
                        "description": "Replace an exact snippet in a file inside the repository.",
                        "parameters": {
                            "path": "str",
                            "old_text": "str",
                            "new_text": "str",
                        },
                        "scope": "write",
                    },
                ]
            )
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        dispatch = {
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_files": self.search_files,
            "write_file": self.write_file,
            "apply_edit": self.apply_edit,
        }
        if tool_name not in dispatch:
            raise ValueError(f"Unknown fs tool: {tool_name}")
        return dispatch[tool_name](**args)

    # Alias for compatibility with MCP clients that call `.execute()` or `.invoke()`
    execute = call_tool
    invoke = call_tool

    def _resolve_safe_path(self, rel_path: str | Path, must_exist: bool = True) -> Path:
        candidate = (self.repo_root / Path(rel_path)).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise PermissionDeniedError(
                f"Path escapes repository root ({self.repo_root}): {rel_path}"
            ) from exc

        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"File not found in repo: {rel_path}")
        return candidate

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        target = self._resolve_safe_path(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"Cannot read directory as file: {path}")

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        s_idx = max(1, start_line or 1)
        e_idx = min(total_lines, end_line or total_lines)
        selected = lines[s_idx - 1 : e_idx] if total_lines > 0 else []

        return {
            "path": target.relative_to(self.repo_root).as_posix(),
            "start_line": s_idx,
            "end_line": e_idx,
            "total_lines": total_lines,
            "content": "\n".join(selected),
            "untrusted": True,
        }

    def list_files(self, directory: str = ".", pattern: str = "*") -> dict[str, Any]:
        base = self._resolve_safe_path(directory, must_exist=True)
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        results: list[str] = []
        ignored_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}
        for item in sorted(base.rglob("*")):
            if any(part in ignored_dirs for part in item.parts):
                continue
            if item.is_file():
                rel = item.relative_to(self.repo_root).as_posix()
                if fnmatch.fnmatch(item.name, pattern) or fnmatch.fnmatch(rel, pattern):
                    results.append(rel)

        return {"directory": directory, "pattern": pattern, "files": results}

    def search_files(
        self, query: str, glob: str = "*.py", max_results: int = 50
    ) -> dict[str, Any]:
        if not query:
            return {"query": query, "matches": []}

        matches: list[dict[str, Any]] = []
        file_list = self.list_files(".", pattern=glob)["files"]
        for rel_path in file_list:
            full_path = self._resolve_safe_path(rel_path, must_exist=True)
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for idx, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(
                        {
                            "path": rel_path,
                            "line": idx,
                            "snippet": line.strip(),
                            "untrusted": True,
                        }
                    )
                    if len(matches) >= max_results:
                        return {"query": query, "matches": matches}
        return {"query": query, "matches": matches}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        if not self.allow_writes:
            raise PermissionDeniedError(
                "Write tools are disabled on FileSystemMCPServer until unlocked."
            )
        target = self._resolve_safe_path(path, must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "path": target.relative_to(self.repo_root).as_posix(),
            "bytes_written": len(content.encode("utf-8")),
            "status": "ok",
        }

    def apply_edit(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        if not self.allow_writes:
            raise PermissionDeniedError(
                "Write tools are disabled on FileSystemMCPServer until unlocked."
            )
        target = self._resolve_safe_path(path, must_exist=True)
        original = target.read_text(encoding="utf-8")
        if old_text not in original:
            raise ValueError(f"Exact old_text snippet not found in {path}")
        updated = original.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        return {
            "path": target.relative_to(self.repo_root).as_posix(),
            "status": "ok",
        }