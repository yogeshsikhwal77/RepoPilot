# tests/unit/test_mcp_fs_and_sandbox.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repopilot.mcp.servers.fs import FileSystemMCPServer, PermissionDeniedError
from repopilot.mcp.servers.sandbox import SandboxMCPServer


def test_fs_server_reads_and_blocks_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    fs = FileSystemMCPServer(tmp_path, allow_writes=False)
    res = fs.call_tool("read_file", {"path": "src/app.py", "start_line": 1, "end_line": 1})
    assert res["content"] == "def add(a, b):"
    assert res["untrusted"] is True

    with pytest.raises(PermissionDeniedError):
        fs.call_tool("read_file", {"path": "../../etc/passwd"})


def test_fs_server_enforces_least_privilege_write_lock(tmp_path: Path) -> None:
    fs = FileSystemMCPServer(tmp_path, allow_writes=False)
    with pytest.raises(PermissionDeniedError):
        fs.call_tool("write_file", {"path": "new.py", "content": "x = 1\n"})

    fs.set_write_enabled(True)
    write_res = fs.call_tool("write_file", {"path": "new.py", "content": "x = 1\n"})
    assert write_res["status"] == "ok"
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_sandbox_server_runs_impacted_then_full_suite(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_sample.py").write_text("def test_one(): assert True\n", encoding="utf-8")

    fake_runner = MagicMock()
    fake_runner.run.side_effect = [
        {"exit_code": 0, "stdout": "1 passed in 0.02s", "stderr": "", "timed_out": False},
        {"exit_code": 0, "stdout": "5 passed in 0.10s", "stderr": "", "timed_out": False},
    ]

    server = SandboxMCPServer(repo_root=tmp_path, runner=fake_runner)
    report = server.call_tool(
        "run_tests",
        {"impacted_tests": ["test_sample.py::test_one"], "run_full_suite": True},
    )

    assert report["passed"] is True
    assert report["impacted_passed"] is True
    assert report["full_suite_passed"] is True
    assert len(report["stages"]) == 2
    assert report["stages"][0]["stage"] == "impacted"
    assert report["stages"][1]["stage"] == "full_suite"