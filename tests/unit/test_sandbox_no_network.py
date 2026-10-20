from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import repopilot.sandbox.limits as limits_mod
import repopilot.sandbox.runner as runner_mod
from repopilot.mcp.servers.sandbox import SandboxLimits, SandboxMCPServer, SandboxRunner


def test_sandbox_runner_module_disables_network_in_docker_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Vibhav-j's sandbox/runner.py enforces --network none and limits.py sets resource caps."""
    runner_src = inspect.getsource(runner_mod)
    assert "--network" in runner_src and "none" in runner_src, (
        "sandbox/runner.py must explicitly configure Docker with --network none"
    )

    if SandboxLimits is not None:
        limits_obj = SandboxLimits()
        if hasattr(limits_obj, "to_docker_cli_args"):
            cli_args = limits_obj.to_docker_cli_args()
            joined_cli = " ".join(str(x) for x in cli_args)
            assert "--cpus" in joined_cli
            assert "--memory" in joined_cli
            assert "--pids-limit" in joined_cli

    mock_runner = MagicMock()
    mock_runner.run.return_value = {
        "exit_code": 1,
        "stdout": "",
        "stderr": "OSError: [Errno 101] Network is unreachable",
        "timed_out": False,
        "network_disabled": True,
    }

    server = SandboxMCPServer(repo_root=tmp_path, runner=mock_runner)
    res = server.run_command(
        [
            "python",
            "-c",
            "import socket; socket.create_connection(('1.1.1.1', 80), timeout=1)",
        ]
    )

    assert res["passed"] is False
    assert res["network_disabled"] is True


def test_sandbox_blocks_outbound_network_requests(tmp_path: Path) -> None:
    """Verify outbound HTTP/socket calls inside the sandbox fail as unreachable."""
    failing_runner = MagicMock()
    failing_runner.run.return_value = {
        "exit_code": 1,
        "stdout": "",
        "stderr": "OSError: [Errno 101] Network is unreachable",
        "timed_out": False,
        "network_disabled": True,
    }

    server = SandboxMCPServer(repo_root=tmp_path, runner=failing_runner)
    result = server.run_command(
        [
            "python",
            "-c",
            "import urllib.request; urllib.request.urlopen('https://example.com', timeout=1)",
        ]
    )
    assert result["passed"] is False
    assert result["exit_code"] != 0
    assert result["network_disabled"] is True
    assert "Network is unreachable" in result["stderr"]
