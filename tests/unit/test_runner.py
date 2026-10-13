from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

import pytest

from repopilot.sandbox.limits import SandboxLimits
from repopilot.sandbox.runner import DockerSandboxRunner, ExecutionResult


def test_runner_enforces_no_network():
    runner = DockerSandboxRunner()
    assert runner.network_mode == "none"


def test_build_docker_command_security_flags(tmp_path):
    runner = DockerSandboxRunner(image="test-image:v1")
    limits = SandboxLimits(cpus=2.0, memory_mb=1024, pids_limit=50)

    cmd = runner.build_docker_command(
        command=["python", "-m", "pytest"],
        workdir_host=tmp_path,
        limits=limits,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        read_only_root=True,
    )

    # Must contain docker run
    assert cmd[0] == "docker"
    assert cmd[1] == "run"
    assert "--rm" in cmd
    assert "--network=none" in cmd
    assert "--cpus=2.0" in cmd
    assert "--memory=1024m" in cmd
    assert "--pids-limit=50" in cmd
    assert "--security-opt" in cmd
    assert "no-new-privileges:true" in cmd
    assert "--read-only" in cmd
    assert any("/tmp:rw" in arg for arg in cmd)
    assert "-e" in cmd
    assert "PYTHONDONTWRITEBYTECODE=1" in cmd
    assert "test-image:v1" in cmd
    assert cmd[-3:] == ["python", "-m", "pytest"]


def test_runner_requires_docker_when_fallback_disabled(tmp_path):
    runner = DockerSandboxRunner(docker_binary="nonexistent-docker-bin", allow_local_fallback=False)
    with pytest.raises(RuntimeError, match="not available"):
        runner.run(command=["echo", "1"], workdir=tmp_path)


def test_runner_local_fallback_execution(tmp_path):
    runner = DockerSandboxRunner(allow_local_fallback=True)
    runner.is_docker_available = MagicMock(return_value=False)

    res = runner.run(
        command=["python", "-c", "print('hello from sandbox')"],
        workdir=tmp_path,
        limits=SandboxLimits(timeout_seconds=10),
    )

    assert res.success
    assert "hello from sandbox" in res.stdout
    assert res.exit_code == 0
    assert not res.timed_out
    assert res.network_disabled


def test_runner_local_fallback_timeout(tmp_path):
    runner = DockerSandboxRunner(allow_local_fallback=True)
    runner.is_docker_available = MagicMock(return_value=False)

    res = runner.run(
        command=["python", "-c", "import time; time.sleep(5)"],
        workdir=tmp_path,
        limits=SandboxLimits(timeout_seconds=1),
    )

    assert not res.success
    assert res.timed_out
    assert res.exit_code == 124


def test_runner_docker_mocked_execution(tmp_path):
    runner = DockerSandboxRunner(allow_local_fallback=False)
    runner.is_docker_available = MagicMock(return_value=True)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "all tests passed\n"
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        res = runner.run(
            command=["pytest"],
            workdir=tmp_path,
        )

        assert res.success
        assert res.stdout == "all tests passed\n"
        assert res.exit_code == 0
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0]
        assert "--network=none" in called_cmd


def test_runner_run_tests_helper(tmp_path):
    runner = DockerSandboxRunner(allow_local_fallback=True)
    runner.is_docker_available = MagicMock(return_value=False)

    with patch.object(runner, "run") as mock_run:
        runner.run_tests(
            test_targets=["tests/unit/test_calc.py::test_add"],
            pytest_args=["-v", "--tb=short"],
            repo_dir=tmp_path,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[1]["command"]
        assert cmd[:3] == ["python", "-m", "pytest"]
        assert "tests/unit/test_calc.py::test_add" in cmd
        assert "-v" in cmd
        assert "--tb=short" in cmd
