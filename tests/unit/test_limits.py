import pytest

from repopilot.sandbox.limits import SandboxLimits


def test_sandbox_limits_defaults():
    limits = SandboxLimits()
    assert limits.cpus == 2.0
    assert limits.memory_mb == 2048
    assert limits.pids_limit == 100
    assert limits.timeout_seconds == 60
    assert limits.max_output_bytes == 10 * 1024 * 1024


def test_sandbox_limits_validation():
    with pytest.raises(ValueError, match="cpus must be positive"):
        SandboxLimits(cpus=0.0)

    with pytest.raises(ValueError, match="memory_mb must be at least 32MB"):
        SandboxLimits(memory_mb=16)

    with pytest.raises(ValueError, match="pids_limit must be at least 1"):
        SandboxLimits(pids_limit=0)

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        SandboxLimits(timeout_seconds=-5)

    with pytest.raises(ValueError, match="max_output_bytes must be positive"):
        SandboxLimits(max_output_bytes=0)


def test_sandbox_limits_docker_cli_args():
    limits = SandboxLimits(cpus=1.5, memory_mb=1024, pids_limit=50)
    args = limits.to_docker_cli_args()
    assert "--cpus=1.5" in args
    assert "--memory=1024m" in args
    assert "--memory-swap=1024m" in args
    assert "--pids-limit=50" in args


def test_sandbox_limits_docker_host_config():
    limits = SandboxLimits(cpus=2.0, memory_mb=512, pids_limit=25)
    config = limits.to_docker_host_config()
    assert config["nano_cpus"] == int(2.0 * 1e9)
    assert config["mem_limit"] == "512m"
    assert config["pids_limit"] == 25


def test_sandbox_limits_dict_roundtrip():
    limits = SandboxLimits(cpus=4.0, memory_mb=4096, pids_limit=200, timeout_seconds=120)
    d = limits.to_dict()
    restored = SandboxLimits.from_dict(d)
    assert restored.cpus == 4.0
    assert restored.memory_mb == 4096
    assert restored.pids_limit == 200
    assert restored.timeout_seconds == 120
