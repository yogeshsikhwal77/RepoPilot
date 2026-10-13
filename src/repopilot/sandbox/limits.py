"""Resource limits for the RepoPilot Docker sandbox.

Enforces CPU, memory, process (PID), and execution time constraints
to prevent resource exhaustion, fork bombs, and infinite loops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class SandboxLimits:
    """Resource constraints for sandboxed test/code execution.

    Defaults:
        cpus: 2.0 (equivalent to 2 CPU cores)
        memory_mb: 2048 (2 GB RAM limit)
        pids_limit: 100 (prevents fork bombs and process leaks)
        timeout_seconds: 60 (time limit per run)
        max_output_bytes: 10 * 1024 * 1024 (10 MB stdout/stderr cap)
    """

    cpus: float = 2.0
    memory_mb: int = 2048
    pids_limit: int = 100
    timeout_seconds: int = 60
    max_output_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.cpus <= 0:
            raise ValueError(f"cpus must be positive, got {self.cpus}")
        if self.memory_mb < 32:
            raise ValueError(f"memory_mb must be at least 32MB, got {self.memory_mb}")
        if self.pids_limit < 1:
            raise ValueError(f"pids_limit must be at least 1, got {self.pids_limit}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.max_output_bytes <= 0:
            raise ValueError(f"max_output_bytes must be positive, got {self.max_output_bytes}")

    def to_docker_cli_args(self) -> List[str]:
        """Returns standard docker CLI flags enforcing these resource limits."""
        return [
            f"--cpus={self.cpus}",
            f"--memory={self.memory_mb}m",
            f"--memory-swap={self.memory_mb}m",  # Disallow swap beyond memory
            f"--pids-limit={self.pids_limit}",
        ]

    def to_docker_host_config(self) -> Dict[str, Any]:
        """Returns parameters suitable for Docker SDK host_config / containers.run."""
        return {
            "nano_cpus": int(self.cpus * 1e9),
            "mem_limit": f"{self.memory_mb}m",
            "memswap_limit": f"{self.memory_mb}m",
            "pids_limit": self.pids_limit,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes limits to dictionary."""
        return {
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "pids_limit": self.pids_limit,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SandboxLimits:
        """Constructs limits from dictionary."""
        return cls(
            cpus=float(data.get("cpus", 2.0)),
            memory_mb=int(data.get("memory_mb", 2048)),
            pids_limit=int(data.get("pids_limit", 100)),
            timeout_seconds=int(data.get("timeout_seconds", 60)),
            max_output_bytes=int(data.get("max_output_bytes", 10 * 1024 * 1024)),
        )
