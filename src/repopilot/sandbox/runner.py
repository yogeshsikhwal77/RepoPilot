"""No-network Docker runner with resource limits for RepoPilot.

Executes test suites and verification scripts in an isolated container
with strictly disabled network access (--network none), capped CPU,
memory, process count, and execution time.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from repopilot.sandbox.limits import SandboxLimits

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Outcome of a sandboxed execution."""

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool = False
    oom_killed: bool = False
    network_disabled: bool = True

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.oom_killed


class DockerSandboxRunner:
    """Docker-based sandbox runner enforcing network isolation and resource limits."""

    DEFAULT_IMAGE = "repopilot-runner:latest"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        default_limits: Optional[SandboxLimits] = None,
        docker_binary: str = "docker",
        allow_local_fallback: bool = False,
    ) -> None:
        self.image = image
        self.default_limits = default_limits or SandboxLimits()
        self.docker_binary = docker_binary
        self.allow_local_fallback = allow_local_fallback
        self._network_mode = "none"  # Enforced: immutable

    @property
    def network_mode(self) -> str:
        """Returns the enforced network mode, which is always 'none'."""
        return self._network_mode

    def is_docker_available(self) -> bool:
        """Checks if Docker CLI is installed and the Docker daemon is responsive."""
        if not shutil.which(self.docker_binary):
            return False
        try:
            res = subprocess.run(
                [self.docker_binary, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return res.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def build_docker_command(
        self,
        command: List[str],
        workdir_host: Union[str, Path],
        container_workdir: str = "/workspace",
        limits: Optional[SandboxLimits] = None,
        env: Optional[Dict[str, str]] = None,
        read_only_root: bool = False,
    ) -> List[str]:
        """Constructs the full docker run CLI invocation with security & resource flags."""
        active_limits = limits or self.default_limits
        host_path = str(Path(workdir_host).resolve())

        cmd = [
            self.docker_binary,
            "run",
            "--rm",
            "--network=none",  # Strict no-network policy
        ]

        # Resource limits (CPU, memory, PIDs)
        cmd.extend(active_limits.to_docker_cli_args())

        # Workdir mount
        cmd.extend(["-v", f"{host_path}:{container_workdir}"])
        cmd.extend(["-w", container_workdir])

        # Security hardening
        cmd.extend(["--security-opt", "no-new-privileges:true"])
        if read_only_root:
            cmd.extend(["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m"])

        # Environment variables
        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])

        # Image and command
        cmd.append(self.image)
        cmd.extend(command)

        return cmd

    def run(
        self,
        command: List[str],
        workdir: Union[str, Path],
        limits: Optional[SandboxLimits] = None,
        env: Optional[Dict[str, str]] = None,
        read_only_root: bool = False,
    ) -> ExecutionResult:
        """Executes a command inside the isolated Docker sandbox."""
        active_limits = limits or self.default_limits
        workdir_path = Path(workdir).resolve()

        if not self.is_docker_available():
            if self.allow_local_fallback:
                logger.warning("Docker unavailable. Using local fallback runner.")
                return self._run_local_fallback(command, workdir_path, active_limits, env)
            raise RuntimeError(
                f"Docker binary '{self.docker_binary}' or daemon is not available. "
                "Set allow_local_fallback=True if testing in non-Docker environments."
            )

        docker_cmd = self.build_docker_command(
            command=command,
            workdir_host=workdir_path,
            limits=active_limits,
            env=env,
            read_only_root=read_only_root,
        )

        start_time = time.perf_counter()
        timed_out = False
        oom_killed = False

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=active_limits.timeout_seconds,
                errors="replace",
            )
            duration = time.perf_counter() - start_time
            stdout = proc.stdout[: active_limits.max_output_bytes]
            stderr = proc.stderr[: active_limits.max_output_bytes]
            exit_code = proc.returncode

            # Exit code 137 typically indicates SIGKILL (e.g., OOM killer or host kill)
            if exit_code == 137:
                oom_killed = True

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_seconds=round(duration, 3),
                timed_out=False,
                oom_killed=oom_killed,
                network_disabled=True,
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            stdout = (exc.stdout or "")[: active_limits.max_output_bytes] if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "")[: active_limits.max_output_bytes] if isinstance(exc.stderr, str) else ""
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr + f"\n[Execution timed out after {active_limits.timeout_seconds}s]",
                exit_code=124,
                duration_seconds=round(duration, 3),
                timed_out=True,
                oom_killed=False,
                network_disabled=True,
            )

    def run_tests(
        self,
        test_targets: Optional[List[str]] = None,
        pytest_args: Optional[List[str]] = None,
        repo_dir: Union[str, Path] = ".",
        limits: Optional[SandboxLimits] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Executes targeted pytest tests inside the sandbox."""
        cmd = ["python", "-m", "pytest"]
        if test_targets:
            cmd.extend(test_targets)
        if pytest_args:
            cmd.extend(pytest_args)

        test_env = dict(env or {})
        # Force offline/no network markers for tools if applicable
        test_env.setdefault("http_proxy", "http://127.0.0.1:0")
        test_env.setdefault("https_proxy", "http://127.0.0.1:0")

        return self.run(
            command=cmd,
            workdir=repo_dir,
            limits=limits,
            env=test_env,
        )

    def _run_local_fallback(
        self,
        command: List[str],
        workdir: Path,
        limits: SandboxLimits,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Fallback executor for environments where Docker is not installed."""
        start_time = time.perf_counter()
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        # In local fallback, block network proxy vars to simulate no-network
        exec_env["http_proxy"] = "http://127.0.0.1:0"
        exec_env["https_proxy"] = "http://127.0.0.1:0"
        exec_env["ALL_PROXY"] = "http://127.0.0.1:0"

        try:
            proc = subprocess.run(
                command,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=limits.timeout_seconds,
                env=exec_env,
                errors="replace",
            )
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                stdout=proc.stdout[: limits.max_output_bytes],
                stderr=proc.stderr[: limits.max_output_bytes],
                exit_code=proc.returncode,
                duration_seconds=round(duration, 3),
                timed_out=False,
                oom_killed=False,
                network_disabled=True,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ExecutionResult(
                stdout=out[: limits.max_output_bytes],
                stderr=err + f"\n[Execution timed out after {limits.timeout_seconds}s]",
                exit_code=124,
                duration_seconds=round(duration, 3),
                timed_out=True,
                oom_killed=False,
                network_disabled=True,
            )
