"""Tester agent: applies draft patches in an isolated workspace/sandbox,
runs impacted tests first (fail-fast), then runs the full test suite.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from repopilot.agents.base import BaseAgent
from repopilot.orchestrator.state import AgentState, Patch, TestReport
from repopilot.sandbox.limits import SandboxLimits
from repopilot.sandbox.runner import DockerSandboxRunner, ExecutionResult


class PatchApplyError(RuntimeError):
    """Raised when a generated Patch cannot be cleanly applied to the repo."""


class TesterAgent(BaseAgent):
    """Executes impacted tests first, then the full suite inside the sandbox."""
    
    __test__ = False  # Prevents pytest from treating this agent as a test suite

    def __init__(
        self,
        repo_path: str | Path,
        sandbox_runner: Optional[DockerSandboxRunner] = None,
        limits: Optional[SandboxLimits] = None,
        use_local_fallback: bool = True,
        exec_fn: Optional[Callable[[Sequence[str], Path], ExecutionResult]] = None,
    ) -> None:
        # Safely handle Vibhav-j's BaseAgent initialization regardless of its signature
        try:
            super().__init__(name="tester", allowed_tools=["fs.read_file", "sandbox.run_tests"])
        except TypeError:
            super().__init__()
            self.name = "tester"
            self.allowed_tools = ["fs.read_file", "sandbox.run_tests"]
            
        self.repo_path = Path(repo_path).resolve()
        self.limits = limits or SandboxLimits()
        self.sandbox_runner = sandbox_runner
        self.use_local_fallback = use_local_fallback
        self._exec_fn = exec_fn

    def _run(self, state: AgentState, **kwargs: Any) -> dict[str, Any]:
        """Satisfies BaseAgent abstract method requirement."""
        report, tool_calls = self.evaluate_patches(
            patches=state.draft_patches,
            impacted_tests=state.impact_set,
        )
        return {
            "current_test_report": report,
            "tool_calls": tool_calls,
        }

    def run(self, state: AgentState, **kwargs: Any) -> dict[str, Any]:
        """LangGraph node entry point."""
        return self._run(state, **kwargs)

    def evaluate_patches(
        self,
        patches: Sequence[Patch],
        impacted_tests: Sequence[str] = (),
    ) -> tuple[TestReport, int]:
        """Copy repo to an isolated temp workspace, apply patches, and run 2-stage tests."""
        unique_impacted = list(dict.fromkeys(t for t in impacted_tests if t))
        tool_calls = 0

        with tempfile.TemporaryDirectory(prefix="repopilot_test_") as tmp_dir:
            work_repo = Path(tmp_dir) / "repo"
            shutil.copytree(
                self.repo_path,
                work_repo,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"
                ),
            )

            try:
                for patch in patches:
                    self._apply_patch(work_repo, patch)
            except PatchApplyError as exc:
                return (
                    TestReport(
                        passed=False,
                        total_run=0,
                        failed_tests=["<patch_apply>"],
                        impacted_tests=unique_impacted,
                        ran_full_suite=False,
                        error_logs=str(exc),
                    ),
                    tool_calls,
                )

            base_cmd = self._detect_base_test_cmd(work_repo)

            if unique_impacted:
                impact_cmd = [*base_cmd, *unique_impacted]
                impact_res = self._execute_command(impact_cmd, work_repo)
                tool_calls += 1
                impact_total, impact_failed = self._parse_pytest_output(
                    impact_res.stdout + "\n" + impact_res.stderr
                )

                if impact_res.exit_code != 0 or impact_res.timed_out:
                    if not impact_failed:
                        impact_failed = list(unique_impacted)
                    return (
                        TestReport(
                            passed=False,
                            total_run=max(impact_total, len(unique_impacted)),
                            failed_tests=impact_failed,
                            impacted_tests=unique_impacted,
                            ran_full_suite=False,
                            error_logs=self._format_logs(impact_res),
                        ),
                        tool_calls,
                    )

            full_res = self._execute_command(base_cmd, work_repo)
            tool_calls += 1
            full_total, full_failed = self._parse_pytest_output(
                full_res.stdout + "\n" + full_res.stderr
            )
            passed = (full_res.exit_code == 0) and (not full_res.timed_out)

            if not passed and not full_failed:
                full_failed = ["<test_suite_error>"]

            return (
                TestReport(
                    passed=passed,
                    total_run=full_total,
                    failed_tests=full_failed,
                    impacted_tests=unique_impacted,
                    ran_full_suite=True,
                    error_logs="" if passed else self._format_logs(full_res),
                ),
                tool_calls,
            )

    def _detect_base_test_cmd(self, work_repo: Path) -> list[str]:
        return ["pytest", "-q"]

    def _execute_command(self, cmd: Sequence[str], work_repo: Path) -> ExecutionResult:
        if self._exec_fn is not None:
            return self._exec_fn(cmd, work_repo)

        if self.sandbox_runner is not None:
            try:
                return self.sandbox_runner.run(list(cmd), workdir=work_repo)
            except Exception as exc:
                if not self.use_local_fallback:
                    raise
                _ = exc

        start_time = time.perf_counter()
        timeout = getattr(self.limits, "timeout_seconds", 30)
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=str(work_repo),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
                duration_seconds=round(time.perf_counter() - start_time, 3),
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nTimed out.",
                timed_out=True,
                duration_seconds=round(time.perf_counter() - start_time, 3),
            )

    def _apply_patch(self, work_repo: Path, patch: Patch) -> None:
        target = (work_repo / patch.filepath).resolve()
        if not str(target).startswith(str(work_repo.resolve())):
            raise PatchApplyError(f"Unsafe patch filepath outside repo: {patch.filepath}")

        diff_text = patch.diff.strip()
        if not diff_text:
            raise PatchApplyError(f"Empty diff for {patch.filepath}")

        if "@@" in diff_text or diff_text.startswith("--- ") or diff_text.startswith("diff --git"):
            self._apply_unified_diff(work_repo, target, patch)
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch.diff, encoding="utf-8")

    def _apply_unified_diff(self, work_repo: Path, target: Path, patch: Patch) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as tmp:
            tmp.write(patch.diff if patch.diff.endswith("\n") else patch.diff + "\n")
            patch_path = Path(tmp.name)

        try:
            git_proc = subprocess.run(
                ["git", "apply", "--unsafe-paths", str(patch_path)],
                cwd=str(work_repo),
                capture_output=True,
                text=True,
                check=False,
            )
            if git_proc.returncode == 0:
                return

            self._apply_hunks_python(target, patch.diff)
        finally:
            patch_path.unlink(missing_ok=True)

    @staticmethod
    def _apply_hunks_python(target: Path, diff_text: str) -> None:
        if not target.exists():
            raise PatchApplyError(f"Target file does not exist: {target.name}")

        original_lines = target.read_text(encoding="utf-8").splitlines()
        lines = list(original_lines)
        hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

        diff_lines = diff_text.splitlines()
        idx = 0
        offset = 0
        applied_any = False

        while idx < len(diff_lines):
            m = hunk_header_re.match(diff_lines[idx])
            if not m:
                idx += 1
                continue

            applied_any = True
            start_line = int(m.group(1)) - 1 + offset
            idx += 1

            remove_segment: list[str] = []
            add_segment: list[str] = []

            while idx < len(diff_lines) and not diff_lines[idx].startswith("@@"):
                raw = diff_lines[idx]
                if raw.startswith("---") or raw.startswith("+++") or raw.startswith("\\"):
                    idx += 1
                    continue
                if raw.startswith("-"):
                    remove_segment.append(raw[1:])
                elif raw.startswith("+"):
                    add_segment.append(raw[1:])
                elif raw.startswith(" "):
                    remove_segment.append(raw[1:])
                    add_segment.append(raw[1:])
                else:
                    remove_segment.append(raw)
                    add_segment.append(raw)
                idx += 1

            seg_len = len(remove_segment)
            match_pos: Optional[int] = None
            if lines[start_line : start_line + seg_len] == remove_segment:
                match_pos = start_line
            else:
                for pos in range(len(lines) - seg_len + 1):
                    if lines[pos : pos + seg_len] == remove_segment:
                        match_pos = pos
                        break

            if match_pos is None:
                raise PatchApplyError(f"Hunk mismatch while patching {target.name}")

            lines[match_pos : match_pos + seg_len] = add_segment
            offset += len(add_segment) - seg_len

        if not applied_any:
            raise PatchApplyError(f"No valid unified diff hunks found for {target.name}")

        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple[int, list[str]]:
        failed_tests: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED "):
                node_id = stripped.split()[1]
                if node_id not in failed_tests:
                    failed_tests.append(node_id)

        passed_count = 0
        failed_count = 0
        error_count = 0

        m_pass = re.search(r"(\d+)\s+passed", output)
        if m_pass:
            passed_count = int(m_pass.group(1))
        m_fail = re.search(r"(\d+)\s+failed", output)
        if m_fail:
            failed_count = int(m_fail.group(1))
        m_err = re.search(r"(\d+)\s+errors?", output)
        if m_err:
            error_count = int(m_err.group(1))

        total = passed_count + failed_count + error_count
        if total == 0 and failed_tests:
            total = len(failed_tests)
        return total, failed_tests

    @staticmethod
    def _format_logs(res: ExecutionResult) -> str:
        combined = (res.stdout + "\n" + res.stderr).strip()
        if len(combined) > 4000:
            return combined[-4000:]
        return combined