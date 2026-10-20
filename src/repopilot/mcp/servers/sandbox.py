from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Sequence

import repopilot.sandbox.limits as limits_mod
import repopilot.sandbox.runner as runner_mod
from repopilot.sandbox.test_detect import build_test_plan, detect_test_framework


def resolve_limits_class() -> Any:
    for name in ("SandboxLimits", "ResourceLimits", "Limits", "RunnerLimits"):
        cls = getattr(limits_mod, name, None)
        if inspect.isclass(cls):
            return cls
    for name, obj in inspect.getmembers(limits_mod, inspect.isclass):
        if obj.__module__ == limits_mod.__name__:
            return obj
    return None


def resolve_runner_class() -> Any:
    for name in ("SandboxRunner", "DockerRunner", "DockerSandboxRunner", "Runner", "Sandbox"):
        cls = getattr(runner_mod, name, None)
        if inspect.isclass(cls):
            return cls
    for name, obj in inspect.getmembers(runner_mod, inspect.isclass):
        if (
            obj.__module__ == runner_mod.__name__
            and "Error" not in name
            and "Result" not in name
            and "Config" not in name
        ):
            return obj
    return None


SandboxLimits = resolve_limits_class()
SandboxRunner = resolve_runner_class()


class SandboxMCPServer:
    """
    MCP server exposing no-network sandbox test execution and framework detection.
    """

    name: str = "sandbox"

    def __init__(
        self,
        repo_root: str | Path,
        runner: Any | None = None,
        limits: Any | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.limits = limits or (SandboxLimits() if callable(SandboxLimits) else None)
        if runner is not None:
            self.runner = runner
        elif callable(SandboxRunner):
            self.runner = self._instantiate_runner(SandboxRunner, self.repo_root, self.limits)
        else:
            self.runner = None

    @staticmethod
    def _instantiate_runner(cls: Any, repo_root: Path, limits: Any) -> Any:
        try:
            sig = inspect.signature(cls.__init__)
            kwargs: dict[str, Any] = {}
            if "limits" in sig.parameters and limits is not None:
                kwargs["limits"] = limits
            if "repo_path" in sig.parameters:
                kwargs["repo_path"] = repo_root
            elif "repo_root" in sig.parameters:
                kwargs["repo_root"] = repo_root
            elif "workdir" in sig.parameters:
                kwargs["workdir"] = repo_root
            if "allow_local_fallback" in sig.parameters:
                kwargs["allow_local_fallback"] = True
            return cls(**kwargs)
        except Exception:
            return cls()

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "detect_tests",
                "description": "Detect the test framework and test files in the repository.",
                "parameters": {},
                "scope": "read",
            },
            {
                "name": "run_tests",
                "description": "Run impacted tests first, then full test suite inside the no-network sandbox.",
                "parameters": {
                    "impacted_tests": "list[str] | None",
                    "run_full_suite": "bool",
                },
                "scope": "execute",
            },
            {
                "name": "run_command",
                "description": "Execute a single command inside the no-network sandbox.",
                "parameters": {"command": "list[str]"},
                "scope": "execute",
            },
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        dispatch = {
            "detect_tests": self.detect_tests,
            "run_tests": self.run_tests,
            "run_command": self.run_command,
        }
        if tool_name not in dispatch:
            raise ValueError(f"Unknown sandbox tool: {tool_name}")
        return dispatch[tool_name](**args)

    execute = call_tool
    invoke = call_tool

    def detect_tests(self) -> dict[str, Any]:
        spec = detect_test_framework(self.repo_root)
        return {
            "framework": spec.framework,
            "language": spec.language,
            "base_cmd": spec.base_cmd,
            "test_dirs": spec.test_dirs,
            "test_files": spec.test_files,
        }

    def run_command(self, command: Sequence[str]) -> dict[str, Any]:
        cmd_list = [str(part) for part in command]
        if self.runner is None:
            raise RuntimeError("No SandboxRunner configured on SandboxMCPServer.")

        raw = self._invoke_runner(cmd_list)
        return self._normalize_runner_output(cmd_list, raw)

    def run_tests(
        self,
        impacted_tests: Sequence[str] | None = None,
        run_full_suite: bool = True,
    ) -> dict[str, Any]:
        spec = detect_test_framework(self.repo_root)
        stages: list[dict[str, Any]] = []

        if impacted_tests:
            impacted_cmd = spec.build_command(impacted_tests)
            impacted_res = self.run_command(impacted_cmd)
            impacted_res["stage"] = "impacted"
            stages.append(impacted_res)
            if not impacted_res["passed"]:
                return {
                    "framework": spec.framework,
                    "passed": False,
                    "impacted_passed": False,
                    "full_suite_passed": None,
                    "stages": stages,
                }

        full_passed: bool | None = None
        if run_full_suite or not impacted_tests:
            full_cmd = spec.build_command(None)
            full_res = self.run_command(full_cmd)
            full_res["stage"] = "full_suite"
            stages.append(full_res)
            full_passed = full_res["passed"]

        overall_passed = all(stage["passed"] for stage in stages)
        return {
            "framework": spec.framework,
            "passed": overall_passed,
            "impacted_passed": stages[0]["passed"] if impacted_tests else None,
            "full_suite_passed": full_passed,
            "stages": stages,
        }

    def _invoke_runner(self, cmd_list: list[str]) -> Any:
        for method_name in ("run", "execute", "run_command", "run_in_sandbox", "run_tests"):
            fn = getattr(self.runner, method_name, None)
            if not callable(fn):
                continue

            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                param_names = [p.name for p in params]
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            except (ValueError, TypeError):
                return fn(cmd_list, repo_path=self.repo_root)

            if has_var_kw:
                return fn(cmd_list, repo_path=self.repo_root)

            kwargs: dict[str, Any] = {}
            for path_key in ("repo_path", "repo_root", "workdir", "cwd", "workspace"):
                if path_key in param_names:
                    kwargs[path_key] = self.repo_root
                    break

            if len(param_names) >= 2 and not kwargs:
                first_name = param_names[0].lower()
                if "repo" in first_name or "path" in first_name or "dir" in first_name:
                    return fn(self.repo_root, cmd_list)

            return fn(cmd_list, **kwargs)

        if callable(self.runner):
            return self.runner(cmd_list)
        raise AttributeError("Runner must provide `.run()` or `.execute()`.")

    @staticmethod
    def _normalize_runner_output(command: list[str], raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            exit_code = int(raw.get("exit_code", raw.get("returncode", 0)))
            stdout = str(raw.get("stdout", ""))
            stderr = str(raw.get("stderr", ""))
            timed_out = bool(raw.get("timed_out", False))
            network_disabled = bool(raw.get("network_disabled", True))
        else:
            exit_code = int(getattr(raw, "exit_code", getattr(raw, "returncode", 0)))
            stdout = str(getattr(raw, "stdout", ""))
            stderr = str(getattr(raw, "stderr", ""))
            timed_out = bool(getattr(raw, "timed_out", False))
            network_disabled = bool(getattr(raw, "network_disabled", True))

        passed_count, failed_count = SandboxMCPServer._parse_pytest_counts(stdout + "\n" + stderr)
        return {
            "command": command,
            "exit_code": exit_code,
            "passed": (exit_code == 0 and not timed_out),
            "timed_out": timed_out,
            "network_disabled": network_disabled,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "stdout": stdout,
            "stderr": stderr,
        }

    @staticmethod
    def _parse_pytest_counts(output: str) -> tuple[int, int]:
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        error_match = re.search(r"(\d+)\s+error", output)
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = (int(failed_match.group(1)) if failed_match else 0) + (
            int(error_match.group(1)) if error_match else 0
        )
        return passed, failed
