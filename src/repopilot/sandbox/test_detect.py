# src/repopilot/sandbox/test_detect.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class TestFrameworkSpec:
    """Detected test runner configuration for a repository."""

    framework: str  # e.g. "pytest", "django", "unittest", "vitest", "jest"
    language: str  # e.g. "python", "javascript"
    base_cmd: list[str]
    test_dirs: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    def build_command(self, target_tests: Sequence[str] | None = None) -> list[str]:
        """Build the CLI command for impacted tests (if provided) or the full suite."""
        if not target_tests:
            return list(self.base_cmd)

        cleaned = [str(t).strip() for t in target_tests if str(t).strip()]
        if not cleaned:
            return list(self.base_cmd)

        if self.framework == "pytest":
            return [*self.base_cmd, *cleaned]

        if self.framework == "django":
            # Convert path/to/test_foo.py::test_bar -> path.to.test_foo.test_bar
            labels: list[str] = []
            for item in cleaned:
                normalized = item.replace("::", ".").replace("/", ".").replace("\\", ".")
                if normalized.endswith(".py"):
                    normalized = normalized[:-3]
                normalized = normalized.replace(".py.", ".")
                labels.append(normalized.strip("."))
            return [*self.base_cmd, *labels]

        if self.framework == "unittest":
            # If specific files or dotted paths are passed, run them directly
            return ["python", "-m", "unittest", *cleaned]

        if self.framework in {"vitest", "jest", "npm"}:
            return [*self.base_cmd, *cleaned]

        return [*self.base_cmd, *cleaned]


def _discover_python_test_files(repo_root: Path) -> tuple[list[str], list[str]]:
    test_dirs: list[str] = []
    for candidate in ("tests", "test"):
        d = repo_root / candidate
        if d.is_dir():
            test_dirs.append(candidate)

    test_files: list[str] = []
    for path in sorted(repo_root.rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if any(part.startswith(".") or part in {"venv", ".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        name = path.name
        if name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{rel}":
            test_files.append(rel)

    return test_dirs, test_files


def detect_test_framework(repo_path: str | Path) -> TestFrameworkSpec:
    """Inspect a repository directory and detect its primary test framework."""
    root = Path(repo_path).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {root}")

    test_dirs, test_files = _discover_python_test_files(root)

    # 1. Check explicit pytest markers (pytest.ini, conftest.py)
    if (root / "pytest.ini").is_file() or (root / "conftest.py").is_file() or any(
        (root / d / "conftest.py").is_file() for d in test_dirs
    ):
        return TestFrameworkSpec(
            framework="pytest",
            language="python",
            base_cmd=["pytest", "-q"],
            test_dirs=test_dirs,
            test_files=test_files,
        )

    # 2. Check pyproject.toml
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        tool_cfg = data.get("tool", {})
        deps_text = json.dumps(data).lower()
        if "pytest" in tool_cfg or "pytest" in deps_text:
            return TestFrameworkSpec(
                framework="pytest",
                language="python",
                base_cmd=["pytest", "-q"],
                test_dirs=test_dirs,
                test_files=test_files,
            )

    # 3. Check setup.cfg / tox.ini
    for cfg_name in ("setup.cfg", "tox.ini"):
        cfg_file = root / cfg_name
        if cfg_file.is_file():
            content = cfg_file.read_text(encoding="utf-8", errors="ignore").lower()
            if "[tool:pytest]" in content or "pytest" in content:
                return TestFrameworkSpec(
                    framework="pytest",
                    language="python",
                    base_cmd=["pytest", "-q"],
                    test_dirs=test_dirs,
                    test_files=test_files,
                )

    # 4. Check Django manage.py
    if (root / "manage.py").is_file():
        return TestFrameworkSpec(
            framework="django",
            language="python",
            base_cmd=["python", "manage.py", "test", "--noinput"],
            test_dirs=test_dirs,
            test_files=test_files,
        )

    # 5. Check requirements*.txt for pytest
    for req_file in sorted(root.glob("requirements*.txt")):
        content = req_file.read_text(encoding="utf-8", errors="ignore").lower()
        if "pytest" in content:
            return TestFrameworkSpec(
                framework="pytest",
                language="python",
                base_cmd=["pytest", "-q"],
                test_dirs=test_dirs,
                test_files=test_files,
            )

    # 6. Inspect discovered Python test files to distinguish unittest vs pytest
    if test_files:
        uses_unittest = False
        uses_pytest = False
        for rel_file in test_files[:20]:
            src = (root / rel_file).read_text(encoding="utf-8", errors="ignore")
            if "import pytest" in src or "@pytest." in src:
                uses_pytest = True
                break
            if "import unittest" in src or "unittest.TestCase" in src:
                uses_unittest = True

        if uses_pytest or not uses_unittest:
            return TestFrameworkSpec(
                framework="pytest",
                language="python",
                base_cmd=["pytest", "-q"],
                test_dirs=test_dirs,
                test_files=test_files,
            )
        return TestFrameworkSpec(
            framework="unittest",
            language="python",
            base_cmd=["python", "-m", "unittest", "discover"],
            test_dirs=test_dirs,
            test_files=test_files,
        )

    # 7. Check Node/TypeScript package.json
    pkg_json = root / "package.json"
    if pkg_json.is_file():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
        except Exception:
            pkg = {}
        deps = {
            **pkg.get("dependencies", {}),
            **pkg.get("devDependencies", {}),
        }
        scripts = pkg.get("scripts", {})
        if "vitest" in deps or "vitest" in str(scripts.get("test", "")):
            return TestFrameworkSpec(
                framework="vitest",
                language="javascript",
                base_cmd=["npx", "vitest", "run"],
            )
        if "jest" in deps or "jest" in str(scripts.get("test", "")):
            return TestFrameworkSpec(
                framework="jest",
                language="javascript",
                base_cmd=["npx", "jest"],
            )
        if "test" in scripts:
            return TestFrameworkSpec(
                framework="npm",
                language="javascript",
                base_cmd=["npm", "test", "--"],
            )

    # Default fallback to pytest
    return TestFrameworkSpec(
        framework="pytest",
        language="python",
        base_cmd=["pytest", "-q"],
        test_dirs=test_dirs,
        test_files=test_files,
    )


def build_test_plan(
    repo_path: str | Path,
    impacted_tests: Sequence[str] | None = None,
) -> list[list[str]]:
    """
    Return the ordered commands to execute in the sandbox:
    1) Impacted tests first (if non-empty)
    2) Full test suite second
    """
    spec = detect_test_framework(repo_path)
    commands: list[list[str]] = []
    if impacted_tests:
        impacted_cmd = spec.build_command(impacted_tests)
        if impacted_cmd != spec.base_cmd:
            commands.append(impacted_cmd)
    commands.append(spec.build_command(None))
    return commands