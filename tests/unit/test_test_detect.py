# tests/unit/test_test_detect.py
from __future__ import annotations

import json
from pathlib import Path

from repopilot.sandbox.test_detect import build_test_plan, detect_test_framework


def test_detect_sample_repo_fixture() -> None:
    fixture_repo = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"
    spec = detect_test_framework(fixture_repo)
    assert spec.framework == "pytest"
    assert spec.language == "python"
    assert "test_calculator.py" in spec.test_files


def test_detect_django_repo(tmp_path: Path) -> None:
    (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_views.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    spec = detect_test_framework(tmp_path)
    assert spec.framework == "django"
    cmd = spec.build_command(["tests/test_views.py::TestView::test_get"])
    assert cmd == ["python", "manage.py", "test", "--noinput", "tests.test_views.TestView.test_get"]


def test_detect_unittest_repo(tmp_path: Path) -> None:
    (tmp_path / "test_math.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n",
        encoding="utf-8",
    )
    spec = detect_test_framework(tmp_path)
    assert spec.framework == "unittest"
    assert spec.base_cmd == ["python", "-m", "unittest", "discover"]


def test_detect_vitest_repo(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"vitest": "^1.0.0"}, "scripts": {"test": "vitest"}}),
        encoding="utf-8",
    )
    spec = detect_test_framework(tmp_path)
    assert spec.framework == "vitest"
    assert spec.build_command(["src/app.test.ts"]) == ["npx", "vitest", "run", "src/app.test.ts"]


def test_build_test_plan_orders_impacted_before_full_suite(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    plan = build_test_plan(tmp_path, impacted_tests=["tests/test_calc.py::test_add"])
    assert len(plan) == 2
    assert plan[0] == ["pytest", "-q", "tests/test_calc.py::test_add"]
    assert plan[1] == ["pytest", "-q"]