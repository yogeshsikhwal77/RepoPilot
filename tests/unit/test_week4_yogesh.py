"""Unit & end-to-end integration tests for Yogesh's Week 4 deliverables:
- TesterAgent (`src/repopilot/agents/tester.py`)
- TaskScheduler (`src/repopilot/orchestrator/scheduler.py`)
- RepoPilotOrchestrator (`src/repopilot/orchestrator/graph.py`)
- Friday E2E milestone on `tests/fixtures/sample_repo`[cite: 1, 2]
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from repopilot.agents.tester import TesterAgent
from repopilot.orchestrator.graph import RepoPilotOrchestrator
from repopilot.orchestrator.scheduler import DependencyCycleError, TaskScheduler
from repopilot.orchestrator.state import (
    AgentState,
    Citation,
    Evidence,
    Patch,
    TaskNode,
)


def _make_evidence(chunk_id: str, path: str = "calculator.py") -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        citation=Citation(path=path, commit="deadbeef", start_line=1, end_line=10),
        content="def divide(a, b):\n    return a / b\n",
        source_type="code",
        retrieval="bm25",
        score=0.95,
    )


# ------------------------------------------------------------------
# 1. TaskScheduler unit tests
# ------------------------------------------------------------------

def test_topological_batches_orders_dependencies_correctly() -> None:
    nodes = [
        TaskNode(node_id="n1", description="Find symbol", dependencies=[]),
        TaskNode(node_id="n2", description="Find tests", dependencies=[]),
        TaskNode(node_id="n3", description="Trace callers", dependencies=["n1"]),
        TaskNode(node_id="n4", description="Summarize impact", dependencies=["n2", "n3"]),
    ]
    batches = TaskScheduler.topological_batches(nodes)
    batch_ids = [[n.node_id for n in wave] for wave in batches]

    assert batch_ids[0] == ["n1", "n2"]
    assert batch_ids[1] == ["n3"]
    assert batch_ids[2] == ["n4"]


def test_topological_batches_raises_on_cycle() -> None:
    nodes = [
        TaskNode(node_id="a", description="Step A", dependencies=["b"]),
        TaskNode(node_id="b", description="Step B", dependencies=["a"]),
    ]
    with pytest.raises(DependencyCycleError):
        TaskScheduler.topological_batches(nodes)


def test_scheduler_parallel_evidence_deduplicates_and_merges() -> None:
    def fake_retriever(_state: AgentState) -> dict:
        return {
            "collected_evidence": [
                _make_evidence("calculator.py:1-5"),
                _make_evidence("calculator.py:6-10"),
            ],
            "tool_calls": 1,
            "tokens_used": 120,
        }

    def fake_navigator(_state: AgentState) -> dict:
        return {
            "collected_evidence": [
                _make_evidence("calculator.py:1-5"),  # duplicate chunk_id
            ],
            "impact_set": ["test_calculator.py::test_add", "test_calculator.py::test_add"],
            "tool_calls": 2,
            "tokens_used": 80,
        }

    scheduler = TaskScheduler(retriever=fake_retriever, navigator=fake_navigator)
    state = AgentState(task_input="Fix calculator bug")
    delta = scheduler.gather_evidence_parallel(state)

    assert len(delta["collected_evidence"]) == 2
    assert delta["impact_set"] == ["test_calculator.py::test_add"]
    assert delta["tool_calls"] == 3
    assert delta["tokens_used"] == 200


# ------------------------------------------------------------------
# 2. TesterAgent unit tests (impacted first -> full suite)
# ------------------------------------------------------------------

def test_tester_agent_fails_fast_on_impacted_tests(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n"
        "def test_add():\n    assert add(2, 2) == 4\n\n"
        "def test_other():\n    assert add(1, 0) == 1\n",
        encoding="utf-8",
    )

    bad_patch = Patch(
        filepath="calc.py",
        diff="@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b\n",
        base_commit="HEAD",
    )

    tester = TesterAgent(repo_path=tmp_path)
    report, calls = tester.evaluate_patches(
        patches=[bad_patch],
        impacted_tests=["test_calc.py::test_add"],
    )

    # Should fail on Stage 1 (impacted test) without running full suite
    assert report.passed is False
    assert report.ran_full_suite is False
    assert calls == 1
    assert any("test_add" in f for f in report.failed_tests)


def test_tester_agent_runs_full_suite_after_impacted_pass(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_zero():\n    assert add(0, 0) == 0\n",
        encoding="utf-8",
    )

    good_patch = Patch(
        filepath="calc.py",
        diff="@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n",
        base_commit="HEAD",
    )

    tester = TesterAgent(repo_path=tmp_path)
    report, calls = tester.evaluate_patches(
        patches=[good_patch],
        impacted_tests=["test_calc.py::test_add"],
    )

    assert report.passed is True
    assert report.ran_full_suite is True
    assert report.total_run == 2
    assert report.failed_tests == []
    assert calls == 2


# ------------------------------------------------------------------
# 3. LangGraph Orchestrator retry & budget cap tests + Friday E2E run
# ------------------------------------------------------------------

def test_orchestrator_reflect_retry_loop_until_fix_passes(tmp_path: Path) -> None:
    """First Coder attempt fails; Reflector writes a note; second Coder attempt passes."""
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"
    work_repo = tmp_path / "sample_repo"
    if fixture_dir.exists():
        shutil.copytree(fixture_dir, work_repo)
    else:
        work_repo.mkdir(parents=True)
        (work_repo / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n\n"
            "def subtract(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
        (work_repo / "test_calculator.py").write_text(
            "from calculator import add, subtract\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n\n"
            "def test_subtract():\n    assert subtract(5, 3) == 2\n",
            encoding="utf-8",
        )

    # Introduce a deliberate bug into calculator.py so the pipeline has to fix it
    calc_file = work_repo / "calculator.py"
    original_code = calc_file.read_text(encoding="utf-8")
    buggy_code = original_code.replace("return a + b", "return a - b", 1)
    calc_file.write_text(buggy_code, encoding="utf-8")

    def fake_retriever(_state: AgentState) -> dict:
        return {
            "collected_evidence": [_make_evidence("calculator.py:1-10")],
            "impact_set": ["test_calculator.py"],
            "tool_calls": 1,
        }

    def stateful_coder(state: AgentState) -> dict:
        # Attempt 0: still buggy; Attempt 1 (after reflection): restores the correct code
        if state.loop_count == 0:
            return {
                "draft_patches": [
                    Patch(filepath="calculator.py", diff=buggy_code, base_commit="HEAD")
                ],
                "tokens_used": 150,
                "cost_usd": 0.001,
            }
        return {
            "draft_patches": [
                Patch(filepath="calculator.py", diff=original_code, base_commit="HEAD")
            ],
            "tokens_used": 150,
            "cost_usd": 0.001,
        }

    tester = TesterAgent(repo_path=work_repo)
    orchestrator = RepoPilotOrchestrator(
        retriever=fake_retriever,
        coder=stateful_coder,
        tester=tester,
    )

    final_state = orchestrator.invoke(
        AgentState(task_input="Fix addition bug in calculator.py")
    )

    assert final_state.loop_count == 1
    assert len(final_state.reflection_notes) == 1
    assert final_state.current_test_report is not None
    assert final_state.current_test_report.passed is True
    assert final_state.final_answer is not None
    assert final_state.final_answer.test_report.passed is True
    assert final_state.tokens_used == 300


def test_orchestrator_stops_when_loop_cap_reached(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(
        "from mod import f\n\ndef test_f():\n    assert f() == 2\n", encoding="utf-8"
    )

    def always_wrong_coder(_state: AgentState) -> dict:
        return {
            "draft_patches": [
                Patch(filepath="mod.py", diff="def f():\n    return 1\n", base_commit="HEAD")
            ],
            "tokens_used": 50,
        }

    budget_yaml = tmp_path / "budget.yaml"
    budget_yaml.write_text("max_loops: 2\nmax_tokens: 10000\n", encoding="utf-8")

    orchestrator = RepoPilotOrchestrator(
        coder=always_wrong_coder,
        tester=TesterAgent(repo_path=tmp_path),
        budget_config_path=budget_yaml,
    )

    final_state = orchestrator.invoke(AgentState(task_input="Make f() return 2"))

    assert final_state.loop_count == 2
    assert len(final_state.reflection_notes) == 2
    assert final_state.final_answer is not None
    assert final_state.final_answer.test_report.passed is False