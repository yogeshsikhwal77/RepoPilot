import pytest

from repopilot.orchestrator.state import Answer, Citation, Evidence, TestReport

from eval import run_eval
from eval.metrics import (
    EvalRecord, EvalTask, across_runs, cost_per_resolved, mean_recall_at_k,
    ranked_files, recall_at_k, regression_rate, resolve_rate,
)


def ev(path, score, method="bm25"):
    return Evidence(
        chunk_id=f"{path}:1-1",
        citation=Citation(path=path, commit="abc", start_line=1, end_line=1),
        content="", source_type="code", retrieval=method, score=score,
    )


def answer(cost, tokens=100):
    return Answer(diffs=[], test_report=TestReport(passed=True, total_run=0),
                  verified_claims=[], cost_usd=cost, tokens=tokens)


def rec(task_id="t1", run_idx=0, resolved=None, regressed=None, cost=None, evidence=()):
    return EvalRecord(task_id=task_id, config="c", run_idx=run_idx, evidence=list(evidence),
                      resolved=resolved, regressed=regressed,
                      answer=answer(cost) if cost is not None else None)


# ----------------------------------------------------------- retrieval

def test_ranked_files_orders_by_score_and_dedupes():
    e = [ev("a.py", 0.2), ev("b.py", 0.9), ev("a.py", 0.8)]
    assert ranked_files(e) == ["b.py", "a.py"]


def test_ranked_files_filters_by_method():
    e = [ev("a.py", 0.9, "bm25"), ev("b.py", 0.5, "vector")]
    assert ranked_files(e, "vector") == ["b.py"]


def test_recall_two_of_three_gold_found():
    e = [ev("a.py", 0.9), ev("x.py", 0.8), ev("b.py", 0.7), ev("c.py", 0.1)]
    assert recall_at_k(e, ["a.py", "b.py", "c.py"], k=3) == pytest.approx(2 / 3)


def test_recall_respects_k():
    e = [ev("x.py", 0.9), ev("a.py", 0.5)]
    assert recall_at_k(e, ["a.py"], k=1) == 0.0
    assert recall_at_k(e, ["a.py"], k=2) == 1.0


def test_recall_with_no_gold_files_is_none():
    assert recall_at_k([ev("a.py", 1.0)], [], k=5) is None


def test_crashed_run_counts_as_zero_recall():
    task = EvalTask(task_id="t1", repo="o/r", base_commit="abc",
                    problem_statement="", gold_files=["a.py"])
    crashed = rec(evidence=[])
    assert mean_recall_at_k([crashed], {"t1": task}, k=5) == 0.0


# ------------------------------------------------------------ outcomes

def test_resolve_rate_counts_ungraded_as_unresolved():
    assert resolve_rate([rec(resolved=True), rec(resolved=False), rec(resolved=None)]) == pytest.approx(1 / 3)


def test_cost_per_resolved_includes_failed_runs():
    rs = [rec(resolved=True, cost=1.0), rec(resolved=False, cost=3.0)]
    assert cost_per_resolved(rs) == 4.0


def test_cost_per_resolved_zero_resolved_is_none():
    assert cost_per_resolved([rec(resolved=False, cost=1.0)]) is None


def test_regression_rate_ignores_ungraded():
    assert regression_rate([rec(regressed=True), rec(regressed=False), rec()]) == 0.5
    assert regression_rate([rec()]) is None


def test_across_runs_mean_and_std():
    rs = [rec(run_idx=0, resolved=True), rec(run_idx=1, resolved=False)]
    m, s = across_runs(rs, resolve_rate)
    assert m == 0.5 and s == 0.5


# --------------------------------------------------------------- harness

def test_parse_gold_files():
    patch = (
        "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n--- a/pkg/b.py\n+++ b/pkg/b.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/pkg/a.py b/pkg/a.py\n"
    )
    assert run_eval.parse_gold_files(patch) == ["pkg/a.py", "pkg/b.py"]


def test_dummy_gold_gives_full_recall_and_dummy_none_gives_zero():
    task = EvalTask(task_id="t1", repo="o/r", base_commit="abc",
                    problem_statement="", gold_files=["a.py", "b.py"])
    gold = run_eval.DummyGold().run(task, 0)
    none = run_eval.DummyNone().run(task, 0)
    assert recall_at_k(gold.evidence, task.gold_files, 5) == 1.0
    assert recall_at_k(none.evidence, task.gold_files, 5) == 0.0


def test_cache_prevents_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "CACHE_DIR", tmp_path)
    calls = []

    class Counting:
        name = "counting"
        fingerprint = "v1"

        def run(self, task, run_idx):
            calls.append(1)
            return EvalRecord(task_id=task.task_id, config=self.name, run_idx=run_idx)

    task = EvalTask(task_id="t1", repo="o/r", base_commit="abc",
                    problem_statement="", gold_files=["a.py"])
    run_eval.run_one(Counting(), task, 0)
    run_eval.run_one(Counting(), task, 0)
    assert len(calls) == 1


def test_crash_is_recorded_not_raised_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "CACHE_DIR", tmp_path)

    class Boom:
        name = "boom"
        fingerprint = "v1"

        def run(self, task, run_idx):
            raise RuntimeError("nope")

    task = EvalTask(task_id="t1", repo="o/r", base_commit="abc",
                    problem_statement="", gold_files=["a.py"])
    r = run_eval.run_one(Boom(), task, 0)
    assert r.error and "nope" in r.error
    assert list(tmp_path.iterdir()) == []


def test_forgotten_cost_is_caught():
    bad = rec(cost=0.0)  # tokens=100 but cost_usd=0.0
    with pytest.raises(ValueError):
        run_eval.check_record(bad)