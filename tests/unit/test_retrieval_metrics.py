import pytest

from eval.retrieval_metrics import (
    GoldSpan,
    gold_locations_from_patch,
    recall_at_k,
    recall_by_method,
)
from repopilot.orchestrator.state import Citation, Evidence
from repopilot.retrieval.hybrid import HybridResult

SAMPLE_DIFF = """
--- a/src/app.py
+++ b/src/app.py
@@ -10,5 +10,6 @@ def run():
-    old_code()
+    new_code()
--- a/src/utils.py
+++ b/src/utils.py
@@ -50,0 +51,3 @@
+    added_line()
"""


def _make_evidence(path: str, start: int, end: int, score: float = 1.0) -> Evidence:
    return Evidence(
        chunk_id=f"{path}:{start}-{end}",
        citation=Citation(path=path, commit="c1", start_line=start, end_line=end),
        content="code",
        source_type="code",
        retrieval="bm25",
        score=score,
    )


def test_gold_locations_from_patch():
    spans = gold_locations_from_patch(SAMPLE_DIFF)
    assert len(spans) == 2

    # app.py: 10, count 5 -> 10..14
    assert spans[0] == GoldSpan(path="src/app.py", start_line=10, end_line=14)

    # utils.py: 50, count 0 -> pure addition anchor at 50..50
    assert spans[1] == GoldSpan(path="src/utils.py", start_line=50, end_line=50)


def test_recall_at_k_file_level():
    gold = [
        GoldSpan(path="src/app.py", start_line=10, end_line=15),
        GoldSpan(path="src/utils.py", start_line=1, end_line=5),
    ]

    ev = [
        _make_evidence("src/app.py", 1, 20),
        _make_evidence("src/other.py", 1, 10),
    ]

    # At k=1: hits src/app.py (1 of 2 gold files) -> 0.5
    assert recall_at_k(ev, gold, k=1, level="file") == 0.5

    # At k=2: still hits only src/app.py -> 0.5
    assert recall_at_k(ev, gold, k=2, level="file") == 0.5

    # If second evidence hits utils.py:
    ev2 = [
        _make_evidence("src/app.py", 1, 20),
        _make_evidence("src/utils.py", 1, 10),
    ]
    assert recall_at_k(ev2, gold, k=2, level="file") == 1.0


def test_recall_at_k_span_level_and_boundaries():
    gold = [
        GoldSpan(path="src/app.py", start_line=10, end_line=20),
        GoldSpan(path="src/app.py", start_line=50, end_line=60),
    ]

    # Non-overlapping: lines 1..9 does not overlap 10..20
    ev_miss = [_make_evidence("src/app.py", 1, 9)]
    assert recall_at_k(ev_miss, gold, k=1, level="span") == 0.0

    # Exact boundary overlap: line 20 overlaps 10..20
    ev_boundary = [_make_evidence("src/app.py", 20, 25)]
    assert recall_at_k(ev_boundary, gold, k=1, level="span") == 0.5

    # Enclosing overlap: lines 5..30 encloses 10..20
    ev_both = [
        _make_evidence("src/app.py", 5, 30),
        _make_evidence("src/app.py", 45, 55),
    ]
    assert recall_at_k(ev_both, gold, k=2, level="span") == 1.0


def test_empty_gold_raises_value_error():
    ev = [_make_evidence("src/app.py", 1, 10)]
    with pytest.raises(ValueError, match="cannot be empty"):
        recall_at_k(ev, [], k=1)


def test_recall_by_method():
    gold = [GoldSpan(path="src/app.py", start_line=10, end_line=20)]
    bm25_ev = [_make_evidence("src/app.py", 10, 20)]
    vec_ev = [_make_evidence("src/other.py", 1, 5)]
    fused_ev = [_make_evidence("src/app.py", 10, 20)]

    detailed = HybridResult(fused=fused_ev, bm25=bm25_ev, vector=vec_ev)
    res = recall_by_method(detailed, gold, k=1, level="span")

    assert res["bm25"] == 1.0
    assert res["vector"] == 0.0
    assert res["fused"] == 1.0
