import pytest

from repopilot.indexing.coverage_map import (
    CoverageMap, _clean_context, build_coverage_map, collect_coverage,
    from_coverage_data, from_static_analysis, is_test_path, symbol_id,
)
from tests.unit.helpers import py_chunks, write_sample_repo

ADD = symbol_id("shop/cart.py", "Cart.add_item")
TOTAL = symbol_id("shop/cart.py", "Cart.total")
INIT = symbol_id("shop/cart.py", "Cart.__init__")
FMT = symbol_id("shop/cart.py", "format_price")
DISC = symbol_id("shop/pricing.py", "apply_discount")
T_ADD, T_TOTAL = "tests/test_cart.py::test_add_item", "tests/test_cart.py::test_total"
T_FMT, T_DISC = "tests/test_cart.py::test_format_price", "tests/test_pricing.py::test_discount"


def test_helpers():
    assert _clean_context("tests/t.py::test_a|run") == "tests/t.py::test_a"
    assert _clean_context("tests/t.py::test_a[1|2]") == "tests/t.py::test_a[1|2]"
    assert _clean_context("") == ""
    assert is_test_path("tests/test_cart.py") and is_test_path("pkg/foo_test.py")
    assert not is_test_path("shop/cart.py")


def _check(cmap):
    assert cmap.tests_for([TOTAL]) == [T_TOTAL]
    assert cmap.tests_for([ADD]) == [T_ADD, T_TOTAL]
    assert cmap.tests_for([INIT]) == [T_ADD, T_TOTAL]
    assert cmap.tests_for([FMT]) == [T_FMT]
    assert cmap.tests_for([DISC]) == [T_DISC]
    assert cmap.tests_for([TOTAL, DISC]) == [T_TOTAL, T_DISC]
    assert cmap.tests_for(["nope::x"]) == []
    # test files are never treated as sources
    assert not any(s.startswith("tests/") for s in cmap.symbol_to_tests)


def test_static(tmp_path):
    repo = write_sample_repo(tmp_path)
    _check(from_static_analysis(repo, py_chunks(repo)))


def test_dynamic_end_to_end(tmp_path):
    pytest.importorskip("pytest_cov")
    repo = write_sample_repo(tmp_path / "repo")
    cov = collect_coverage(repo, tmp_path / ".coverage")
    _check(from_coverage_data(cov, repo, py_chunks(repo)))


def test_build_falls_back_to_static_without_coverage_file(tmp_path):
    repo = write_sample_repo(tmp_path)
    assert build_coverage_map(repo, py_chunks(repo), tmp_path / "missing").source == "static"


def test_query_stats_save_load_merge(tmp_path):
    repo = write_sample_repo(tmp_path / "repo")
    cmap = from_static_analysis(repo, py_chunks(repo))
    assert cmap.symbols_for(T_DISC) == [DISC]
    assert cmap.uncovered([DISC, "shop/x.py::ghost"]) == ["shop/x.py::ghost"]
    assert cmap.stats()["tests_seen"] == 4
    cmap.save(tmp_path / "cov.json")
    loaded = CoverageMap.load(tmp_path / "cov.json")
    assert loaded.tests_for([ADD]) == cmap.tests_for([ADD])
    extra = CoverageMap({ADD: {"tests/extra.py::test_x"}}, source="dynamic")
    assert "tests/extra.py::test_x" in loaded.merge(extra).tests_for([ADD])
    assert loaded.source.endswith("+dynamic")