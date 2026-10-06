from pathlib import Path
from textwrap import dedent

import pytest

from repopilot.indexing.parser import parse_repo
from repopilot.indexing.symbol_graph import EdgeKind, SymbolGraph

FILES = {
    "pkg/__init__.py": "",
    "pkg/base.py": """
        class Base:
            def run(self):
                return 1
    """,
    "pkg/util.py": """
        def helper(x):
            return x


        class Serializer:
            def dumps(self):
                return "{}"


        class Config:
            def __init__(self):
                self.debug = False
    """,
    "pkg/shop.py": """
        import json
        from pkg.base import Base
        from . import util
        from .util import helper, Config


        class Cart(Base):
            def add(self, x):
                self.total()
                self.run()
                return helper(x)

            def total(self):
                return json.dumps({})


        def checkout(cart):
            cart.add(1)
            Config()
            return Cart()
    """,
}


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dedent(src).lstrip("\n"), encoding="utf-8")


def _graph(tmp_path, files, **kw) -> SymbolGraph:
    _write(tmp_path, files)
    return SymbolGraph.build(parse_repo(tmp_path, "abc123"), **kw)


@pytest.fixture
def g(tmp_path) -> SymbolGraph:
    return _graph(tmp_path, FILES)


def _triples(graph, kind):
    return {(e.src, e.dst, e.confidence) for e in graph.edges(kind)}


# ---------- call resolution ----------
def test_self_call_is_exact(g):
    assert ("pkg.shop.Cart.add", "pkg.shop.Cart.total", "exact") in _triples(g, EdgeKind.CALLS)


def test_inherited_method_via_self_is_exact(g):
    assert ("pkg.shop.Cart.add", "pkg.base.Base.run", "exact") in _triples(g, EdgeKind.CALLS)


def test_imported_function_resolves_through_relative_import(g):
    assert ("pkg.shop.Cart.add", "pkg.util.helper", "exact") in _triples(g, EdgeKind.CALLS)


def test_class_call_targets_init_when_present_else_class(g):
    calls = _triples(g, EdgeKind.CALLS)
    assert ("pkg.shop.checkout", "pkg.util.Config.__init__", "exact") in calls   # has __init__
    assert ("pkg.shop.checkout", "pkg.shop.Cart", "exact") in calls              # no __init__


def test_unknown_receiver_falls_back_to_heuristic(g):
    assert ("pkg.shop.checkout", "pkg.shop.Cart.add", "heuristic") in _triples(g, EdgeKind.CALLS)


def test_external_call_does_not_match_repo_method(g):
    # json.dumps must NOT link to Serializer.dumps just because the short name matches
    assert g.callers("pkg.util.Serializer.dumps", worst="ambiguous") == []
    assert g.unresolved_calls == 1                      # only json.dumps is unresolved here


# ---------- other edge kinds ----------
def test_inherits_edge(g):
    assert _triples(g, EdgeKind.INHERITS) == {("pkg.shop.Cart", "pkg.base.Base", "exact")}


def test_import_edges_skip_external_and_self(g):
    pairs = {(s, d) for s, d, _ in _triples(g, EdgeKind.IMPORTS)}
    assert pairs == {("pkg.shop", "pkg.base"), ("pkg.shop", "pkg.util")}


def test_contains_edges(g):
    pairs = {(s, d) for s, d, _ in _triples(g, EdgeKind.CONTAINS)}
    assert ("pkg.shop", "pkg.shop.Cart") in pairs
    assert ("pkg.shop.Cart", "pkg.shop.Cart.add") in pairs
    assert ("pkg.util", "pkg.util.helper") in pairs


def test_root_init_symbols_have_no_empty_parent_edge(tmp_path):
    graph = _graph(tmp_path, {"__init__.py": "def f(): pass\n"})
    assert "f" in graph.symbols
    assert all(e.src != "" for e in graph.edges())


def test_stats_snapshot(g):
    assert g.stats() == {
        "modules": 4,
        "symbols": 11,
        "edges": {
            "calls/exact": 5,
            "calls/heuristic": 1,
            "contains/exact": 11,
            "imports/exact": 2,
            "inherits/exact": 1,
        },
        "unresolved_calls": 1,
    }


# ---------- confidence handling ----------
AMBIG = {"m.py": """
    class A:
        def go(self): pass

    class B:
        def go(self): pass

    def f(x):
        x.go()
"""}


def test_ambiguous_edges_and_worst_filter(tmp_path):
    graph = _graph(tmp_path, AMBIG)
    assert _triples(graph, EdgeKind.CALLS) == {
        ("m.f", "m.A.go", "ambiguous"), ("m.f", "m.B.go", "ambiguous"),
    }
    assert graph.callers("m.A.go") == []                                  # default worst="heuristic"
    assert [e.src for e in graph.callers("m.A.go", worst="ambiguous")] == ["m.f"]


def test_max_ambiguity_and_heuristic_switch(tmp_path):
    for kw in ({"max_ambiguity": 1}, {"include_heuristic": False}):
        graph = _graph(tmp_path / str(sorted(kw)), AMBIG, **kw)
        assert graph.edges(EdgeKind.CALLS) == []
        assert graph.unresolved_calls == 1


def test_best_confidence_wins_for_same_edge(tmp_path):
    graph = _graph(tmp_path, {"m.py": """
        class A:
            def go(self): pass

            def run(self):
                self.go()      # exact
                other.go()     # heuristic, same destination
    """})
    calls = graph.edges(EdgeKind.CALLS)
    assert [(e.src, e.dst, e.confidence) for e in calls] == [("m.A.run", "m.A.go", "exact")]


# ---------- queries ----------
def test_lookup_matches_suffix_and_returns_local_symbols(g):
    assert [c.symbol for c in g.lookup("Cart.add")] == ["Cart.add"]
    assert [c.symbol for c in g.lookup("Cart")] == ["Cart"]
    assert [c.symbol for c in g.lookup("pkg.shop.Cart.add")] == ["Cart.add"]
    assert g.lookup("nope") == []


def test_callers_and_callees(g):
    assert {e.src for e in g.callers("pkg.util.helper")} == {"pkg.shop.Cart.add"}
    assert {e.dst for e in g.callees("pkg.shop.checkout")} == {
        "pkg.shop.Cart.add", "pkg.shop.Cart", "pkg.util.Config.__init__",
    }
    assert {e.dst for e in g.callees("pkg.shop.checkout", worst="exact")} == {
        "pkg.shop.Cart", "pkg.util.Config.__init__",
    }


def test_neighbors_out_hop_distances(g):
    assert g.neighbors("pkg.shop.checkout", hops=1, direction="out") == {
        "pkg.shop.Cart.add": 1, "pkg.shop.Cart": 1, "pkg.util.Config.__init__": 1,
    }
    assert g.neighbors("pkg.shop.checkout", hops=2, direction="out") == {
        "pkg.shop.Cart.add": 1, "pkg.shop.Cart": 1, "pkg.util.Config.__init__": 1,
        "pkg.shop.Cart.total": 2, "pkg.base.Base.run": 2, "pkg.util.helper": 2,
    }


def test_neighbors_in_is_reverse_traversal_for_impact(g):
    assert g.neighbors("pkg.util.helper", hops=2, direction="in") == {
        "pkg.shop.Cart.add": 1, "pkg.shop.checkout": 2,
    }
    # an exact-only walk cannot cross the heuristic checkout -> Cart.add edge
    assert g.neighbors("pkg.util.helper", hops=2, direction="in", worst="exact") == {
        "pkg.shop.Cart.add": 1,
    }


def test_neighbors_excludes_start_node_and_handles_cycles(tmp_path):
    graph = _graph(tmp_path, {"m.py": """
        def a(): b()
        def b(): a()
    """})
    assert graph.neighbors("m.a", hops=5) == {"m.b": 1}


# ---------- contract between parser, state.py and graph ----------
def test_every_symbol_respects_chunk_contract(g):
    for qualname, sym in g.symbols.items():
        c = sym.chunk
        assert qualname.endswith(c.symbol)
        assert qualname == ".".join(filter(None, [g.files[c.path].module, c.symbol]))
        assert c.id == f"{c.path}:{c.start_line}-{c.end_line}"


# ---------- smoke test on RepoPilot itself ----------
def test_self_index_resolves_real_imports_and_calls():
    root = Path(__file__).resolve().parents[2]
    if not (root / "src" / "repopilot").is_dir():
        pytest.skip("repo layout not found")
    graph = SymbolGraph.build(parse_repo(root, "abc123"))

    # `from repopilot.indexing.parser import ...` must match the file parsed as src.repopilot...
    assert ("src.repopilot.indexing.symbol_graph", "src.repopilot.indexing.parser") in {
        (e.src, e.dst) for e in graph.edges(EdgeKind.IMPORTS)
    }
    assert "src.repopilot.indexing.parser.parse_repo" in {
        e.src for e in graph.callers("src.repopilot.indexing.parser.parse_file")
    }
    known = set(graph.symbols) | set(graph.modules)
    assert all(e.src in known and e.dst in known for e in graph.edges())