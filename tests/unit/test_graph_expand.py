from pathlib import Path
from textwrap import dedent

import pytest

from repopilot.indexing.parser import parse_repo
from repopilot.indexing.symbol_graph import SymbolGraph
from repopilot.orchestrator.state import Citation, Evidence
from repopilot.retrieval.graph_expand import ExpandConfig, expand

SHAPE_REPO = {
    "shapes.py": """
        class Shape:
            def draw(self):
                self.setup()
                return "shape"

            def setup(self):
                return "setup"


        class Square(Shape):
            def draw(self):
                return "square"


        def render_unique(s):
            return s.setup()
    """
}


def _write_repo(root: Path, files: dict[str, str]) -> SymbolGraph:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dedent(src).lstrip("\n"), encoding="utf-8")
    return SymbolGraph.build(parse_repo(root, "commit1"))


@pytest.fixture
def shape_graph(tmp_path):
    return _write_repo(tmp_path, SHAPE_REPO)


def test_expand_disabled_ablation(shape_graph):
    seed = Evidence(
        chunk_id="shapes.py:2-5",
        citation=Citation(path="shapes.py", commit="commit1", start_line=2, end_line=5),
        content="def draw(self):",
        source_type="code",
        retrieval="bm25",
        score=1.0,
    )
    res = expand(shape_graph, [seed], cfg=ExpandConfig(enabled=False))
    assert res == []


def test_expand_no_seed_leakage(shape_graph):
    # Shape.draw calls Shape.setup
    draw_sym = shape_graph.symbols["shapes.Shape.draw"]
    seed = Evidence(
        chunk_id=draw_sym.chunk.id,
        citation=shape_graph.citation("shapes.Shape.draw"),
        content="def draw",
        source_type="code",
        retrieval="bm25",
        score=1.0,
    )
    expanded = expand(shape_graph, [seed], cfg=ExpandConfig(hops=1, direction="out"))
    expanded_ids = [e.chunk_id for e in expanded]
    assert seed.chunk_id not in expanded_ids
    assert shape_graph.symbols["shapes.Shape.setup"].chunk.id in expanded_ids
    for e in expanded:
        assert e.retrieval == "graph"


def test_hop_decay_values(shape_graph):
    # render_unique -> Shape.setup has single candidate -> confidence heuristic (conf_weight = 0.6)
    # decay = 0.5, seed_score = 1.0
    # hop 1 score = 1.0 * (0.5^1) * 0.6 = 0.30
    render_sym = shape_graph.symbols["shapes.render_unique"]
    seed = Evidence(
        chunk_id=render_sym.chunk.id,
        citation=shape_graph.citation("shapes.render_unique"),
        content="def render_unique",
        source_type="code",
        retrieval="bm25",
        score=1.0,
    )
    cfg = ExpandConfig(
        hops=1,
        direction="out",
        decay=0.5,
        worst_conf="heuristic",
        conf_weight={"exact": 1.0, "heuristic": 0.6, "ambiguous": 0.3},
    )
    res = expand(shape_graph, [seed], cfg)
    setup_ev = next(e for e in res if e.chunk_id == shape_graph.symbols["shapes.Shape.setup"].chunk.id)
    assert abs(setup_ev.score - 0.30) < 1e-4


def test_worst_conf_exact_excludes_heuristic(shape_graph):
    # render_unique calls s.setup(), which is heuristic (unknown receiver)
    render_sym = shape_graph.symbols["shapes.render_unique"]
    seed = Evidence(
        chunk_id=render_sym.chunk.id,
        citation=shape_graph.citation("shapes.render_unique"),
        content="def render_unique",
        source_type="code",
        retrieval="bm25",
        score=1.0,
    )
    cfg = ExpandConfig(hops=2, direction="out", worst_conf="exact")
    res = expand(shape_graph, [seed], cfg)
    setup_chunk_id = shape_graph.symbols["shapes.Shape.setup"].chunk.id
    assert not any(e.chunk_id == setup_chunk_id for e in res)


def test_max_nodes_cap(shape_graph):
    draw_sym = shape_graph.symbols["shapes.Shape.draw"]
    seed = Evidence(
        chunk_id=draw_sym.chunk.id,
        citation=shape_graph.citation("shapes.Shape.draw"),
        content="def draw",
        source_type="code",
        retrieval="bm25",
        score=1.0,
    )
    res = expand(shape_graph, [seed], cfg=ExpandConfig(hops=2, max_nodes=1))
    assert len(res) <= 1


def test_include_overrides_flag(shape_graph):
    # Shape.draw has an override in Square.draw
    draw_sym = shape_graph.symbols["shapes.Shape.draw"]
    seed = Evidence(
        chunk_id=draw_sym.chunk.id,
        citation=shape_graph.citation("shapes.Shape.draw"),
        content="def draw",
        source_type="code",
        retrieval="bm25",
        score=1.0,
    )
    # Without overrides
    res_no = expand(shape_graph, [seed], cfg=ExpandConfig(hops=1, include_overrides=False))
    square_draw_id = shape_graph.symbols["shapes.Square.draw"].chunk.id
    assert square_draw_id not in [e.chunk_id for e in res_no]

    # With overrides
    res_yes = expand(shape_graph, [seed], cfg=ExpandConfig(hops=1, include_overrides=True))
    assert square_draw_id in [e.chunk_id for e in res_yes]
