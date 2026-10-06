import logging
from textwrap import dedent

import pytest

from repopilot.indexing.parser import ImportRef, _module_name, parse_file, parse_repo
from repopilot.orchestrator.state import Chunk


def _parse(tmp_path, source, name="mod.py"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source).lstrip("\n"), encoding="utf-8")
    return parse_file(path, tmp_path, "abc123")


def _by_local(parsed):
    return {s.chunk.symbol: s for s in parsed.symbols}


# ---------- contract with state.py ----------
def test_chunk_contract_local_symbol_and_id(tmp_path):
    pf = _parse(tmp_path, """
        class Cart:
            def add(self, x):
                return x
        def helper():
            pass
    """)
    syms = _by_local(pf)
    assert set(syms) == {"Cart", "Cart.add", "helper"}            # local names, not module-qualified
    assert syms["Cart.add"].qualname == "mod.Cart.add"            # qualified name lives here
    assert syms["Cart"].kind == "class"
    assert syms["Cart.add"].kind == "method"
    assert syms["helper"].kind == "function"

    chunk = syms["Cart.add"].chunk
    assert isinstance(chunk, Chunk)
    assert chunk.id == "mod.py:2-3" == f"{chunk.path}:{chunk.start_line}-{chunk.end_line}"
    assert chunk.commit == "abc123"
    assert syms["Cart"].chunk.id == "mod.py:1-3"
    assert syms["helper"].chunk.id == "mod.py:4-5"
    assert pf.chunks == [s.chunk for s in pf.symbols]


def test_commit_is_required(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    with pytest.raises(ValueError):
        parse_file(tmp_path / "a.py", tmp_path, "")


# ---------- spans and text ----------
def test_decorators_are_part_of_the_chunk(tmp_path):
    pf = _parse(tmp_path, """
        import functools

        @functools.lru_cache
        def cached(x):
            return x
    """)
    chunk = _by_local(pf)["cached"].chunk
    assert (chunk.start_line, chunk.end_line) == (3, 5)
    assert chunk.text.startswith("@functools.lru_cache")


def test_method_text_is_dedented(tmp_path):
    pf = _parse(tmp_path, """
        class A:
            def m(self):
                return 42
    """)
    assert _by_local(pf)["A.m"].chunk.text == "def m(self):\n    return 42"


def test_class_chunk_excludes_method_bodies(tmp_path):
    pf = _parse(tmp_path, """
        class A:
            \"\"\"Doc.\"\"\"
            x = 1

            @property
            def v(self):
                return 99

            def m(self):
                return 42
    """)
    cls = _by_local(pf)["A"].chunk
    assert "Doc." in cls.text and "x = 1" in cls.text
    assert "return 42" not in cls.text and "return 99" not in cls.text
    assert "@property" not in cls.text
    assert cls.end_line == 10                                     # id still spans the whole class
    assert "return 99" in _by_local(pf)["A.v"].chunk.text         # methods keep their own text


# ---------- calls ----------
def test_nested_function_calls_do_not_leak_to_outer(tmp_path):
    pf = _parse(tmp_path, """
        def outer():
            def inner():
                hidden()
            visible()
    """)
    syms = _by_local(pf)
    assert syms["outer"].calls == ("visible",)
    assert syms["outer.inner"].calls == ("hidden",)
    assert syms["outer.inner"].kind == "function"


def test_call_collection_forms(tmp_path):
    pf = _parse(tmp_path, """
        class S:
            def save(self, x):
                helper(x)
                self.repo.save(x)
                return [f(i) for i in x]

        async def fetch():
            await go()
    """)
    syms = _by_local(pf)
    assert syms["S.save"].calls == ("f", "helper", "self.repo.save")   # sorted, comprehension included
    assert syms["fetch"].calls == ("go",)
    assert syms["fetch"].kind == "function"


def test_bases(tmp_path):
    pf = _parse(tmp_path, """
        class B(A, mod.C):
            pass
    """)
    assert _by_local(pf)["B"].bases == ("A", "mod.C")


# ---------- imports and module names ----------
def test_imports_are_recorded(tmp_path):
    pf = _parse(tmp_path, """
        import os
        import numpy as np
        from a.b import c as d
        from ..x import y
    """)
    assert pf.imports == [
        ImportRef(module="os"),
        ImportRef(module="numpy", alias="np"),
        ImportRef(module="a.b", name="c", alias="d"),
        ImportRef(module="x", name="y", level=2),
    ]


@pytest.mark.parametrize("rel,is_pkg,expected", [
    ("a/b.py", False, "a.b"),
    ("pkg/__init__.py", True, "pkg"),
    ("__init__.py", True, ""),
    ("src/repopilot/x.py", False, "src.repopilot.x"),
])
def test_module_name(rel, is_pkg, expected):
    assert _module_name(rel, is_pkg) == expected


def test_package_init_flags(tmp_path):
    pf = _parse(tmp_path, "def f(): pass\n", name="pkg/__init__.py")
    assert pf.module == "pkg" and pf.is_package is True
    assert _by_local(pf)["f"].qualname == "pkg.f"


def test_sha256_tracks_content(tmp_path):
    a = _parse(tmp_path, "x = 1\n", name="a.py")
    b = _parse(tmp_path, "x = 1\n", name="b.py")
    c = _parse(tmp_path, "x = 2\n", name="c.py")
    assert a.sha256 == b.sha256 != c.sha256


# ---------- parse_repo ----------
def test_parse_repo_skips_ignored_and_logs_broken_files(tmp_path, caplog):
    (tmp_path / "good.py").write_text("def f(): pass\n")
    (tmp_path / "bad.py").write_text("def (:\n")
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00bad")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("def hidden(): pass\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.py").write_text("def hidden2(): pass\n")

    with caplog.at_level(logging.WARNING):
        files = parse_repo(tmp_path, "abc123")

    assert [pf.path for pf in files] == ["good.py"]
    assert "bad.py" in caplog.text and "binary.py" in caplog.text