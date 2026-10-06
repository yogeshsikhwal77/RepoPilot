"""Symbol -> covering tests. Feeds retrieval/impact.py (changed symbols -> tests to run first).

Keys are symbol ids: f"{path}::{symbol}", e.g. "shop/cart.py::Cart.add_item".
AGREE THIS FORMAT WITH VIBHAV: symbol_graph.py must use the same ids, otherwise
impact.py cannot join graph nodes to this map. Use `symbol_id()` below.

Test ids use pytest node-id style ("tests/test_cart.py::test_total",
"tests/test_cart.py::TestCart::test_add"), so they can be passed straight to
`pytest` and compared with TestReport.failed_tests / AgentState.impact_set.

Two ways to build the map:

1. DYNAMIC (precise): run the suite once with per-test coverage contexts
      python -m pytest --cov=. --cov-context=test
   then map executed lines back to chunks. `collect_coverage()` runs that
   command. It executes the repo's code on THIS machine, so use it only on
   trusted repos (the sample repo, SWE-bench containers). For untrusted repos
   run the same command inside sandbox/runner.py and hand the .coverage file
   to `from_coverage_data()`.

2. STATIC (fallback, Python only): parse test files with `ast`, link each test
   to symbols whose name it mentions. Over-approximates (may add tests that do
   not really cover a symbol) but never needs to execute anything. That is the
   safe direction for impact analysis: extra tests run, none are missed.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Sequence

from repopilot.orchestrator.state import Chunk

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", "build", "dist", ".mypy_cache"}
_TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")
_PHASES = {"run", "setup", "teardown"}


def symbol_id(path: str, symbol: str) -> str:
    return f"{path}::{symbol}"


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(_TEST_FILE.search(p.as_posix())) or any(part in {"tests", "test"} for part in p.parts[:-1])


class CoverageMap:
    def __init__(self, symbol_to_tests: Optional[dict[str, set[str]]] = None, source: str = "unknown") -> None:
        self.symbol_to_tests: dict[str, set[str]] = defaultdict(set)
        for sym, tests in (symbol_to_tests or {}).items():
            self.symbol_to_tests[sym] |= set(tests)
        self.source = source

    # ---- build helpers ---------------------------------------------------- #
    def add(self, symbol: str, test_id: str) -> None:
        self.symbol_to_tests[symbol].add(test_id)

    def merge(self, other: "CoverageMap") -> "CoverageMap":
        for sym, tests in other.symbol_to_tests.items():
            self.symbol_to_tests[sym] |= tests
        self.source = f"{self.source}+{other.source}"
        return self

    # ---- queries ---------------------------------------------------------- #
    def tests_for(self, symbols: Iterable[str]) -> list[str]:
        """Tests covering ANY of the given symbol ids. Sorted, de-duplicated."""
        found: set[str] = set()
        for s in symbols:
            found |= self.symbol_to_tests.get(s, set())
        return sorted(found)

    def symbols_for(self, test_id: str) -> list[str]:
        return sorted(s for s, tests in self.symbol_to_tests.items() if test_id in tests)

    def uncovered(self, all_symbols: Iterable[str]) -> list[str]:
        """Symbols no test touches: useful for the failure analysis."""
        return sorted(s for s in all_symbols if not self.symbol_to_tests.get(s))

    def stats(self) -> dict:
        all_tests = {t for tests in self.symbol_to_tests.values() for t in tests}
        covered = [s for s, t in self.symbol_to_tests.items() if t]
        return {
            "source": self.source,
            "symbols_covered": len(covered),
            "tests_seen": len(all_tests),
            "avg_tests_per_symbol": (sum(len(self.symbol_to_tests[s]) for s in covered) / len(covered)) if covered else 0.0,
        }

    # ---- persistence ------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        payload = {
            "source": self.source,
            "symbol_to_tests": {s: sorted(t) for s, t in sorted(self.symbol_to_tests.items()) if t},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load(path: str | Path) -> "CoverageMap":
        data = json.loads(Path(path).read_text())
        return CoverageMap({s: set(t) for s, t in data["symbol_to_tests"].items()}, data.get("source", "loaded"))


# --------------------------------------------------------------------------- #
# 1. Dynamic
# --------------------------------------------------------------------------- #
def _safe_core() -> str:
    """Tracer that keeps per-test contexts correct.

    coverage's sys.monitoring core (default on Python 3.14) records a line only
    once, so a line shared by several tests is credited to the first test only.
    Use the C tracer when available, else the slower pure-Python one.
    """
    try:
        from coverage.tracer import CTracer  # noqa: F401

        return "ctrace"
    except ImportError:
        return "pytrace"


def collect_coverage(
    repo_root: str | Path,
    out_file: str | Path,
    pytest_args: Sequence[str] = (),
    timeout: int = 900,
) -> Path:
    """Run pytest with per-test contexts. TRUSTED REPOS ONLY (see module docstring).

    Needs `pytest` and `pytest-cov` installed in the repo's environment.
    A non-zero pytest exit (failing tests) still leaves usable coverage data.
    """
    out_file = Path(out_file).resolve()
    env = {**os.environ, "COVERAGE_FILE": str(out_file), "COVERAGE_CORE": _safe_core()}
    cmd = [sys.executable, "-m", "pytest", "--cov=.", "--cov-context=test", "-q", "-p", "no:cacheprovider", *pytest_args]
    subprocess.run(cmd, cwd=str(repo_root), env=env, timeout=timeout, capture_output=True, text=True)
    if not out_file.exists():
        raise RuntimeError("pytest produced no coverage data (is pytest-cov installed? did collection fail?)")
    return out_file


def _clean_context(raw: str) -> str:
    """'tests/t.py::test_a|run' -> 'tests/t.py::test_a'. '' (import time) -> ''."""
    base, sep, phase = raw.rpartition("|")
    return base if sep and phase in _PHASES else raw


def from_coverage_data(coverage_file: str | Path, repo_root: str | Path, chunks: Iterable[Chunk]) -> CoverageMap:
    """Map executed lines back to symbol chunks using the per-test contexts."""
    from coverage import CoverageData  # lazy: only needed for the dynamic path

    root = Path(repo_root).resolve()
    by_path: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.symbol and not is_test_path(c.path):
            by_path[c.path].append(c)

    data = CoverageData(basename=str(coverage_file))
    data.read()
    cmap = CoverageMap(source="dynamic")
    for measured in data.measured_files():
        p = Path(measured)
        p = p if p.is_absolute() else root / p
        try:
            rel = p.resolve().relative_to(root).as_posix()
        except ValueError:
            continue  # file outside the repo (stdlib, site-packages)
        if rel not in by_path:
            continue
        lines_to_ctx = data.contexts_by_lineno(measured)
        for chunk in by_path[rel]:
            sid = symbol_id(chunk.path, chunk.symbol)  # type: ignore[arg-type]
            for line in range(chunk.start_line, chunk.end_line + 1):
                for raw in lines_to_ctx.get(line, ()):
                    test_id = _clean_context(raw)
                    if test_id:  # skip '' = executed at import, not by a test
                        cmap.add(sid, test_id)
    return cmap


# --------------------------------------------------------------------------- #
# 2. Static fallback (Python)
# --------------------------------------------------------------------------- #
def _names_in(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
    return names


def _is_test_fn(n: ast.AST) -> bool:
    return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")


def _collect_tests(tree: ast.Module, rel: str) -> dict[str, set[str]]:
    """{test_id: names it mentions}. Class helpers/fixtures count for every test in the class."""
    tests: dict[str, set[str]] = {}
    for node in tree.body:
        if _is_test_fn(node):
            tests[f"{rel}::{node.name}"] = _names_in(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            shared: set[str] = set()
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _is_test_fn(sub):
                    shared |= _names_in(sub)
            for sub in node.body:
                if _is_test_fn(sub):
                    tests[f"{rel}::{node.name}::{sub.name}"] = _names_in(sub) | shared
    return tests


def _iter_test_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                rel = (Path(dirpath) / fn).relative_to(root).as_posix()
                if _TEST_FILE.search(rel):
                    yield Path(dirpath) / fn


def from_static_analysis(repo_root: str | Path, chunks: Iterable[Chunk]) -> CoverageMap:
    """Link a test to every symbol whose (short) name it mentions."""
    root = Path(repo_root).resolve()

    # short name -> symbol ids. "Cart.add_item" -> "add_item". "Cart.__init__" is
    # reached by instantiation, so it is keyed under the class name "Cart".
    index: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        if not c.symbol or is_test_path(c.path):
            continue
        parts = c.symbol.split(".")
        short = parts[-1]
        if short == "__init__" and len(parts) > 1:
            short = parts[-2]
        elif short.startswith("__"):
            continue
        index[short].add(symbol_id(c.path, c.symbol))

    cmap = CoverageMap(source="static")
    for file in _iter_test_files(root):
        rel = file.relative_to(root).as_posix()
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for test_id, names in _collect_tests(tree, rel).items():
            for name in names:
                for sid in index.get(name, ()):
                    cmap.add(sid, test_id)
    return cmap


# --------------------------------------------------------------------------- #
def build_coverage_map(
    repo_root: str | Path,
    chunks: Sequence[Chunk],
    coverage_file: str | Path | None = None,
) -> CoverageMap:
    """Dynamic if a .coverage file is given and exists, otherwise static."""
    if coverage_file and Path(coverage_file).exists():
        return from_coverage_data(coverage_file, repo_root, chunks)
    return from_static_analysis(repo_root, chunks)