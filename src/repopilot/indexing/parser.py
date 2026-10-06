"""Parse Python source into indexed chunks and graph metadata.

Contract with orchestrator/state.py:
  Chunk.id     = f"{path}:{start_line}-{end_line}"
  Chunk.symbol = name local to the file, e.g. "Cart.add_item" (NOT module-qualified)
The module-qualified name lives on ParsedSymbol.qualname and is what the graph keys on.
"""
from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Iterator, Literal, Union

from repopilot.orchestrator.state import Chunk

logger = logging.getLogger(__name__)

IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}

SymbolKind = Literal["class", "method", "function"]
DefNode = Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class ImportRef:
    module: str
    name: str | None = None
    alias: str | None = None
    level: int = 0


@dataclass(frozen=True)
class ParsedSymbol:
    chunk: Chunk
    kind: SymbolKind
    qualname: str                      # module + chunk.symbol; unique key in the symbol graph
    calls: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()


@dataclass
class ParsedFile:
    path: str
    module: str
    sha256: str
    chunks: list[Chunk] = field(default_factory=list)
    symbols: list[ParsedSymbol] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)
    is_package: bool = False


def _span(node: DefNode) -> tuple[int, int]:
    """Line span of a definition, including its decorators."""
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return start, node.end_lineno or node.lineno


def _own_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.Call]:
    """Calls made by this function itself, not by nested defs/classes."""
    stack = list(reversed(node.body))
    while stack:
        n = stack.pop()
        if isinstance(n, _DEF_TYPES):
            continue
        if isinstance(n, ast.Call):
            yield n
        stack.extend(ast.iter_child_nodes(n))


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, module: str, path: str, source: str, commit: str):
        self.module = module
        self.path = path
        self.lines = source.split("\n")
        self.source = source
        self.commit = commit
        self.stack: list[tuple[str, str]] = []
        self.chunks: list[Chunk] = []
        self.symbols: list[ParsedSymbol] = []
        self.imports: list[ImportRef] = []

    @staticmethod
    def _callee(func: ast.expr) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return ast.unparse(func)
        return ""

    # ---------- imports ----------
    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(
            ImportRef(module=alias.name, alias=alias.asname) for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.extend(
            ImportRef(module=node.module or "", name=alias.name,
                      alias=alias.asname, level=node.level)
            for alias in node.names
        )

    # ---------- definitions ----------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node, "class", bases=tuple(ast.unparse(base) for base in node.bases))
        self._descend(node, "class")

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kind: SymbolKind = "method" if self.stack and self.stack[-1][1] == "class" else "function"
        calls = {self._callee(call.func) for call in _own_calls(node)}
        self._add(node, kind, calls=tuple(sorted(calls - {""})))
        self._descend(node, "function")

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def _descend(self, node: DefNode, kind: str) -> None:
        self.stack.append((node.name, kind))
        self.generic_visit(node)
        self.stack.pop()

    def _text(self, node: DefNode, start: int, end: int) -> str:
        """Function: full source. Class: header, docstring and fields, without
        method bodies (those get their own chunks)."""
        skip: set[int] = set()
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, _DEF_TYPES):
                    s, e = _span(child)
                    skip.update(range(s, e + 1))
        kept = [self.lines[i - 1] for i in range(start, end + 1) if i not in skip]
        return dedent("\n".join(kept)) or ast.unparse(node)

    def _add(self, node: DefNode, kind: SymbolKind,
             calls: tuple[str, ...] = (), bases: tuple[str, ...] = ()) -> None:
        local = ".".join([*(name for name, _ in self.stack), node.name])
        qualname = ".".join(filter(None, [self.module, local]))
        start_line, end_line = _span(node)
        chunk = Chunk(
            id=f"{self.path}:{start_line}-{end_line}",
            path=self.path,
            start_line=start_line,
            end_line=end_line,
            symbol=local,
            text=self._text(node, start_line, end_line),
            commit=self.commit,
        )
        self.chunks.append(chunk)
        self.symbols.append(
            ParsedSymbol(chunk=chunk, kind=kind, qualname=qualname, calls=calls, bases=bases)
        )


def _module_name(relative_path: str, is_package: bool) -> str:
    module = relative_path.removesuffix(".py").replace("/", ".")
    if is_package:
        module = module.removesuffix(".__init__")
        if module == "__init__":
            return ""
    return module


def parse_file(path: Path, root: Path, commit: str) -> ParsedFile:
    if not commit:
        raise ValueError("commit must be provided when creating indexed chunks")
    source = path.read_text(encoding="utf-8")
    relative_path = path.relative_to(root).as_posix()
    is_package = path.name == "__init__.py"
    module = _module_name(relative_path, is_package)
    visitor = _SymbolVisitor(module, relative_path, source, commit)
    visitor.visit(ast.parse(source, filename=relative_path))
    return ParsedFile(
        path=relative_path,
        module=module,
        sha256=hashlib.sha256(source.encode()).hexdigest(),
        chunks=visitor.chunks,
        symbols=visitor.symbols,
        imports=visitor.imports,
        is_package=is_package,
    )


def parse_repo(root: Path, commit: str) -> list[ParsedFile]:
    parsed_files: list[ParsedFile] = []
    for path in sorted(root.rglob("*.py")):
        if IGNORED_DIRS & set(path.relative_to(root).parts):
            continue
        try:
            parsed_files.append(parse_file(path, root, commit))
        except (SyntaxError, UnicodeDecodeError) as exc:
            logger.warning("skipped %s: %s", path.relative_to(root), exc)
    return parsed_files
