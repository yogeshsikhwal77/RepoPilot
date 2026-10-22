"""Parse Python source into indexed chunks and graph metadata.

Contract with orchestrator/state.py:
  Chunk.id     = f"{path}:{start_line}-{end_line}"
  Chunk.symbol = name local to the file, e.g. "Cart.add_item" (NOT module-qualified)
The module-qualified name lives on Symbol.qualname and is what the graph keys on.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Iterator, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from repopilot.orchestrator.state import Chunk

logger = logging.getLogger(__name__)

IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}

SymbolKind = Literal["class", "function", "method"]
DefNode = Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class ImportRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    module: str
    name: Optional[str] = None
    alias: Optional[str] = None
    level: int = 0


class Symbol(BaseModel):
    model_config = ConfigDict(frozen=True)

    qualname: str
    local_name: str
    kind: SymbolKind
    file: str
    parent: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: Optional[str] = None
    calls: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    chunk: Optional[Chunk] = None  # primary chunk for backward-compatibility


# Alias for backwards compatibility
ParsedSymbol = Symbol


class ParsedFile(BaseModel):
    path: str
    module: str
    sha256: str
    commit: str
    is_package: bool = False
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[ImportRef] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)


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


def _callee(func: ast.expr) -> str:
    """Extract dotted callee name; '?.' prefix for non-name receiver."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        val = _callee(func.value)
        if val and not val.startswith("?"):
            return f"{val}.{func.attr}"
        return f"?.{func.attr}"
    return "?."


def _strip_blank_edges(lines: list[str], start_line: int, end_line: int) -> tuple[int, int]:
    """Strip leading and trailing blank lines from a 1-based line range."""
    s = start_line
    e = end_line
    while s <= e and (s - 1 >= len(lines) or lines[s - 1].strip() == ""):
        s += 1
    while e >= s and (e - 1 >= len(lines) or lines[e - 1].strip() == ""):
        e -= 1
    return s, e


def _make_windowed_chunks(
    lines: list[str],
    path: str,
    start_line: int,
    end_line: int,
    symbol: Optional[str],
    commit: str,
    max_chunk_lines: int,
) -> list[Chunk]:
    """Create one or more chunks for a line span, splitting into windows if > max_chunk_lines."""
    s, e = _strip_blank_edges(lines, start_line, end_line)
    if s > e:
        return []

    chunks: list[Chunk] = []
    curr = s
    while curr <= e:
        w_end = min(curr + max_chunk_lines - 1, e)
        raw_text = "\n".join(lines[curr - 1 : w_end])
        text = dedent(raw_text)
        chunks.append(
            Chunk(
                id=f"{path}:{curr}-{w_end}",
                path=path,
                start_line=curr,
                end_line=w_end,
                symbol=symbol,
                text=text,
                commit=commit,
            )
        )
        curr = w_end + 1
    return chunks


class _FileParser:
    def __init__(
        self,
        module: str,
        path: str,
        source: str,
        commit: str,
        max_chunk_lines: int = 80,
    ):
        self.module = module
        self.path = path
        self.lines = source.split("\n")
        self.source = source
        self.commit = commit
        self.max_chunk_lines = max_chunk_lines

        self.symbols: list[Symbol] = []
        self.imports: list[ImportRef] = []
        self.chunks: list[Chunk] = []
        self._covered_lines: set[int] = set()

    def parse(self, tree: ast.AST) -> None:
        self._collect_imports(tree)
        self._collect_definitions(tree)
        self._collect_residuals()
        # Sort chunks deterministically by start_line
        self.chunks.sort(key=lambda c: (c.start_line, c.end_line))

    def _collect_imports(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(
                        ImportRef(module=alias.name, alias=alias.asname)
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.imports.append(
                        ImportRef(
                            module=node.module or "",
                            name=alias.name,
                            alias=alias.asname,
                            level=node.level,
                        )
                    )

    def _collect_definitions(self, tree: ast.AST) -> None:
        """Walk classes and top-level functions."""
        for node in tree.body if isinstance(tree, ast.Module) else []:
            if isinstance(node, ast.ClassDef):
                self._handle_class(node, parent=self.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._handle_function(node, parent=self.module, kind="function")

    def _handle_class(self, node: ast.ClassDef, parent: str) -> None:
        local_name = node.name
        qualname = f"{parent}.{local_name}" if parent else local_name
        cls_start, cls_end = _span(node)
        bases = tuple(ast.unparse(b) for b in node.bases)
        doc = ast.get_docstring(node)

        # Class header chunk: from class start line to line before first member
        first_member_line = cls_end + 1
        members: list[DefNode] = []
        for child in node.body:
            if isinstance(child, _DEF_TYPES):
                members.append(child)
                c_start, _ = _span(child)
                if c_start < first_member_line:
                    first_member_line = c_start

        hdr_end = first_member_line - 1 if members else cls_end
        hdr_chunks = _make_windowed_chunks(
            self.lines,
            self.path,
            cls_start,
            hdr_end,
            symbol=local_name,
            commit=self.commit,
            max_chunk_lines=self.max_chunk_lines,
        )
        self.chunks.extend(hdr_chunks)
        primary_chunk = hdr_chunks[0] if hdr_chunks else None
        for c in hdr_chunks:
            self._covered_lines.update(range(c.start_line, c.end_line + 1))

        sym = Symbol(
            qualname=qualname,
            local_name=local_name,
            kind="class",
            file=self.path,
            parent=parent,
            start_line=cls_start,
            end_line=cls_end,
            signature=f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
            docstring=doc,
            bases=bases,
            chunk=primary_chunk,
        )
        self.symbols.append(sym)

        # Methods of the class (nested folded into each method)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._handle_function(child, parent=qualname, kind="method")
            elif isinstance(child, ast.ClassDef):
                self._handle_class(child, parent=qualname)

    def _handle_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent: str,
        kind: SymbolKind,
    ) -> None:
        local_name = f"{parent.split('.')[-1]}.{node.name}" if kind == "method" else node.name
        qualname = f"{parent}.{node.name}" if parent else node.name
        start_line, end_line = _span(node)
        doc = ast.get_docstring(node)

        # Calls inside this function (nested defs excluded from caller's calls)
        calls_set = set()
        for call in _own_calls(node):
            callee_str = _callee(call.func)
            if callee_str and callee_str != "?.":
                calls_set.add(callee_str)
        calls = tuple(sorted(calls_set))

        # Make chunks (nested folded into parent span)
        fn_chunks = _make_windowed_chunks(
            self.lines,
            self.path,
            start_line,
            end_line,
            symbol=local_name,
            commit=self.commit,
            max_chunk_lines=self.max_chunk_lines,
        )
        self.chunks.extend(fn_chunks)
        primary_chunk = fn_chunks[0] if fn_chunks else None
        for c in fn_chunks:
            self._covered_lines.update(range(c.start_line, c.end_line + 1))

        args_str = ast.unparse(node.args) if hasattr(node, "args") else ""
        sig = f"def {node.name}({args_str})"

        sym = Symbol(
            qualname=qualname,
            local_name=local_name,
            kind=kind,
            file=self.path,
            parent=parent,
            start_line=start_line,
            end_line=end_line,
            signature=sig,
            docstring=doc,
            calls=calls,
            chunk=primary_chunk,
        )
        self.symbols.append(sym)

        # Collect nested functions as Symbols (chunk references enclosing function chunk)
        self._collect_nested_symbols(node, parent=qualname, enclosing_chunk=primary_chunk)

    def _collect_nested_symbols(
        self, parent_node: ast.AST, parent: str, enclosing_chunk: Optional[Chunk]
    ) -> None:
        for child in parent_node.body if hasattr(parent_node, "body") else []:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_l, end_l = _span(child)
                nested_local = f"{parent.split('.')[-1]}.{child.name}"
                nested_qual = f"{parent}.{child.name}"
                calls_set = set()
                for call in _own_calls(child):
                    c_str = _callee(call.func)
                    if c_str and c_str != "?.":
                        calls_set.add(c_str)
                args_str = ast.unparse(child.args) if hasattr(child, "args") else ""
                self.symbols.append(
                    Symbol(
                        qualname=nested_qual,
                        local_name=nested_local,
                        kind="function",
                        file=self.path,
                        parent=parent,
                        start_line=start_l,
                        end_line=end_l,
                        signature=f"def {child.name}({args_str})",
                        docstring=ast.get_docstring(child),
                        calls=tuple(sorted(calls_set)),
                        chunk=enclosing_chunk,
                    )
                )
                self._collect_nested_symbols(child, nested_qual, enclosing_chunk)

    def _collect_residuals(self) -> None:
        """Create residual chunks for uncovered module-level line runs."""
        total_lines = len(self.lines)
        curr = 1
        while curr <= total_lines:
            if curr in self._covered_lines:
                curr += 1
                continue
            run_start = curr
            while curr <= total_lines and curr not in self._covered_lines:
                curr += 1
            run_end = curr - 1
            res_chunks = _make_windowed_chunks(
                self.lines,
                self.path,
                run_start,
                run_end,
                symbol=None,
                commit=self.commit,
                max_chunk_lines=self.max_chunk_lines,
            )
            self.chunks.extend(res_chunks)


def _module_name(relative_path: str, is_package: bool) -> str:
    module = relative_path.removesuffix(".py").replace("/", ".")
    if is_package:
        module = module.removesuffix(".__init__")
        if module == "__init__":
            return ""
    return module


def parse_file(
    path: Path,
    root: Path,
    commit: str,
    max_chunk_lines: int = 80,
) -> ParsedFile:
    if not commit:
        raise ValueError("commit must be provided when creating indexed chunks")
    source = path.read_text(encoding="utf-8-sig")
    relative_path = path.relative_to(root).as_posix()
    is_package = path.name == "__init__.py"
    module = _module_name(relative_path, is_package)

    parser = _FileParser(
        module=module,
        path=relative_path,
        source=source,
        commit=commit,
        max_chunk_lines=max_chunk_lines,
    )
    tree = ast.parse(source, filename=relative_path)
    parser.parse(tree)

    return ParsedFile(
        path=relative_path,
        module=module,
        sha256=hashlib.sha256(source.encode()).hexdigest(),
        commit=commit,
        is_package=is_package,
        symbols=parser.symbols,
        imports=parser.imports,
        chunks=parser.chunks,
    )


def parse_repo(
    root: Path | str,
    commit: Optional[str] = None,
    max_chunk_lines: int = 80,
) -> list[ParsedFile]:
    root_path = Path(root)
    if commit is None:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root_path),
                capture_output=True,
                text=True,
                check=True,
            )
            commit = res.stdout.strip()
        except Exception:
            commit = "WORKTREE"

    parsed_files: list[ParsedFile] = []
    for path in sorted(root_path.rglob("*.py")):
        if IGNORED_DIRS & set(path.relative_to(root_path).parts):
            continue
        try:
            parsed_files.append(
                parse_file(
                    path,
                    root_path,
                    commit,
                    max_chunk_lines=max_chunk_lines,
                )
            )
        except (SyntaxError, UnicodeDecodeError) as exc:
            logger.warning("skipped %s: %s", path.relative_to(root_path), exc)
    return parsed_files
