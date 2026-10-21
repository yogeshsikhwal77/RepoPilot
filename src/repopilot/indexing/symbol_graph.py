"""Symbol graph: containment, import, inheritance and call edges over parsed symbols.

Resolution is static and name-based (no type inference). Every edge carries a confidence so
consumers can trade precision for recall.

Nodes are module-qualified names (Symbol.qualname). Chunk.symbol stays local to its
file, as defined in orchestrator/state.py.

Bridge to coverage_map.py: symbol_id_of / qualname_of / to_symbol_ids convert between graph
nodes and the "path::Class.method" ids used by CoverageMap, so impact.py can do
    graph.neighbors(q, direction="in") -> to_symbol_ids() -> CoverageMap.tests_for()
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from enum import Enum
from typing import Any, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from repopilot.indexing.coverage_map import symbol_id
from repopilot.indexing.parser import ImportRef, ParsedFile, Symbol
from repopilot.orchestrator.state import Chunk, Citation

Confidence = Literal["exact", "heuristic", "ambiguous"]


class EdgeKind(str, Enum):
    CONTAINS = "contains"
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"


CONFIDENCE_RANK: dict[str, int] = {"exact": 0, "heuristic": 1, "ambiguous": 2}


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    src: str
    dst: str
    kind: Literal["contains", "calls", "imports", "inherits"]
    conf: Literal["exact", "heuristic", "ambiguous"] = "exact"

    @property
    def confidence(self) -> str:
        """Backward-compatibility property matching old .confidence attribute."""
        return self.conf


class SymbolGraph:
    def __init__(
        self,
        files: Iterable[ParsedFile],
        *,
        include_heuristic: bool = True,
        max_ambiguity: int = 4,
    ) -> None:
        files = list(files)
        self.files: dict[str, ParsedFile] = {pf.path: pf for pf in files}
        self.symbols: dict[str, Symbol] = {
            symbol.qualname: symbol
            for parsed_file in files
            for symbol in parsed_file.symbols
        }
        self.modules: dict[str, ParsedFile] = {pf.module: pf for pf in files}

        # coverage_map ids: "path::Class.method"
        self._qual_to_id: dict[str, str] = {
            q: symbol_id(s.file, s.local_name)
            for q, s in self.symbols.items()
        }
        self._id_to_qual: dict[str, str] = {i: q for q, i in self._qual_to_id.items()}

        self.include_heuristic = include_heuristic
        self.max_ambiguity = max_ambiguity
        self.unresolved_calls = 0

        self.out_edges: dict[str, list[Edge]] = defaultdict(list)
        self.in_edges: dict[str, list[Edge]] = defaultdict(list)
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._scopes: dict[str, dict[str, list[ImportRef]]] = {}
        self._methods_by_name: dict[str, list[str]] = defaultdict(list)
        self._repo_name_cache: dict[str, bool] = {}
        self._mod_index = self._build_module_index()

        self._index()
        self._link()
        for e in self._edges.values():
            self.out_edges[e.src].append(e)
            self.in_edges[e.dst].append(e)

    @classmethod
    def build(cls, files: Iterable[ParsedFile], **kw: Any) -> SymbolGraph:
        return cls(files, **kw)

    # ---------- indexing ----------
    def _build_module_index(self) -> dict[str, str | None]:
        """Map every dotted suffix of a module name to its canonical name (None if ambiguous)."""
        idx: dict[str, str | None] = {}
        for pf in self.files.values():
            parts = pf.module.split(".")
            for i in range(1, len(parts)):
                suf = ".".join(parts[i:])
                idx[suf] = pf.module if idx.get(suf, pf.module) == pf.module else None
        for m in self.modules:
            idx[m] = m
        return idx

    def _module(self, name: str) -> str | None:
        return self._mod_index.get(name)

    def _is_repo_name(self, name: str) -> bool:
        """True if `name` is a repo module/package, or a prefix of one."""
        hit = self._repo_name_cache.get(name)
        if hit is None:
            hit = bool(name) and any(
                k == name or k.startswith(name + ".") for k in self._mod_index
            )
            self._repo_name_cache[name] = hit
        return hit

    def _index(self) -> None:
        for pf in self.files.values():
            scope: dict[str, list[ImportRef]] = defaultdict(list)
            for ref in pf.imports:
                key = ref.alias or ref.name or ref.module.split(".")[0]
                scope[key].append(ref)
            self._scopes[pf.path] = scope
            for symbol in pf.symbols:
                if symbol.kind == "method":
                    self._methods_by_name[symbol.local_name.split(".")[-1]].append(
                        symbol.qualname
                    )

    # ---------- resolution ----------
    def _absolute_module(self, pf: ParsedFile, ref: ImportRef) -> str:
        """Turn a possibly relative import into an absolute dotted module name."""
        if not ref.level:
            return ref.module
        pkg = pf.module.split(".") if pf.is_package else pf.module.split(".")[:-1]
        base = pkg[: len(pkg) - (ref.level - 1)] if ref.level <= len(pkg) + 1 else []
        return ".".join(filter(None, [*base, ref.module]))

    def _is_external(self, pf: ParsedFile, callee: str) -> bool:
        head = callee.split(".")[0]
        if head in ("self", "cls"):
            return False
        for ref in self._scopes[pf.path].get(head, ()):
            if not ref.level and not self._is_repo_name(ref.module.split(".")[0]):
                return True
        return False

    def _lookup(self, pf: ParsedFile, name: str) -> str | None:
        """Resolve a dotted reference visible from pf's module to a symbol qualname."""
        parts = name.split(".")
        if (q := ".".join(filter(None, [pf.module, name]))) in self.symbols:
            return q
        for ref in self._scopes[pf.path].get(parts[0], ()):
            if ref.name is not None:
                base = self._module(self._absolute_module(pf, ref))
                cands = [".".join([base, ref.name, *parts[1:]])] if base else []
            elif ref.alias:
                base = self._module(ref.module)
                cands = [".".join([base, *parts[1:]])] if base and len(parts) > 1 else []
            else:
                cands = []
                for i in range(len(parts) - 1, 0, -1):
                    if base := self._module(".".join(parts[:i])):
                        cands.append(".".join([base, *parts[i:]]))
            for c in cands:
                if c in self.symbols:
                    return c
        return None

    def _method(self, cls_q: str, name: str, seen: set[str] | None = None) -> str | None:
        """Find `name` on a class or its resolved bases (depth-first MRO approximation)."""
        seen = seen if seen is not None else set()
        if cls_q in seen:
            return None
        seen.add(cls_q)
        if (q := f"{cls_q}.{name}") in self.symbols:
            return q
        cls_sym = self.symbols[cls_q]
        pf = self.files[cls_sym.file]
        for b in cls_sym.bases:
            bq = self._lookup(pf, b)
            if bq and self.symbols[bq].kind == "class" and (r := self._method(bq, name, seen)):
                return r
        return None

    def _find_enclosing_class(self, qualname: str) -> str | None:
        """Climb out of functions/closures to find the enclosing class qualname."""
        curr = qualname
        while "." in curr:
            curr = curr.rpartition(".")[0]
            if curr in self.symbols and self.symbols[curr].kind == "class":
                return curr
        return None

    def _resolve_call(
        self, pf: ParsedFile, caller: Symbol, callee: str
    ) -> list[tuple[str, Confidence]]:
        parts = callee.split(".")
        # Ladder step 1: self. / cls. via enclosing class (climb out of closures)
        if parts[0] in ("self", "cls") and len(parts) == 2:
            cls_q = self._find_enclosing_class(caller.qualname)
            if cls_q:
                if m := self._method(cls_q, parts[1]):
                    return [(m, "exact")]

        # Ladder step 2: names via module scope & import scope; class call -> __init__
        clean_callee = callee[2:] if callee.startswith("?.") else callee
        if q := self._lookup(pf, clean_callee):
            if self.symbols[q].kind == "class" and f"{q}.__init__" in self.symbols:
                q = f"{q}.__init__"
            return [(q, "exact")]

        # Ladder step 3: unresolved dotted call -> same-named methods
        if self.include_heuristic and len(parts) >= 1:
            method_name = parts[-1]
            if not self._is_external(pf, callee):
                cands = self._methods_by_name.get(method_name, [])
                if 1 <= len(cands) <= self.max_ambiguity:
                    conf: Confidence = "heuristic" if len(cands) == 1 else "ambiguous"
                    return [(c, conf) for c in cands]

        return []

    # ---------- linking ----------
    def _add(self, src: str, dst: str, kind: str, confidence: Confidence = "exact") -> None:
        old = self._edges.get((src, dst, kind))
        if old is None or CONFIDENCE_RANK[confidence] < CONFIDENCE_RANK[old.conf]:
            self._edges[(src, dst, kind)] = Edge(src=src, dst=dst, kind=kind, conf=confidence)

    def _link(self) -> None:
        for pf in self.files.values():
            for ref in pf.imports:
                base = self._absolute_module(pf, ref)
                target = (
                    ref.name and self._module(".".join(filter(None, [base, ref.name])))
                ) or self._module(base)
                if target and target != pf.module:
                    self._add(pf.module, target, "imports", "exact")
            for symbol in pf.symbols:
                qualname = symbol.qualname
                if parent := qualname.rpartition(".")[0]:
                    self._add(parent, qualname, "contains", "exact")
                for base_name in symbol.bases:
                    bq = self._lookup(pf, base_name)
                    if bq and self.symbols[bq].kind == "class":
                        self._add(qualname, bq, "inherits", "exact")
                for callee in symbol.calls:
                    targets = self._resolve_call(pf, symbol, callee)
                    if not targets:
                        self.unresolved_calls += 1
                    for dst, confidence in targets:
                        if dst != qualname:
                            self._add(qualname, dst, "calls", confidence)

    # ---------- queries ----------
    def edges(self, kind: Optional[str] = None) -> list[Edge]:
        return [e for e in self._edges.values() if kind is None or e.kind == kind]

    def callers(self, q: str, worst: Confidence = "heuristic") -> list[Edge]:
        return [
            e
            for e in self.in_edges.get(q, ())
            if e.kind == "calls" and CONFIDENCE_RANK[e.conf] <= CONFIDENCE_RANK[worst]
        ]

    def callees(self, q: str, worst: Confidence = "heuristic") -> list[Edge]:
        return [
            e
            for e in self.out_edges.get(q, ())
            if e.kind == "calls" and CONFIDENCE_RANK[e.conf] <= CONFIDENCE_RANK[worst]
        ]

    def lookup(self, name: str) -> list[Chunk]:
        """Match by full qualname or any dotted suffix, e.g. 'Cart.add_item'."""
        matching_chunks: list[Chunk] = []
        for qualname, symbol in self.symbols.items():
            if qualname == name or qualname.endswith("." + name):
                matching_chunks.extend(self.chunks_for(qualname))
        return matching_chunks

    def neighbors(
        self,
        node: str,
        hops: int = 2,
        kinds: tuple[str, ...] = ("calls",),
        direction: str = "both",
        worst: Confidence = "heuristic",
    ) -> dict[str, int]:
        """BFS from `node`; returns {reached node: hop distance}."""
        limit = CONFIDENCE_RANK[worst]
        dist: dict[str, int] = {node: 0}
        queue = deque([node])
        while queue:
            n = queue.popleft()
            if dist[n] == hops:
                continue
            nxt: list[str] = []
            if direction in ("out", "both"):
                nxt += [
                    e.dst
                    for e in self.out_edges.get(n, ())
                    if e.kind in kinds and CONFIDENCE_RANK[e.conf] <= limit
                ]
            if direction in ("in", "both"):
                nxt += [
                    e.src
                    for e in self.in_edges.get(n, ())
                    if e.kind in kinds and CONFIDENCE_RANK[e.conf] <= limit
                ]
            for m in nxt:
                if m not in dist:
                    dist[m] = dist[n] + 1
                    queue.append(m)
        del dist[node]
        return dist

    def chunks_for(self, qualname: str) -> list[Chunk]:
        """Return chunks for a symbol."""
        sym = self.symbols.get(qualname)
        if not sym:
            return []
        pf = self.files.get(sym.file)
        if not pf:
            return [sym.chunk] if sym.chunk else []
        # Return all chunks matching local symbol name
        res = [c for c in pf.chunks if c.symbol == sym.local_name]
        return res if res else ([sym.chunk] if sym.chunk else [])

    def citation(self, qualname: str) -> Citation:
        """Citation for a symbol's exact location."""
        sym = self.symbols[qualname]
        pf = self.files[sym.file]
        return Citation(
            path=sym.file,
            commit=pf.commit,
            start_line=sym.start_line,
            end_line=sym.end_line,
        )

    def symbol_at(self, path: str, line: int) -> Optional[str]:
        """Find the innermost symbol covering `line` in file `path`."""
        matching: list[Symbol] = []
        for s in self.symbols.values():
            if s.file == path and s.start_line <= line <= s.end_line:
                matching.append(s)
        if not matching:
            return None
        # Innermost: smallest line span
        innermost = min(matching, key=lambda s: (s.end_line - s.start_line + 1, s.start_line))
        return innermost.qualname

    # ---------- bridge to coverage_map ids ("path::Class.method") ----------
    def symbol_id_of(self, qualname: str) -> str | None:
        """Coverage-map id for a graph node; None for modules and unknown names."""
        return self._qual_to_id.get(qualname)

    def qualname_of(self, sid: str) -> str | None:
        """Graph node for a coverage-map id; None if unknown."""
        return self._id_to_qual.get(sid)

    def to_symbol_ids(self, qualnames: Iterable[str]) -> list[str]:
        """Sorted, de-duplicated coverage-map ids. Module nodes are dropped."""
        return sorted({i for q in qualnames if (i := self._qual_to_id.get(q))})

    def stats(self) -> dict[str, Any]:
        by = Counter((e.kind, e.conf) for e in self._edges.values())
        return {
            "modules": len(self.modules),
            "symbols": len(self.symbols),
            "edges": {f"{k}/{c}": n for (k, c), n in sorted(by.items())},
            "unresolved_calls": self.unresolved_calls,
        }
