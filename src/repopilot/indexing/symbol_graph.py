"""Symbol graph: containment, import, inheritance and call edges over parsed symbols.

Resolution is static and name-based (no type inference). Every edge carries a confidence so
consumers can trade precision for recall.

Nodes are module-qualified names (ParsedSymbol.qualname). Chunk.symbol stays local to its
file, as defined in orchestrator/state.py.

Bridge to coverage_map.py: symbol_id_of / qualname_of / to_symbol_ids convert between graph
nodes and the "path::Class.method" ids used by CoverageMap, so impact.py can do
    graph.neighbors(q, direction="in") -> to_symbol_ids() -> CoverageMap.tests_for()
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Literal

from repopilot.orchestrator.state import Chunk
from repopilot.indexing.coverage_map import symbol_id
from repopilot.indexing.parser import ImportRef, ParsedFile, ParsedSymbol

Confidence = Literal["exact", "heuristic", "ambiguous"]


class EdgeKind(str, Enum):
    CONTAINS = "contains"
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"


CONFIDENCE_RANK: dict[str, int] = {"exact": 0, "heuristic": 1, "ambiguous": 2}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: EdgeKind
    confidence: Confidence = "exact"


class SymbolGraph:
    def __init__(self, files: Iterable[ParsedFile], *, include_heuristic: bool = True,
                 max_ambiguity: int = 4):
        files = list(files)
        self.files = {pf.path: pf for pf in files}
        self.symbols: dict[str, ParsedSymbol] = {
            symbol.qualname: symbol for parsed_file in files for symbol in parsed_file.symbols
        }
        self.modules: dict[str, ParsedFile] = {pf.module: pf for pf in files}
        # coverage_map ids: "path::Class.method"
        self._qual_to_id: dict[str, str] = {
            q: symbol_id(s.chunk.path, s.chunk.symbol or "") for q, s in self.symbols.items()
        }
        self._id_to_qual: dict[str, str] = {i: q for q, i in self._qual_to_id.items()}
        self.include_heuristic, self.max_ambiguity = include_heuristic, max_ambiguity
        self.unresolved_calls = 0      # call refs resolving to nothing (builtins/external/dynamic)
        self.out_edges: dict[str, list[Edge]] = defaultdict(list)
        self.in_edges: dict[str, list[Edge]] = defaultdict(list)
        self._edges: dict[tuple[str, str, EdgeKind], Edge] = {}
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
    def build(cls, files: Iterable[ParsedFile], **kw) -> "SymbolGraph":
        return cls(files, **kw)

    # ---------- indexing ----------
    def _build_module_index(self) -> dict[str, str | None]:
        """Map every dotted suffix of a module name to its canonical name (None if ambiguous).
        Lets `import repopilot.x` match a file parsed as `src.repopilot.x`."""
        idx: dict[str, str | None] = {}
        for pf in self.files.values():
            parts = pf.module.split(".")
            for i in range(1, len(parts)):
                suf = ".".join(parts[i:])
                idx[suf] = pf.module if idx.get(suf, pf.module) == pf.module else None
        for m in self.modules:                       # full names always win
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
            for import_ref in pf.imports:
                scope[
                    import_ref.alias
                    or import_ref.name
                    or import_ref.module.split(".")[0]
                ].append(import_ref)
            self._scopes[pf.path] = scope
            for symbol in pf.symbols:
                if symbol.kind == "method":
                    names = self._methods_by_name[symbol.qualname.rpartition(".")[2]]
                    if symbol.qualname not in names:          # redefinitions / setters
                        names.append(symbol.qualname)

    def _absolute_module(self, pf: ParsedFile, ref: ImportRef) -> str:
        if ref.level == 0:
            return ref.module
        pkg = pf.module if pf.is_package else pf.module.rpartition(".")[0]
        for _ in range(ref.level - 1):
            pkg = pkg.rpartition(".")[0]
        return ".".join(filter(None, [pkg, ref.module]))

    def _is_external(self, pf: ParsedFile, name: str) -> bool:
        """True if the head of a dotted reference is imported from outside the repo."""
        refs = self._scopes[pf.path].get(name.split(".")[0], ())
        if not refs:
            return False
        return not any(
            r.level > 0 or self._is_repo_name(self._absolute_module(pf, r)) for r in refs
        )

    # ---------- resolution ----------
    def _lookup(self, pf: ParsedFile, name: str) -> str | None:
        """Resolve a dotted reference visible from pf's module to a symbol qualname."""
        parts = name.split(".")
        if (q := ".".join(filter(None, [pf.module, name]))) in self.symbols:
            return q
        for ref in self._scopes[pf.path].get(parts[0], ()):
            if ref.name is not None:                               # from m import n [as k]
                base = self._module(self._absolute_module(pf, ref))
                cands = [".".join([base, ref.name, *parts[1:]])] if base else []
            elif ref.alias:                                        # import m as k
                base = self._module(ref.module)
                cands = [".".join([base, *parts[1:]])] if base and len(parts) > 1 else []
            else:                                                  # import a.b.c
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
        pf = self.files[cls_sym.chunk.path]
        for b in cls_sym.bases:
            bq = self._lookup(pf, b)
            if bq and self.symbols[bq].kind == "class" and (r := self._method(bq, name, seen)):
                return r
        return None

    def _resolve_call(self, pf: ParsedFile, caller: ParsedSymbol,
                      callee: str) -> list[tuple[str, Confidence]]:
        parts = callee.split(".")
        if parts[0] in ("self", "cls") and len(parts) == 2:
            parent = caller.qualname.rpartition(".")[0]
            if parent in self.symbols and self.symbols[parent].kind == "class":
                if m := self._method(parent, parts[1]):
                    return [(m, "exact")]
        if q := self._lookup(pf, callee):
            if self.symbols[q].kind == "class" and f"{q}.__init__" in self.symbols:
                q = f"{q}.__init__"
            return [(q, "exact")]
        if self.include_heuristic and len(parts) >= 2:
            if self._is_external(pf, callee):        # json.dumps, os.path.join, ...
                return []
            cands = self._methods_by_name.get(parts[-1], [])
            if 1 <= len(cands) <= self.max_ambiguity:
                confidence: Confidence = "heuristic" if len(cands) == 1 else "ambiguous"
                return [(c, confidence) for c in cands]
        return []

    # ---------- linking ----------
    def _add(self, src: str, dst: str, kind: EdgeKind, confidence: Confidence = "exact") -> None:
        old = self._edges.get((src, dst, kind))
        if old is None or CONFIDENCE_RANK[confidence] < CONFIDENCE_RANK[old.confidence]:
            self._edges[(src, dst, kind)] = Edge(src, dst, kind, confidence)

    def _link(self) -> None:
        for pf in self.files.values():
            for ref in pf.imports:
                base = self._absolute_module(pf, ref)
                target = (ref.name and self._module(".".join(filter(None, [base, ref.name])))) \
                    or self._module(base)
                if target and target != pf.module:
                    self._add(pf.module, target, EdgeKind.IMPORTS)
            for symbol in pf.symbols:
                qualname = symbol.qualname
                if parent := qualname.rpartition(".")[0]:
                    self._add(parent, qualname, EdgeKind.CONTAINS)
                for base_name in symbol.bases:
                    bq = self._lookup(pf, base_name)
                    if bq and self.symbols[bq].kind == "class":
                        self._add(qualname, bq, EdgeKind.INHERITS)
                for callee in symbol.calls:
                    targets = self._resolve_call(pf, symbol, callee)
                    if not targets:
                        self.unresolved_calls += 1
                    for dst, confidence in targets:
                        if dst != qualname:
                            self._add(qualname, dst, EdgeKind.CALLS, confidence)

    # ---------- queries ----------
    def edges(self, kind: EdgeKind | None = None) -> list[Edge]:
        return [e for e in self._edges.values() if kind is None or e.kind is kind]

    def callers(self, q: str, worst: Confidence = "heuristic") -> list[Edge]:
        return [e for e in self.in_edges.get(q, ())
                if e.kind is EdgeKind.CALLS and CONFIDENCE_RANK[e.confidence] <= CONFIDENCE_RANK[worst]]

    def callees(self, q: str, worst: Confidence = "heuristic") -> list[Edge]:
        return [e for e in self.out_edges.get(q, ())
                if e.kind is EdgeKind.CALLS and CONFIDENCE_RANK[e.confidence] <= CONFIDENCE_RANK[worst]]

    def lookup(self, name: str) -> list[Chunk]:
        """Match by full qualname or any dotted suffix, e.g. "Cart.add_item"."""
        return [
            symbol.chunk for qualname, symbol in self.symbols.items()
            if qualname == name or qualname.endswith("." + name)
        ]

    def neighbors(self, node: str, hops: int = 2, kinds: tuple[EdgeKind, ...] = (EdgeKind.CALLS,),
                  direction: str = "both", worst: Confidence = "heuristic") -> dict[str, int]:
        """BFS from `node`; returns {reached node: hop distance}."""
        limit = CONFIDENCE_RANK[worst]
        dist, queue = {node: 0}, deque([node])
        while queue:
            n = queue.popleft()
            if dist[n] == hops:
                continue
            nxt: list[str] = []
            if direction in ("out", "both"):
                nxt += [e.dst for e in self.out_edges.get(n, ())
                        if e.kind in kinds and CONFIDENCE_RANK[e.confidence] <= limit]
            if direction in ("in", "both"):
                nxt += [e.src for e in self.in_edges.get(n, ())
                        if e.kind in kinds and CONFIDENCE_RANK[e.confidence] <= limit]
            for m in nxt:
                if m not in dist:
                    dist[m] = dist[n] + 1
                    queue.append(m)
        del dist[node]
        return dist

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

    def stats(self) -> dict:
        by = Counter((e.kind.value, e.confidence) for e in self._edges.values())
        return {"modules": len(self.modules), "symbols": len(self.symbols),
                "edges": {f"{k}/{c}": n for (k, c), n in sorted(by.items())},
                "unresolved_calls": self.unresolved_calls}