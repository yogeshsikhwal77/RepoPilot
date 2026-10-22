"""Retriever specialist agent for hybrid code search and graph expansion."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Sequence, Set, Union

import yaml
from pydantic import BaseModel, Field

from repopilot.agents.base import AgentConfig, BaseAgent, Usage, load_agent_config
from repopilot.indexing.symbol_graph import SymbolGraph
from repopilot.orchestrator.state import AgentState, Evidence
from repopilot.retrieval.graph_expand import ExpandConfig, expand
from repopilot.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class RetrievalConfig(BaseModel):
    """Configuration settings for retrieval pipelines."""

    top_k: int = 10
    hop_limit: int = 2
    candidate_k: int = 50
    max_queries: int = 8
    w_lexical: float = 1.0
    w_dense: float = 1.0
    rrf_k: int = 60

    @classmethod
    def from_yaml(cls, path: Union[str, Path] = "configs/retrieval.yaml") -> RetrievalConfig:
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cls()
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            ret_sec = data.get("retrieval", data)
            g_sec = data.get("graph", {})
            w = ret_sec.get("fusion_weights", ret_sec.get("weights", {}))

            top_k = ret_sec.get("final_top_k", ret_sec.get("top_k", 10))
            hop_limit = g_sec.get("hop_limit", g_sec.get("max_hops", 2))
            rrf_k = ret_sec.get("fusion_k", ret_sec.get("rrf_k", 60))
            w_lex = w.get("bm25", w.get("lexical", 1.0))
            w_dense = w.get("vector", w.get("dense", 1.0))

            return cls(
                top_k=top_k,
                hop_limit=hop_limit,
                rrf_k=rrf_k,
                w_lexical=w_lex,
                w_dense=w_dense,
            )
        except Exception as exc:
            logger.warning("Failed to parse retrieval config from %s: %s", cfg_path, exc)
            return cls()


def extract_queries(text: str, max_queries: int = 8) -> list[str]:
    """Deterministically extract search queries from input task text."""
    queries: list[str] = []
    seen: set[str] = set()

    def add_q(q: str) -> None:
        clean = q.strip().strip(",.;:()[]{}")
        if clean and len(clean) >= 2 and clean not in seen:
            seen.add(clean)
            queries.append(clean)

    # 1. File paths (e.g. calculator.py, src/app.py)
    for m in re.findall(r"[\w\-./\\]+\.(?:py|json|md|txt|yaml|yml)\b", text):
        add_q(m)

    # 2. Exception and Error types (e.g. ZeroDivisionError, TypeError)
    for m in re.findall(r"\b[A-Z][a-zA-Z]*(?:Error|Exception|Warning|Fault)\b", text):
        add_q(m)

    # 3. Dotted identifiers (e.g. Cart.add, shop.pricing.calculate)
    for m in re.findall(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w+)+\b", text):
        add_q(m)

    # 4. Quoted strings and code identifiers (e.g. `divide`, "apply_discount")
    for m in re.findall(r"[`'\"]([^`'\"]+)[`'\"]", text):
        add_q(m)

    # 5. CamelCase names (e.g. SymbolGraph, HybridRetriever)
    for m in re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b", text):
        add_q(m)

    # 6. Snake_case identifiers (e.g. run_batch, test_calc)
    for m in re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text):
        add_q(m)

    # 7. Truncated raw sentence snippet
    first_sentence = text.split("\n")[0].split(". ")[0].strip()
    if first_sentence:
        add_q(first_sentence[:100])

    return queries[:max_queries]


class RetrieverAgent(BaseAgent):
    """Retriever agent finding relevant source code via hybrid search and graph expansion."""

    name: ClassVar[str] = "retriever"
    writes: ClassVar[frozenset[str]] = frozenset({"collected_evidence"})

    def __init__(
        self,
        hybrid: HybridRetriever,
        graph: SymbolGraph,
        cfg: Optional[RetrievalConfig] = None,
        llm: Optional[Any] = None,
        use_graph: bool = True,
        agent_config: Optional[AgentConfig] = None,
    ) -> None:
        a_cfg = agent_config or load_agent_config(self.name)
        super().__init__(config=a_cfg)
        self.hybrid = hybrid
        self.graph = graph
        self.cfg = cfg or RetrievalConfig.from_yaml()
        self.llm = llm
        self.use_graph = use_graph

    def _run(self, state: AgentState, usage: Usage) -> dict[str, Any]:
        self.step()
        queries = extract_queries(state.task_input, max_queries=self.cfg.max_queries)

        # Optional LLM query rewrite
        if self.llm is not None:
            try:
                rewrite_prompt = (
                    "Given the following issue description, list 1 to 3 targeted code search keywords "
                    f"separated by commas:\n\n{state.task_input}"
                )
                resp = self.llm.generate(
                    model=self.config.model,
                    prompt=rewrite_prompt,
                    temperature=0.0,
                    max_tokens=64,
                )
                # Usage accounting
                p_tok = resp.get("prompt_tokens", 0) if hasattr(resp, "get") else getattr(resp, "prompt_tokens", 0)
                c_tok = resp.get("completion_tokens", 0) if hasattr(resp, "get") else getattr(resp, "completion_tokens", 0)
                cost = resp.get("cost_usd", 0.0) if hasattr(resp, "get") else getattr(resp, "cost_usd", 0.0)
                usage.add(Usage(tokens=p_tok + c_tok, cost_usd=cost, tool_calls=1))

                text = resp.get("text", "") if hasattr(resp, "get") else getattr(resp, "text", str(resp))
                llm_queries = [q.strip() for q in text.split(",") if q.strip()]
                for q in llm_queries:
                    if q and q not in queries:
                        queries.append(q)
            except Exception as exc:
                logger.warning("LLM query rewrite failed; falling back to deterministic extraction: %s", exc)

        # Hybrid search across extracted queries
        seeds = self.hybrid.search_many(queries, k=self.cfg.top_k)

        # Optional graph expansion
        expanded: list[Evidence] = []
        if self.use_graph and seeds:
            expanded = expand(
                self.graph,
                seeds,
                cfg=ExpandConfig(hops=self.cfg.hop_limit),
            )

        # Combine candidates and deduplicate
        all_candidates = seeds + expanded
        existing_cids = {ev.chunk_id for ev in state.collected_evidence}

        new_evidence: list[Evidence] = []
        seen: set[str] = set(existing_cids)
        for ev in all_candidates:
            if ev.chunk_id not in seen:
                seen.add(ev.chunk_id)
                new_evidence.append(ev)

        self._partial_delta = {"collected_evidence": new_evidence}
        return {"collected_evidence": new_evidence}
