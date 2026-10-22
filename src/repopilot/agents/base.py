"""Base agent class with contract and budget enforcement for RepoPilot.

Enforces:
1. Typed hand-offs adhering strictly to AgentState fields.
2. Least-privilege write permissions (only declared `writes` fields allowed).
3. Reducer list validation (deltas must be lists).
4. Budget and Usage tracking (accumulated and added into deltas).
5. Max step limit enforcement with partial delta return.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import yaml
from pydantic import BaseModel, Field

from repopilot.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

BUDGET_COUNTERS = frozenset({"tokens_used", "cost_usd", "tool_calls"})
REDUCER_LIST_FIELDS = frozenset({"collected_evidence", "impact_set"})


class ContractError(Exception):
    """Raised when an agent violates its contract with AgentState."""


class StepLimitExceeded(Exception):
    """Raised when an agent attempts to execute more steps than its configured max_steps."""


class AgentConfig(BaseModel):
    model: str = "gpt-4o-mini"
    max_steps: int = 10
    temperature: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)


def load_agent_config(
    name: str,
    path: Union[str, Path] = "configs/agents.yaml",
) -> AgentConfig:
    """Load configuration for a named agent from YAML, falling back to defaults."""
    cfg_path = Path(path)
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and name in data:
                agent_data = data[name]
                if isinstance(agent_data, dict):
                    return AgentConfig(
                        model=agent_data.get("model", "gpt-4o-mini"),
                        max_steps=agent_data.get("max_steps", 10),
                        temperature=agent_data.get("temperature", 0.0),
                        extra=agent_data.get("extra", {}),
                    )
        except Exception as exc:
            logger.warning("Failed to load agent config from %s: %s", cfg_path, exc)
    return AgentConfig()


class Usage(BaseModel):
    tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0

    def add(self, other: Union[Usage, dict[str, Any]]) -> None:
        if isinstance(other, Usage):
            self.tokens += other.tokens
            self.cost_usd += other.cost_usd
            self.tool_calls += other.tool_calls
        elif isinstance(other, dict):
            self.tokens += other.get("tokens", other.get("tokens_used", 0))
            self.cost_usd += other.get("cost_usd", 0.0)
            self.tool_calls += other.get("tool_calls", 0)


class BaseAgent(ABC):
    name: ClassVar[str] = "base"
    writes: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.current_step: int = 0
        self._partial_delta: dict[str, Any] = {}

    def step(self) -> int:
        """Advance the step counter; raises StepLimitExceeded if max_steps is exceeded."""
        self.current_step += 1
        if self.current_step > self.config.max_steps:
            raise StepLimitExceeded(
                f"Agent '{self.name}' exceeded maximum steps of {self.config.max_steps}"
            )
        return self.current_step

    def trace(self, event: str, payload: Any = None) -> None:
        """No-op trace hook for observability."""
        pass

    def run(self, state: AgentState) -> dict[str, Any]:
        """Template method executing the agent with contract and budget validation."""
        self.current_step = 0
        self._partial_delta = {}
        usage = Usage()
        self.trace("start", {"state": state})

        try:
            delta = self._run(state, usage)
        except StepLimitExceeded:
            logger.warning("Agent '%s' reached step limit; returning partial delta.", self.name)
            delta = self._partial_delta

        self.trace("end", {"delta": delta, "usage": usage})

        # (a) Reject keys not in AgentState.model_fields or not in writes (budget counters always allowed)
        allowed_writes = self.writes | BUDGET_COUNTERS
        for key in delta:
            if key not in AgentState.model_fields:
                raise ContractError(f"Key '{key}' is not a valid AgentState field.")
            if key not in allowed_writes:
                raise ContractError(
                    f"Agent '{self.name}' is not authorized to write to field '{key}'. "
                    f"Allowed writes: {sorted(allowed_writes)}"
                )

        # (b) Require reducer list fields to be lists (deltas)
        for r_field in REDUCER_LIST_FIELDS:
            if r_field in delta and not isinstance(delta[r_field], list):
                raise ContractError(
                    f"Reducer field '{r_field}' must be a list delta, got {type(delta[r_field]).__name__}."
                )

        # (c) Merge Usage into budget counters (sum, never overwrite)
        result = dict(delta)
        result["tokens_used"] = result.get("tokens_used", 0) + usage.tokens
        result["cost_usd"] = round(result.get("cost_usd", 0.0) + usage.cost_usd, 6)
        result["tool_calls"] = result.get("tool_calls", 0) + usage.tool_calls

        return result

    @abstractmethod
    def _run(self, state: AgentState, usage: Usage) -> dict[str, Any]:
        """Abstract execution method implemented by specialist agents."""
        raise NotImplementedError

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """LangGraph node callable interface."""
        return self.run(state)
