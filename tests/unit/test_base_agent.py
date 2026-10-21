import pytest

from repopilot.agents.base import (
    AgentConfig,
    BaseAgent,
    ContractError,
    StepLimitExceeded,
    Usage,
    load_agent_config,
)
from repopilot.orchestrator.state import AgentState, Evidence, Citation


class DummyValidAgent(BaseAgent):
    name = "dummy_valid"
    writes = frozenset({"collected_evidence"})

    def _run(self, state: AgentState, usage: Usage) -> dict:
        self.step()
        usage.add(Usage(tokens=50, cost_usd=0.001, tool_calls=2))
        ev = Evidence(
            chunk_id="f.py:1-2",
            citation=Citation(path="f.py", commit="c1", start_line=1, end_line=2),
            content="print(1)",
            source_type="code",
            retrieval="bm25",
            score=1.0,
        )
        return {"collected_evidence": [ev]}


class DummyInvalidKeyAgent(BaseAgent):
    name = "dummy_invalid_key"
    writes = frozenset({"collected_evidence"})

    def _run(self, state: AgentState, usage: Usage) -> dict:
        return {"non_existent_state_field": 123}


class DummyUndeclaredAgent(BaseAgent):
    name = "dummy_undeclared"
    writes = frozenset({"collected_evidence"})

    def _run(self, state: AgentState, usage: Usage) -> dict:
        # draft_patches is a valid AgentState field, but not in `writes`
        return {"draft_patches": []}


class DummyNonListDeltaAgent(BaseAgent):
    name = "dummy_non_list"
    writes = frozenset({"collected_evidence"})

    def _run(self, state: AgentState, usage: Usage) -> dict:
        return {"collected_evidence": "not_a_list"}


class DummyStepLimitAgent(BaseAgent):
    name = "dummy_step_limit"
    writes = frozenset({"collected_evidence"})

    def _run(self, state: AgentState, usage: Usage) -> dict:
        self._partial_delta = {"collected_evidence": []}
        usage.add(Usage(tokens=10, cost_usd=0.0001, tool_calls=1))
        self.step()  # step 1 (limit is 1)
        self.step()  # step 2 -> raises StepLimitExceeded
        return {"collected_evidence": ["should_not_reach"]}


def test_agent_valid_run_and_usage_aggregation():
    state = AgentState(task_input="test task")
    agent = DummyValidAgent()
    delta = agent(state)

    assert "collected_evidence" in delta
    assert len(delta["collected_evidence"]) == 1
    assert delta["tokens_used"] == 50
    assert delta["cost_usd"] == 0.001
    assert delta["tool_calls"] == 2


def test_agent_rejects_invalid_state_key():
    state = AgentState(task_input="test task")
    agent = DummyInvalidKeyAgent()
    with pytest.raises(ContractError, match="not a valid AgentState field"):
        agent.run(state)


def test_agent_rejects_undeclared_writes():
    state = AgentState(task_input="test task")
    agent = DummyUndeclaredAgent()
    with pytest.raises(ContractError, match="not authorized to write to field 'draft_patches'"):
        agent.run(state)


def test_agent_rejects_non_list_reducer_delta():
    state = AgentState(task_input="test task")
    agent = DummyNonListDeltaAgent()
    with pytest.raises(ContractError, match="must be a list delta"):
        agent.run(state)


def test_agent_enforces_step_limit_and_returns_partial():
    cfg = AgentConfig(max_steps=1)
    agent = DummyStepLimitAgent(config=cfg)
    state = AgentState(task_input="test task")
    delta = agent.run(state)

    assert delta["collected_evidence"] == []
    assert delta["tokens_used"] == 10
    assert delta["cost_usd"] == 0.0001
    assert delta["tool_calls"] == 1


def test_load_agent_config(tmp_path):
    cfg_file = tmp_path / "agents.yaml"
    cfg_file.write_text("retriever:\n  model: custom-model\n  max_steps: 42\n", encoding="utf-8")
    cfg = load_agent_config("retriever", path=cfg_file)
    assert cfg.model == "custom-model"
    assert cfg.max_steps == 42

    # Missing agent falls back to defaults
    cfg_missing = load_agent_config("nonexistent", path=cfg_file)
    assert cfg_missing.model == "gpt-4o-mini"
    assert cfg_missing.max_steps == 10
