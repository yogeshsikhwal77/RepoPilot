# tests/unit/test_router.py
from __future__ import annotations

from typing import Any

from repopilot.llm.router import LLMRouter, RouterConfig


class FakeLLMClient:
    def __init__(self, responses_by_model: dict[str, Any]) -> None:
        self.responses_by_model = responses_by_model
        self.calls: list[tuple[str, Any]] = []

    def complete(self, prompt: str, model: str = "openai/gpt-oss-20b", **kwargs: Any) -> Any:
        self.calls.append((model, prompt))
        handler = self.responses_by_model[model]
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(prompt)
        return handler


def test_router_uses_cheap_free_model_when_confident() -> None:
    client = FakeLLMClient(
        {
            "openai/gpt-oss-20b": {
                "content": '{"answer": "ok", "confidence": 0.92}',
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "openai/gpt-oss-120b": {
                "content": '{"answer": "strong", "confidence": 0.99}',
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        }
    )
    router = LLMRouter(client, config=RouterConfig._default())
    res = router.complete("Find bug", require_json=True)

    assert res.tier_used == "cheap"
    assert res.model_used == "openai/gpt-oss-20b"
    assert res.escalated is False
    assert res.attempts == 1
    assert len(client.calls) == 1


def test_router_escalates_on_low_confidence() -> None:
    client = FakeLLMClient(
        {
            "openai/gpt-oss-20b": {
                "content": '{"answer": "unsure", "confidence": 0.40}',
                "prompt_tokens": 100,
                "completion_tokens": 40,
            },
            "openai/gpt-oss-120b": {
                "content": '{"answer": "resolved", "confidence": 0.95}',
                "prompt_tokens": 100,
                "completion_tokens": 60,
            },
        }
    )
    router = LLMRouter(client, config=RouterConfig._default())
    res = router.complete("Complex issue", require_json=True)

    assert res.tier_used == "strong"
    assert res.model_used == "openai/gpt-oss-120b"
    assert res.escalated is True
    assert res.escalation_reason is not None and "low_confidence" in res.escalation_reason
    assert res.attempts == 2
    assert res.prompt_tokens == 200
    assert len(client.calls) == 2


def test_router_escalates_on_invalid_json() -> None:
    client = FakeLLMClient(
        {
            "openai/gpt-oss-20b": "not valid json at all",
            "openai/gpt-oss-120b": '{"valid": true, "confidence": 0.9}',
        }
    )
    router = LLMRouter(client, config=RouterConfig._default())
    res = router.complete("Return JSON", require_json=True)

    assert res.escalated is True
    assert res.escalation_reason == "invalid_json"
    assert res.model_used == "openai/gpt-oss-120b"


def test_router_disabled_ablation_uses_strong_directly() -> None:
    client = FakeLLMClient(
        {
            "openai/gpt-oss-20b": '{"confidence": 0.99}',
            "openai/gpt-oss-120b": '{"confidence": 0.99}',
        }
    )
    router = LLMRouter(client, config=RouterConfig._default(), enabled=False)
    res = router.complete("Run no_router ablation")

    assert res.tier_used == "strong"
    assert res.model_used == "openai/gpt-oss-120b"
    assert len(client.calls) == 1


def test_router_works_completely_offline_without_any_api_key() -> None:
    router = LLMRouter(llm_client=None, config=RouterConfig._default(), offline_fallback=True)
    res = router.complete("Verify claim offline", require_json=True)

    assert res.tier_used == "cheap"
    assert res.escalated is False
    assert '"status": "offline_mock"' in res.content
