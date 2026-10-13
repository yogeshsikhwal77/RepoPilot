import pytest

from repopilot.llm.client import (
    LLMResponse,
    MockLLMClient,
    OpenAIClient,
    calculate_cost,
    get_llm_client,
)


def test_calculate_cost():
    # gpt-4o-mini: 0.15 / 1M in, 0.60 / 1M out
    cost = calculate_cost("gpt-4o-mini", 1000, 1000)
    expected = (1000 / 1_000_000) * 0.15 + (1000 / 1_000_000) * 0.60
    assert abs(cost - expected) < 1e-6

    # mock: 0 cost
    assert calculate_cost("mock", 500, 500) == 0.0


def test_llm_response_dict_and_attr_access():
    resp = LLMResponse(
        text="diff content",
        prompt_tokens=10,
        completion_tokens=20,
        model="gpt-4o-mini",
    )

    # Attribute access
    assert resp.text == "diff content"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 20
    assert resp.total_tokens == 30

    # Dict access
    assert resp["text"] == "diff content"
    assert resp.get("text") == "diff content"
    assert resp.get("nonexistent", "default") == "default"
    assert "text" in resp

    d = resp.to_dict()
    assert d["text"] == "diff content"
    assert d["total_tokens"] == 30


def test_mock_llm_client_generate():
    client = MockLLMClient()
    resp = client.generate(
        model="gpt-4o-mini",
        prompt="Fix the bug in math.py",
        temperature=0.0,
    )

    assert "diff" in resp.text
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert len(client.history) == 1
    assert client.history[0]["prompt"] == "Fix the bug in math.py"


def test_mock_llm_client_prompt_responses():
    client = MockLLMClient(
        prompt_responses={"hello": "world"}
    )
    resp = client.generate(model="mock", prompt="hello")
    assert resp.text == "world"


@pytest.mark.asyncio
async def test_mock_llm_client_async():
    client = MockLLMClient()
    resp = await client.agenerate(model="mock", prompt="test async")
    assert "diff" in resp.text


def test_get_llm_client_factory():
    mock_client = get_llm_client("mock")
    assert isinstance(mock_client, MockLLMClient)

    openai_client = get_llm_client("openai", api_key="test-key")
    assert isinstance(openai_client, OpenAIClient)

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_client("unknown-provider")
