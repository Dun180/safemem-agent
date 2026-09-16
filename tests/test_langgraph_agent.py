from types import SimpleNamespace

import langgraph_agent
import pytest

def model_response(content="", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls or [],
        )
    )


def test_langgraph_agent_returns_model_answer(monkeypatch):
    def fake_chat(messages, tools):
        return model_response(
            content="Canberra"
        )

    monkeypatch.setattr(
        langgraph_agent,
        "chat_with_model",
        fake_chat,
    )

    result = langgraph_agent.run_langgraph_agent(
        "What is the capital of Australia?"
    )

    assert result.answer == "Canberra"
    assert result.tool_events == []

    assert result.messages[0]["role"] == "system"
    assert result.messages[1]["role"] == "user"

    assert result.messages[-1].content == "Canberra"

def tool_call(name, arguments):
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    )

def test_langgraph_agent_routes_through_tool_node(monkeypatch):
    responses = iter(
        [
            # 第一次：LLM 请求 calculator
            model_response(
                tool_calls=[
                    tool_call(
                        "calculator",
                        {
                            "a": 6,
                            "b": 7,
                            "operation": "multiply",
                        },
                    )
                ]
            ),

            # 第二次：看到工具结果后给最终答案
            model_response(
                content="6 × 7 = 42."
            ),
        ]
    )

    monkeypatch.setattr(
        langgraph_agent,
        "chat_with_model",
        lambda messages, tools: next(responses),
    )

    result = langgraph_agent.run_langgraph_agent(
        "What is 6 * 7?"
    )

    assert result.answer == "6 × 7 = 42."

    assert len(result.tool_events) == 1

    assert result.tool_events[0].name == "calculator"

    assert result.tool_events[0].arguments == {
        "a": 6,
        "b": 7,
        "operation": "multiply",
    }

    assert result.tool_events[0].result == "42"

    assert result.messages[-2]["role"] == "tool"
    assert result.messages[-2]["content"] == "42"

def test_langgraph_agent_supports_multiple_tool_rounds(monkeypatch):
    responses = iter(
        [
            # Round 1:
            # 25 * 48
            model_response(
                tool_calls=[
                    tool_call(
                        "calculator",
                        {
                            "a": 25,
                            "b": 48,
                            "operation": "multiply",
                        },
                    )
                ]
            ),

            # Round 2:
            # 1200 / 10
            model_response(
                tool_calls=[
                    tool_call(
                        "calculator",
                        {
                            "a": 1200,
                            "b": 10,
                            "operation": "divide",
                        },
                    )
                ]
            ),

            # Final answer
            model_response(
                content="The final result is 120."
            ),
        ]
    )

    monkeypatch.setattr(
        langgraph_agent,
        "chat_with_model",
        lambda messages, tools: next(responses),
    )

    result = langgraph_agent.run_langgraph_agent(
        "First calculate 25 * 48, then divide the result by 10."
    )

    assert result.answer == "The final result is 120."

    assert len(result.tool_events) == 2

    assert result.tool_events[0].name == "calculator"
    assert result.tool_events[0].arguments == {
        "a": 25,
        "b": 48,
        "operation": "multiply",
    }
    assert result.tool_events[0].result == "1200"

    assert result.tool_events[1].name == "calculator"
    assert result.tool_events[1].arguments == {
        "a": 1200,
        "b": 10,
        "operation": "divide",
    }
    assert result.tool_events[1].result == "120.0"

def test_langgraph_agent_stops_when_model_keeps_calling_tools(
    monkeypatch,
):
    call_count = 0

    def fake_chat(messages, tools):
        nonlocal call_count
        call_count += 1

        return model_response(
            tool_calls=[
                tool_call(
                    "calculator",
                    {
                        "a": 6,
                        "b": 7,
                        "operation": "multiply",
                    },
                )
            ]
        )

    monkeypatch.setattr(
        langgraph_agent,
        "chat_with_model",
        fake_chat,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "execute_tool",
        lambda name, arguments: "42",
    )

    with pytest.raises(
        RuntimeError,
        match="maximum number of steps",
    ):
        langgraph_agent.run_langgraph_agent(
            "Keep calculating 6 * 7 forever."
        )

    assert call_count == langgraph_agent.MAX_STEPS