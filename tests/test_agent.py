from types import SimpleNamespace

import agent


def assistant_message(content="", tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )


def tool_call(name, arguments):
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    )


def response(message):
    return SimpleNamespace(message=message)


def test_run_agent_executes_tool_then_returns_final_answer(monkeypatch):
    responses = iter(
        [
            # 第一次调用模型：模型要求使用 calculator
            response(
                assistant_message(
                    tool_calls=[
                        tool_call(
                            "calculator",
                            {
                                "a": 24,
                                "b": 37,
                                "operation": "multiply",
                            },
                        )
                    ]
                )
            ),

            # 第二次调用模型：看到工具结果后给最终答案
            response(
                assistant_message(
                    content="24 × 37 = 888."
                )
            ),
        ]
    )

    monkeypatch.setattr(
        agent,
        "chat_with_model",
        lambda messages, tools: next(responses),
    )

    result = agent.run_agent("What is 24 * 37?")

    assert result.answer == "24 × 37 = 888."

    assert len(result.tool_events) == 1

    assert result.tool_events[0].name == "calculator"
    assert result.tool_events[0].arguments == {
        "a": 24,
        "b": 37,
        "operation": "multiply",
    }
    assert result.tool_events[0].result == "888"

    assert result.messages[-2]["role"] == "tool"
    assert result.messages[-2]["tool_name"] == "calculator"
    assert result.messages[-2]["content"] == "888"

def test_run_agent_includes_agent_system_prompt(monkeypatch):
    captured_messages = []

    def fake_chat(messages, tools):
        captured_messages.extend(messages)

        return response(
            assistant_message(
                content="Done."
            )
        )

    monkeypatch.setattr(
        agent,
        "chat_with_model",
        fake_chat,
    )

    agent.run_agent("Hello")

    assert captured_messages[0]["role"] == "system"

    assert "tool-using AI agent" in captured_messages[0]["content"]

    assert "multiple turns" in captured_messages[0]["content"]