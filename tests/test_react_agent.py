import pytest

from react_agent import (
    ReActAction,
    ReActFinalAnswer,
    parse_react_response,
    run_react_agent,
)


def test_parse_react_action():
    text = """
Thought: I need to calculate the multiplication first.

Action: calculator

Action Input:
{"a": 25, "b": 48, "operation": "multiply"}
""".strip()

    result = parse_react_response(text)

    assert isinstance(result, ReActAction)

    assert result.thought == (
        "I need to calculate the multiplication first."
    )

    assert result.name == "calculator"

    assert result.arguments == {
        "a": 25,
        "b": 48,
        "operation": "multiply",
    }


def test_parse_react_final_answer():
    text = """
Thought: I now have the result.

Final Answer: 120
""".strip()

    result = parse_react_response(text)

    assert isinstance(result, ReActFinalAnswer)

    assert result.thought == "I now have the result."
    assert result.answer == "120"


def test_parse_react_response_rejects_invalid_output():
    text = """
I think the answer is probably 120.
""".strip()

    with pytest.raises(
        ValueError,
        match="Invalid ReAct response",
    ):
        parse_react_response(text)

from types import SimpleNamespace


def model_response(content):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
        )
    )


def test_run_react_agent_executes_action_then_returns_final_answer(
    monkeypatch,
):
    responses = iter(
        [
            model_response(
                """
Thought: I need to multiply the numbers.

Action: calculator

Action Input:
{"a": 6, "b": 7, "operation": "multiply"}
""".strip()
            ),
            model_response(
                """
Thought: The calculator returned 42, so I can answer now.

Final Answer: 42
""".strip()
            ),
        ]
    )

    monkeypatch.setattr(
        "react_agent.chat_with_model",
        lambda messages, tools: next(responses),
    )

    result = run_react_agent(
        "What is 6 * 7?"
    )

    assert result.answer == "42"

    assert len(result.steps) == 1

    assert result.steps[0].thought == (
        "I need to multiply the numbers."
    )

    assert result.steps[0].action == "calculator"

    assert result.steps[0].arguments == {
        "a": 6,
        "b": 7,
        "operation": "multiply",
    }

    assert result.steps[0].observation == "42"

def test_parse_react_response_rejects_model_generated_observation():
    text = """
Thought: I need to calculate the first step.

Action: calculator

Action Input:
{"a": 25, "b": 48, "operation": "multiply"}

Observation: 1200

Thought: Now I can continue.

Final Answer: 120
""".strip()

    with pytest.raises(
        ValueError,
        match="Observation",
    ):
        parse_react_response(text)