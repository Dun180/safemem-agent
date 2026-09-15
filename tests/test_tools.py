import pytest

from tools import calculator, execute_tool


def test_calculator_add():
    assert calculator(2, 3, "add") == 5


def test_calculator_subtract():
    assert calculator(7, 2, "subtract") == 5


def test_calculator_multiply():
    assert calculator(6, 4, "multiply") == 24


def test_calculator_divide():
    assert calculator(6, 2, "divide") == 3


def test_calculator_divide_by_zero():
    with pytest.raises(ValueError, match="Division by zero"):
        calculator(1, 0, "divide")


def test_calculator_rejects_unknown_operation():
    with pytest.raises(ValueError, match="Unsupported operation"):
        calculator(1, 2, "power")


def test_execute_tool_returns_result_as_text():
    assert execute_tool(
        "calculator",
        {
            "a": 2,
            "b": 8,
            "operation": "multiply",
        },
    ) == "16"


def test_execute_tool_returns_readable_unknown_tool_error():
    result = execute_tool("missing_tool", {})

    assert "Unknown tool" in result
    assert "missing_tool" in result


def test_execute_tool_returns_readable_bad_argument_error():
    result = execute_tool(
        "calculator",
        {
            "a": 1,
        },
    )

    assert "Tool execution failed" in result