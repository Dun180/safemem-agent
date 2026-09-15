from collections.abc import Callable, Mapping
from typing import Any


def calculator(a: float, b: float, operation: str) -> float:
    """Perform a basic arithmetic operation on two numbers.

    Args:
        a: The first number.
        b: The second number.
        operation: One of add, subtract, multiply, or divide.

    Returns:
        The numeric result of the requested operation.

    Raises:
        ValueError: If the operation is unsupported or division by zero is requested.
    """
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    raise ValueError(f"Unsupported operation: {operation}")


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "calculator": calculator,
}


AVAILABLE_TOOLS: list[Callable[..., Any]] = list(
    TOOL_REGISTRY.values()
)


def execute_tool(
    name: str,
    arguments: Mapping[str, Any],
) -> str:
    function = TOOL_REGISTRY.get(name)

    if function is None:
        return f"Unknown tool: {name}"

    try:
        result = function(**dict(arguments))
        return str(result)

    except (TypeError, ValueError) as exc:
        return f"Tool execution failed for {name}: {exc}"