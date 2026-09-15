from dataclasses import dataclass
from typing import Any

from ollama_client import chat_with_model
from tools import AVAILABLE_TOOLS, execute_tool


MAX_STEPS = 5

SYSTEM_PROMPT = """
You are a tool-using AI agent.

When a task requires arithmetic, you MUST use the calculator tool.
Do not perform arithmetic yourself.

For multi-step calculations:
1. Call the calculator for the first operation.
2. Wait for the tool result.
3. Use that result in the next calculator call.
4. Continue until the task is complete.

You are operating inside an agent loop, so you may call tools over multiple turns.
""".strip()


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class AgentResult:
    answer: str
    tool_events: list[ToolEvent]
    messages: list[Any]


def run_agent(user_input: str) -> AgentResult:
    messages: list[Any] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_input,
        }
    ]

    tool_events: list[ToolEvent] = []

    for _ in range(MAX_STEPS):
        response = chat_with_model(
            messages=messages,
            tools=AVAILABLE_TOOLS,
        )

        assistant_message = response.message
        messages.append(assistant_message)

        if not assistant_message.tool_calls:
            return AgentResult(
                answer=assistant_message.content,
                tool_events=tool_events,
                messages=messages,
            )

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            arguments = dict(tool_call.function.arguments)

            result = execute_tool(
                name=name,
                arguments=arguments,
            )

            tool_events.append(
                ToolEvent(
                    name=name,
                    arguments=arguments,
                    result=result,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_name": name,
                }
            )

    raise RuntimeError(
        f"Agent exceeded maximum number of steps: {MAX_STEPS}"
    )