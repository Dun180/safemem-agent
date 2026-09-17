import json
import re
from dataclasses import dataclass
from typing import Any
from ollama_client import chat_with_model
from tools import execute_tool

@dataclass
class ReActAction:
    thought: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ReActFinalAnswer:
    thought: str
    answer: str

MAX_STEPS = 5


REACT_SYSTEM_PROMPT = """
You are a ReAct-style tool-using AI agent.

You have access to the following tool:

calculator
- Purpose: Perform basic arithmetic.
- Arguments:
  - a: number
  - b: number
  - operation: one of add, subtract, multiply, divide

You are operating inside an external agent loop.

IMPORTANT RULES:

1. On each turn, output EXACTLY ONE of the following:

   A) One Thought + one Action + one Action Input

   OR

   B) One Thought + one Final Answer

2. NEVER output an Observation yourself.

3. Observations are produced only by the external runtime after it executes your Action.

4. After producing an Action and Action Input, STOP immediately.

5. Wait for the runtime to provide an Observation before deciding the next step.

When you need to use a tool, respond EXACTLY in this format:

Thought: <brief reasoning>

Action: <tool name>

Action Input:
<valid JSON object>

When you have enough information to answer, respond EXACTLY in this format:

Thought: <brief reasoning>

Final Answer: <answer>

Do not perform arithmetic yourself when the calculator can be used.

You may use multiple Thought -> Action -> Observation cycles before giving the final answer.
""".strip()


@dataclass
class ReActStep:
    thought: str
    action: str
    arguments: dict[str, Any]
    observation: str


@dataclass
class ReActResult:
    answer: str
    steps: list[ReActStep]
    messages: list[dict[str, str]]

def parse_react_response(
    text: str,
) -> ReActAction | ReActFinalAnswer:
    text = text.strip()

    if re.search(
        r"(?m)^\s*Observation\s*:",
        text,
    ):
        raise ValueError(
            "Invalid ReAct response: "
            "Observation must be produced by the runtime, "
            "not by the model"
        )

    final_answer_pattern = re.compile(
        r"""
        ^Thought:\s*
        (?P<thought>.*?)
        \s*\n+
        Final\ Answer:\s*
        (?P<answer>.+?)
        \s*$
        """,
        re.DOTALL | re.VERBOSE,
    )

    final_match = final_answer_pattern.match(text)

    if final_match:
        return ReActFinalAnswer(
            thought=final_match.group("thought").strip(),
            answer=final_match.group("answer").strip(),
        )

    action_pattern = re.compile(
        r"""
        ^Thought:\s*
        (?P<thought>.*?)
        \s*\n+
        Action:\s*
        (?P<name>[^\n]+)
        \s*\n+
        Action\ Input:\s*
        (?P<arguments>\{.*\})
        \s*$
        """,
        re.DOTALL | re.VERBOSE,
    )

    action_match = action_pattern.match(text)

    if action_match:
        try:
            arguments = json.loads(
                action_match.group("arguments")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid ReAct response: "
                "Action Input must be valid JSON"
            ) from exc

        if not isinstance(arguments, dict):
            raise ValueError(
                "Invalid ReAct response: "
                "Action Input must be a JSON object"
            )

        return ReActAction(
            thought=action_match.group("thought").strip(),
            name=action_match.group("name").strip(),
            arguments=arguments,
        )

    raise ValueError(
        "Invalid ReAct response: expected either "
        "Thought + Action + Action Input, "
        "or Thought + Final Answer"
    )

def run_react_agent(
    user_input: str,
) -> ReActResult:
    messages = [
        {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    steps: list[ReActStep] = []

    for _ in range(MAX_STEPS):
        response = chat_with_model(
            messages=messages,
            tools=[],
        )

        content = response.message.content.strip()

        parsed = parse_react_response(content)

        messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

        if isinstance(parsed, ReActFinalAnswer):
            return ReActResult(
                answer=parsed.answer,
                steps=steps,
                messages=messages,
            )

        observation = execute_tool(
            name=parsed.name,
            arguments=parsed.arguments,
        )

        steps.append(
            ReActStep(
                thought=parsed.thought,
                action=parsed.name,
                arguments=parsed.arguments,
                observation=observation,
            )
        )

        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}",
            }
        )

    raise RuntimeError(
        f"ReAct agent exceeded maximum number of steps: {MAX_STEPS}"
    )