import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import AgentResult, SYSTEM_PROMPT, ToolEvent
from ollama_client import chat_with_model
from tools import AVAILABLE_TOOLS, execute_tool

MAX_STEPS = 5

class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]
    tool_events: Annotated[list[ToolEvent], operator.add]
    step_count: int


def agent_node(state: AgentState) -> dict:
    if state["step_count"] >= MAX_STEPS:
        raise RuntimeError(
            f"Agent exceeded maximum number of steps: {MAX_STEPS}"
        )

    response = chat_with_model(
        messages=state["messages"],
        tools=AVAILABLE_TOOLS,
    )

    return {
        "messages": [response.message],
        "step_count": state["step_count"] + 1,
    }


def tool_node(state: AgentState) -> dict:
    assistant_message = state["messages"][-1]

    tool_messages = []
    tool_events = []

    for tool_call in assistant_message.tool_calls:
        name = tool_call.function.name
        arguments = dict(
            tool_call.function.arguments
        )

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

        tool_messages.append(
            {
                "role": "tool",
                "content": result,
                "tool_name": name,
            }
        )

    return {
        "messages": tool_messages,
        "tool_events": tool_events,
    }


def route_after_agent(
    state: AgentState,
) -> str:
    assistant_message = state["messages"][-1]

    if assistant_message.tool_calls:
        return "tools"

    return "end"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node(
        "agent",
        agent_node,
    )

    builder.add_node(
        "tools",
        tool_node,
    )

    builder.add_edge(
        START,
        "agent",
    )

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "end": END,
        },
    )

    builder.add_edge(
        "tools",
        "agent",
    )

    return builder.compile()


graph = build_graph()


def run_langgraph_agent(
    user_input: str,
) -> AgentResult:
    initial_state: AgentState = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        "tool_events": [],
        "step_count": 0,
    }

    final_state = graph.invoke(initial_state)

    final_message = final_state["messages"][-1]

    return AgentResult(
        answer=final_message.content,
        tool_events=final_state["tool_events"],
        messages=final_state["messages"],
    )