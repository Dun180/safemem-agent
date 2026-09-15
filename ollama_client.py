from typing import Any

from ollama import Client


MODEL_NAME = "qwen3:4b"


def chat_with_model(
    messages: list[dict[str, Any]],
    tools: list[Any],
    client: Client | None = None,
):
    if client is None:
        client = Client()

    return client.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
    )