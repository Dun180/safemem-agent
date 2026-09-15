from unittest.mock import Mock

from ollama_client import MODEL_NAME, chat_with_model


def test_chat_with_model_uses_qwen3_4b():
    fake_client = Mock()

    fake_response = Mock()
    fake_client.chat.return_value = fake_response

    messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    tools = []

    response = chat_with_model(
        messages=messages,
        tools=tools,
        client=fake_client,
    )

    fake_client.chat.assert_called_once_with(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
    )

    assert MODEL_NAME == "qwen3:4b"
    assert response is fake_response