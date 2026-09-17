from unittest.mock import Mock

from embedding_client import (
    EMBEDDING_MODEL,
    embed_texts,
    embed_text,

)


def test_embed_texts_uses_qwen3_embedding_model():
    fake_client = Mock()

    fake_client.embed.return_value.embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    result = embed_texts(
        ["alpha", "beta"],
        client=fake_client,
    )

    fake_client.embed.assert_called_once_with(
        model=EMBEDDING_MODEL,
        input=["alpha", "beta"],
    )

    assert EMBEDDING_MODEL == "qwen3-embedding:0.6b"

    assert result == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

def test_embed_text_returns_first_embedding():
    fake_client = Mock()

    fake_client.embed.return_value.embeddings = [
        [0.25, 0.75],
    ]

    result = embed_text(
        "hello",
        client=fake_client,
    )

    assert result == [0.25, 0.75]