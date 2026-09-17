from ollama import Client


EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def embed_texts(
    texts: list[str],
    client: Client | None = None,
) -> list[list[float]]:
    if client is None:
        client = Client()

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [
        list(embedding)
        for embedding in response.embeddings
    ]


def embed_text(
    text: str,
    client: Client | None = None,
) -> list[float]:
    return embed_texts(
        [text],
        client=client,
    )[0]