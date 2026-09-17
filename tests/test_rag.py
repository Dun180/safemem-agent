import pytest

from rag import chunk_text
from rag import load_documents
from rag import cosine_similarity
from rag import (
    DocumentChunk,
    build_index,
)
from rag import (
    DocumentChunk,
    EmbeddedChunk,
    retrieve,
)
def test_chunk_text_creates_overlapping_chunks():
    text = "abcdefghij"

    chunks = chunk_text(
        text=text,
        source="example.txt",
        chunk_size=6,
        overlap=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "abcdef",
        "efghij",
    ]

    assert [chunk.chunk_id for chunk in chunks] == [
        0,
        1,
    ]

    assert all(
        chunk.source == "example.txt"
        for chunk in chunks
    )


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_size",
    ):
        chunk_text(
            text="abc",
            source="example.txt",
            chunk_size=0,
            overlap=0,
        )


def test_chunk_text_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        chunk_text(
            text="abc",
            source="example.txt",
            chunk_size=4,
            overlap=4,
        )

def test_load_documents_reads_only_md_and_txt(tmp_path):
    (tmp_path / "a.md").write_text(
        "alpha",
        encoding="utf-8",
    )

    (tmp_path / "b.txt").write_text(
        "beta",
        encoding="utf-8",
    )

    (tmp_path / "ignored.json").write_text(
        '{"x": 1}',
        encoding="utf-8",
    )

    chunks = load_documents(
        tmp_path,
        chunk_size=100,
        overlap=0,
    )

    assert [chunk.source for chunk in chunks] == [
        "a.md",
        "b.txt",
    ]

    assert [chunk.text for chunk in chunks] == [
        "alpha",
        "beta",
    ]


def test_cosine_similarity_identical_vectors_is_one():
    score = cosine_similarity(
        [1.0, 2.0],
        [1.0, 2.0],
    )

    assert score == pytest.approx(1.0)

def test_build_index_embeds_chunks_in_one_batch():
    chunks = [
        DocumentChunk(
            text="alpha",
            source="a.md",
            chunk_id=0,
        ),
        DocumentChunk(
            text="beta",
            source="b.md",
            chunk_id=0,
        ),
    ]

    calls = []

    def fake_embed_texts(texts):
        calls.append(texts)

        return [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    index = build_index(
        chunks,
        embed_fn=fake_embed_texts,
    )

    assert calls == [
        ["alpha", "beta"]
    ]

    assert index[0].embedding == [
        1.0,
        0.0,
    ]

    assert index[1].embedding == [
        0.0,
        1.0,
    ]

def test_retrieve_returns_most_similar_chunks_first():
    index = [
        EmbeddedChunk(
            chunk=DocumentChunk(
                text="Aurora launches in October.",
                source="project.md",
                chunk_id=0,
            ),
            embedding=[1.0, 0.0],
        ),
        EmbeddedChunk(
            chunk=DocumentChunk(
                text="The team likes coffee.",
                source="team.txt",
                chunk_id=0,
            ),
            embedding=[0.0, 1.0],
        ),
    ]

    results = retrieve(
        query="When does Aurora launch?",
        index=index,
        top_k=1,
        embed_fn=lambda _: [1.0, 0.0],
    )

    assert len(results) == 1

    assert results[0].chunk.source == "project.md"

    assert results[0].score == pytest.approx(1.0)