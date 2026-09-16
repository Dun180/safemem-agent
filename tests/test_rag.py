import pytest

from rag import chunk_text
from rag import load_documents

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