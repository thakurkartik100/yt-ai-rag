"""Unit tests for app.chunking -- pure logic, no network or disk I/O."""

import pytest

from app.chunking import chunk_transcript, Chunk
from app.ingestion import Segment, Transcript


def _make_transcript(words: int, seconds_per_word: float = 0.5) -> Transcript:
    text = " ".join(f"word{i}" for i in range(words))
    seg = Segment(text=text, start=0.0, duration=words * seconds_per_word)
    return Transcript(video_id="fakeid12345", language="en", source="youtube", segments=[seg])


class TestChunkTranscript:
    def test_returns_list_of_chunks(self):
        t = _make_transcript(200)
        chunks = chunk_transcript(t, chunk_size=100, overlap=10)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_count_reasonable(self):
        t = _make_transcript(200)
        chunks = chunk_transcript(t, chunk_size=100, overlap=10)
        assert len(chunks) >= 2

    def test_each_chunk_has_text(self):
        t = _make_transcript(150)
        for chunk in chunk_transcript(t, chunk_size=80, overlap=10):
            assert chunk.text.strip() != ""

    def test_chunk_indices_are_sequential(self):
        t = _make_transcript(300)
        chunks = chunk_transcript(t, chunk_size=100, overlap=20)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_chunk_word_count_within_size(self):
        chunk_size = 60
        t = _make_transcript(200)
        for chunk in chunk_transcript(t, chunk_size=chunk_size, overlap=10):
            assert len(chunk.text.split()) <= chunk_size

    def test_single_chunk_when_transcript_short(self):
        t = _make_transcript(50)
        chunks = chunk_transcript(t, chunk_size=100, overlap=10)
        assert len(chunks) == 1

    def test_empty_transcript_returns_empty_list(self):
        t = Transcript(video_id="fakeid12345", language="en", source="youtube", segments=[])
        assert chunk_transcript(t) == []

    def test_invalid_chunk_size_raises(self):
        t = _make_transcript(50)
        with pytest.raises(ValueError):
            chunk_transcript(t, chunk_size=0)

    def test_invalid_overlap_raises(self):
        t = _make_transcript(50)
        with pytest.raises(ValueError):
            chunk_transcript(t, chunk_size=50, overlap=50)

    def test_overlap_produces_repeated_words(self):
        overlap = 20
        t = _make_transcript(300)
        chunks = chunk_transcript(t, chunk_size=100, overlap=overlap)
        if len(chunks) > 1:
            tail_words = set(chunks[0].text.split()[-overlap:])
            head_words = set(chunks[1].text.split()[:overlap])
            assert tail_words == head_words

    def test_start_timestamps_non_negative(self):
        t = _make_transcript(200)
        for chunk in chunk_transcript(t):
            assert chunk.start >= 0.0
