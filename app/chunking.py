"""Chunking — slice a transcript into small, overlapping, timestamped passages.

This is stage 2 of the RAG pipeline. Embedding models and LLMs work best on short
passages, so we regroup the transcript's words into fixed-size windows that overlap
a little, and we remember the timestamp of each window's first word so answers can
cite it later ("see 6:12").
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion import Transcript


@dataclass
class Chunk:
    """A small, embeddable passage of the transcript."""

    index: int      # position of this chunk in the video (0, 1, 2, ...)
    text: str       # the passage itself
    start: float    # seconds — timestamp of the first word (used for citations)
    end: float      # seconds — timestamp of the last word (approximate)


def _tokenize_with_timestamps(transcript: Transcript) -> list[tuple[str, float]]:
    """Flatten the transcript into a flat list of (word, start_time) pairs.

    Each word inherits the start time of the caption line it came from, so even
    after we regroup words into new windows we still know roughly when each was
    spoken. For a pasted transcript (one segment, no real timestamps) every word
    just gets 0.0 — chunking still works, we simply lose fine-grained citations.
    """
    tokens: list[tuple[str, float]] = []
    for seg in transcript.segments:
        for word in seg.text.split():
            tokens.append((word, seg.start))
    return tokens


def chunk_transcript(
    transcript: Transcript,
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[Chunk]:
    """Group the transcript's words into overlapping windows.

    chunk_size: how many words per chunk. ~120 words (~150-180 subword tokens)
        stays safely under the 256-token input limit of the all-MiniLM embedding
        model, while being big enough to hold a complete thought.
    overlap: how many words each chunk repeats from the end of the previous one,
        so an idea sitting on a boundary isn't split across two chunks and lost
        to retrieval.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    tokens = _tokenize_with_timestamps(transcript)
    if not tokens:
        return []

    step = chunk_size - overlap        # how far the window slides each time
    chunks: list[Chunk] = []
    i = 0
    index = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size]
        text = " ".join(word for word, _ in window)
        chunks.append(
            Chunk(
                index=index,
                text=text,
                start=window[0][1],     # timestamp of the first word
                end=window[-1][1],      # timestamp of the last word (approx)
            )
        )
        index += 1
        if i + chunk_size >= len(tokens):
            break                        # this window reached the end; stop
        i += step
    return chunks
