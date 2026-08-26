"""Vector store — save chunk embeddings and find the closest ones to a query.

Stage 4. We use Chroma locally: it stores each chunk's vector, its text, and its
metadata (video id + timestamp) on disk, and given a query vector it returns the
nearest chunks by cosine similarity. That nearest-neighbor lookup IS the
"retrieval" in Retrieval-Augmented Generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import settings
from app.chunking import Chunk


@dataclass
class SearchHit:
    """One chunk returned by a search, with how similar it is to the query."""

    text: str
    start: float    # timestamp (seconds) — for the citation
    score: float    # 0..1, higher = more similar in meaning


@lru_cache(maxsize=1)
def _collection():
    """Open (once) the on-disk Chroma collection that holds every chunk vector."""
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_dir)
    # "cosine" space matches our normalized embeddings.
    return client.get_or_create_collection(
        name="transcripts",
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(video_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Store a video's chunks + their vectors. Re-running overwrites (upsert)."""
    collection = _collection()
    collection.upsert(
        ids=[f"{video_id}:{chunk.index}" for chunk in chunks],
        embeddings=vectors,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {"video_id": video_id, "index": chunk.index,
             "start": chunk.start, "end": chunk.end}
            for chunk in chunks
        ],
    )


def has_video(video_id: str) -> bool:
    """True if we've already indexed this video (so we can skip re-doing the work)."""
    found = _collection().get(where={"video_id": video_id}, limit=1)
    return len(found["ids"]) > 0


def search(video_id: str, query_vector: list[float], k: int = 5) -> list[SearchHit]:
    """Return the k chunks of this video whose meaning is closest to the query."""
    result = _collection().query(
        query_embeddings=[query_vector],
        n_results=k,
        where={"video_id": video_id},
    )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    hits: list[SearchHit] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        # Chroma returns cosine *distance* (0 = identical); similarity = 1 - distance.
        hits.append(SearchHit(text=text, start=meta.get("start", 0.0), score=1.0 - distance))
    return hits
