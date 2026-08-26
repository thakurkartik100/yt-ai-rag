"""Embeddings — turn chunk text into vectors that capture meaning.

Stage 3. We use sentence-transformers (all-MiniLM-L6-v2) locally: free, no API
key, CPU-friendly. Each text becomes a 384-number vector; texts with similar
meaning land close together, which is what makes semantic search possible later.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    """Load the embedding model once, on first use.

    The import is lazy (inside the function) so the API starts instantly and only
    pays the model-loading cost when something actually needs an embedding.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts at once -> one vector per text."""
    if not texts:
        return []
    model = _get_model()
    # normalize_embeddings=True makes every vector length 1, so later a simple dot
    # product equals cosine similarity — the standard "how similar in meaning" score.
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single query with the same model, so it's comparable to the chunks."""
    return embed_texts([text])[0]
