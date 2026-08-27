"""FastAPI entry point.

Run locally with:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app import __version__
from app.config import settings
from app.ingestion import TranscriptUnavailable, extract_video_id, format_timestamp, get_transcript
from app.chunking import chunk_transcript
from app.embeddings import embed_query, embed_texts
from app.vectorstore import add_chunks, get_all_chunks, has_video, search
from app.generation import generate_answer, summarize

app = FastAPI(
    title="Tube AI RAG API",
    description="Ask questions about any YouTube video using a RAG pipeline. (v1 — backend)",
    version=__version__,
)


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_model: str
    # Shows whether the key is configured — WITHOUT ever returning the key itself.
    llm_key_configured: bool


@app.get("/", tags=["system"])
def root() -> dict:
    """Friendly landing response that points to the interactive docs."""
    return {
        "message": "Tube AI RAG API. Open /docs for the interactive API.",
        "version": __version__,
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check. Also reports config sanity so you can debug deploys fast."""
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_model=settings.llm_model,
        llm_key_configured=bool(settings.groq_api_key),
    )


# --- TEMPORARY debug endpoint (milestone 2) ------------------------------------
# Lets you eyeball a fetched transcript in /docs. This gets folded into /ingest
# in a later step and then removed — it's here so you can see ingestion working.
@app.get("/debug/transcript", tags=["debug"])
def debug_transcript(
    url: str = Query(..., description="YouTube URL or 11-character video id"),
    limit: int = Query(5, ge=1, le=50, description="How many segments to preview"),
) -> dict:
    """Fetch a transcript and return a small preview (counts + first lines)."""
    try:
        transcript = get_transcript(url)
    except ValueError as exc:                 # unparseable URL / id
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptUnavailable as exc:       # blocked / disabled / empty
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "video_id": transcript.video_id,
        "language": transcript.language,
        "source": transcript.source,
        "segment_count": len(transcript.segments),
        "total_characters": len(transcript.full_text),
        "duration": format_timestamp(transcript.duration),
        "preview": [
            {"at": format_timestamp(seg.start), "text": seg.text}
            for seg in transcript.segments[:limit]
        ],
    }


# --- TEMPORARY debug endpoint (milestone 2) ------------------------------------
# Lets you see how the transcript gets sliced into chunks. Also folded into the
# real pipeline later and removed.
@app.get("/debug/chunks", tags=["debug"])
def debug_chunks(
    url: str = Query(..., description="YouTube URL or 11-character video id"),
    chunk_size: int = Query(120, ge=20, le=400, description="Words per chunk"),
    overlap: int = Query(20, ge=0, le=100, description="Words shared with the previous chunk"),
    limit: int = Query(3, ge=1, le=20, description="How many chunks to preview"),
) -> dict:
    """Fetch a transcript, split it into overlapping chunks, and preview the first few."""
    try:
        transcript = get_transcript(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    chunks = chunk_transcript(transcript, chunk_size=chunk_size, overlap=overlap)
    return {
        "video_id": transcript.video_id,
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "preview": [
            {
                "index": chunk.index,
                "starts_at": format_timestamp(chunk.start),
                "words": len(chunk.text.split()),
                "text": chunk.text,
            }
            for chunk in chunks[:limit]
        ],
    }


# --- TEMPORARY debug endpoint (milestone 3) ------------------------------------
# Shows what an embedding actually is: text in, a list of numbers out.
@app.get("/debug/embed", tags=["debug"])
def debug_embed(
    text: str = Query(..., description="Any text to turn into a vector"),
) -> dict:
    """Embed a short piece of text and show the vector's size + first few numbers."""
    vector = embed_query(text)
    return {
        "model": settings.embedding_model,
        "dimensions": len(vector),
        "preview": [round(x, 4) for x in vector[:8]],
    }


# --- Ingest: index a whole video (fetch -> chunk -> embed -> store) -------------
class IngestRequest(BaseModel):
    url: str
    pasted_text: Optional[str] = None


class IngestResponse(BaseModel):
    video_id: str
    chunks_indexed: int
    already_indexed: bool


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
def ingest(req: IngestRequest) -> IngestResponse:
    """Index a video so it can be questioned: fetch transcript -> chunk -> embed -> store."""
    try:
        transcript = get_transcript(req.url, pasted_text=req.pasted_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    video_id = transcript.video_id
    if has_video(video_id):
        return IngestResponse(video_id=video_id, chunks_indexed=0, already_indexed=True)

    chunks = chunk_transcript(transcript)
    vectors = embed_texts([chunk.text for chunk in chunks])
    add_chunks(video_id, chunks, vectors)
    return IngestResponse(video_id=video_id, chunks_indexed=len(chunks), already_indexed=False)


# --- Ask: retrieve the relevant chunks and answer from them (with citations) ----
class AskRequest(BaseModel):
    url: str
    question: str
    k: int = 5          # how many chunks to retrieve as evidence


class Citation(BaseModel):
    timestamp: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Answer a question about an already-ingested video, grounded in its transcript."""
    if not settings.groq_api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not set in your .env.")

    video_id = extract_video_id(req.url)
    if not has_video(video_id):
        raise HTTPException(status_code=404, detail="Video not indexed yet — call /ingest first.")

    hits = search(video_id, embed_query(req.question), k=req.k)
    try:
        answer = generate_answer(req.question, hits)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    citations = [
        Citation(timestamp=format_timestamp(hit.start), text=hit.text, score=round(hit.score, 3))
        for hit in hits
    ]
    return AskResponse(answer=answer, citations=citations)


# --- Summary: condense the whole video into timestamped key points --------------
class SummaryRequest(BaseModel):
    url: str


class SummaryResponse(BaseModel):
    video_id: str
    summary: str


@app.post("/summary", response_model=SummaryResponse, tags=["rag"])
def summarize_video(req: SummaryRequest) -> SummaryResponse:
    """Summarize an already-ingested video using its full transcript."""
    if not settings.groq_api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not set in your .env.")

    video_id = extract_video_id(req.url)
    if not has_video(video_id):
        raise HTTPException(status_code=404, detail="Video not indexed yet — call /ingest first.")

    hits = get_all_chunks(video_id)
    try:
        text = summarize(hits)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    return SummaryResponse(video_id=video_id, summary=text)
