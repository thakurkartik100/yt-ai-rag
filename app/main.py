"""FastAPI entry point.

Run locally with:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app import __version__
from app.config import settings
from app.ingestion import TranscriptUnavailable, format_timestamp, get_transcript

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
