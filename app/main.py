"""FastAPI entry point.

Run locally with:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app import __version__
from app.config import settings

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
