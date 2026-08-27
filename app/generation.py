"""Generation — build a grounded prompt from retrieved chunks and call the LLM.

Stage 5. We hand the model ONLY the retrieved transcript excerpts as context and
tell it to answer strictly from them (and to say so when the answer isn't there).
That grounding is what prevents hallucination and lets the answer cite timestamps.

Groq is OpenAI-compatible, so we use the official `openai` client pointed at Groq's
base URL — the same code would talk to OpenAI by changing the URL and key.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.config import settings
from app.ingestion import format_timestamp
from app.vectorstore import SearchHit

SYSTEM_PROMPT = (
    "You answer questions about a YouTube video using ONLY the transcript excerpts "
    "provided. If the answer is not in the excerpts, say you don't know — do not "
    "guess. Be concise, and cite the timestamps (e.g. [6:12]) of the excerpts you use."
)


@lru_cache(maxsize=1)
def _client():
    """Create the Groq client once (OpenAI-compatible)."""
    from openai import OpenAI

    return OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)


def _build_context(hits: list[SearchHit]) -> str:
    """Turn retrieved chunks into a timestamp-labelled block the model can cite."""
    return "\n\n".join(f"[{format_timestamp(hit.start)}] {hit.text}" for hit in hits)


def generate_answer(question: str, hits: list[SearchHit]) -> str:
    """Ask the LLM to answer the question using only the retrieved excerpts."""
    user_prompt = (
        f"Transcript excerpts:\n{_build_context(hits)}\n\n"
        f"Question: {question}"
    )
    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,   # low = stick to the facts, less creative drift
    )
    return (response.choices[0].message.content or "").strip()


SUMMARY_SYSTEM_PROMPT = (
    "You summarize a YouTube video from its transcript. Write one short overview "
    "sentence, then 3-6 key points as a bulleted list. Begin each bullet with the "
    "timestamp where that point starts, e.g. '- [6:12] ...'. Use ONLY the transcript."
)


def summarize(hits: list[SearchHit]) -> str:
    """Summarize the whole video from all its chunks (not a top-k retrieval)."""
    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{_build_context(hits)}\n\nWrite the summary."},
        ],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


QUIZ_SYSTEM_PROMPT = (
    "You create a multiple-choice quiz from a YouTube transcript. Return ONLY JSON "
    'matching this schema: {"quiz": [{"question": <str>, '
    '"options": [<str>, <str>, <str>, <str>], "answer": <str>, "timestamp": <"M:SS">}]}. '
    "Each question must have exactly 4 options; `answer` must be exactly one of the four "
    "options; `timestamp` is where the answer is discussed. Use ONLY the transcript."
)


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of the model's reply, tolerating stray prose or ``` fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def generate_quiz(hits: list[SearchHit], num_questions: int = 5) -> list[dict]:
    """Ask the LLM for a multiple-choice quiz as structured JSON, and parse it."""
    user_prompt = (
        f"Transcript:\n{_build_context(hits)}\n\n"
        f"Write {num_questions} quiz questions as JSON."
    )
    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},   # ask the API to guarantee valid JSON
    )
    raw = (response.choices[0].message.content or "").strip()
    return _extract_json(raw).get("quiz", [])
