"""Transcript ingestion — turn a YouTube URL into clean, timestamped text.

This is step 1 of the RAG pipeline (Phase A in docs/how-it-works.md). Everything
here is provider-agnostic: it just produces a `Transcript` object that later steps
(chunking, embeddings) will consume.

Design notes (verified 2026-08-26):
* youtube-transcript-api v1.2.3 uses an *instance* API: YouTubeTranscriptApi().fetch(...)
* YouTube blocks most datacenter/cloud IPs, so a direct fetch is reliable from your
  laptop but NOT from free hosts. That's why `get_transcript` also accepts a
  `pasted_text` fallback, and why we surface a clear error instead of retrying forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from youtube_transcript_api import YouTubeTranscriptApi

# The 11-character video id as it appears inside common YouTube URL shapes.
_YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|watch\?v=|/embed/|/shorts/|/v/|[?&]v=)([A-Za-z0-9_-]{11})"
)
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptUnavailable(Exception):
    """Raised when a transcript can't be retrieved (blocked, disabled, empty, ...)."""


@dataclass
class Segment:
    """One caption line: its text and where it appears in the video."""

    text: str
    start: float      # seconds from the start of the video
    duration: float   # how long the line stays on screen


@dataclass
class Transcript:
    """A whole video's transcript, ready for the next pipeline steps."""

    video_id: str
    language: str
    source: str               # "youtube" (fetched) or "manual" (pasted fallback)
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.start + last.duration


def extract_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL *or* a raw 11-char id and return just the id."""
    text = url_or_id.strip()
    if _BARE_ID_RE.match(text):
        return text
    match = _YOUTUBE_ID_RE.search(text)
    if match:
        return match.group(1)
    raise ValueError(f"Could not find a YouTube video id in: {url_or_id!r}")


def format_timestamp(seconds: float) -> str:
    """Turn 372.0 seconds into '6:12' (or '1:02:03' for long videos)."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _clean(text: str) -> str:
    """Tidy raw caption text: drop [Music]/[Applause] tags and collapse whitespace."""
    text = re.sub(r"\[[^\]]{0,40}\]", " ", text)   # [Music], [Applause], ...
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_from_youtube(video_id: str, languages: list[str] | None = None) -> Transcript:
    """Fetch captions from YouTube. Raises TranscriptUnavailable on any failure."""
    languages = languages or ["en", "en-US", "en-GB"]
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=languages)
    except Exception as exc:  # classify by class name so we don't depend on exact imports
        name = type(exc).__name__
        if name in {"TranscriptsDisabled", "NoTranscriptFound"}:
            raise TranscriptUnavailable(
                "This video has no captions/transcript available."
            ) from exc
        if name in {"RequestBlocked", "IpBlocked"}:
            raise TranscriptUnavailable(
                "YouTube blocked this request (common from cloud/datacenter IPs). "
                "Paste the transcript manually, or run from a residential connection."
            ) from exc
        if name == "VideoUnavailable":
            raise TranscriptUnavailable(
                "Video is unavailable (private, removed, or region-locked)."
            ) from exc
        raise TranscriptUnavailable(f"Could not retrieve transcript ({name}).") from exc

    # v1.x returns a FetchedTranscript; normalize to plain segments.
    raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
    segments: list[Segment] = []
    for item in raw:
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start", 0.0)
            dur = item.get("duration", 0.0)
        else:  # snippet object with attributes
            text = item.text
            start = item.start
            dur = getattr(item, "duration", 0.0)
        cleaned = _clean(text)
        if cleaned:
            segments.append(Segment(text=cleaned, start=float(start), duration=float(dur)))

    if not segments:
        raise TranscriptUnavailable("Transcript came back empty after cleaning.")

    language = getattr(fetched, "language_code", languages[0])
    return Transcript(video_id=video_id, language=language, source="youtube", segments=segments)


def from_pasted_text(video_id: str, text: str) -> Transcript:
    """Fallback path: build a Transcript from user-pasted text (no real timestamps)."""
    cleaned = _clean(text)
    if not cleaned:
        raise TranscriptUnavailable("The pasted transcript was empty.")
    return Transcript(
        video_id=video_id,
        language="unknown",
        source="manual",
        segments=[Segment(text=cleaned, start=0.0, duration=0.0)],
    )


def get_transcript(
    url_or_id: str,
    pasted_text: str | None = None,
    languages: list[str] | None = None,
) -> Transcript:
    """High-level entry point used by the API.

    Tries YouTube first; if `pasted_text` is provided we trust that instead (useful
    when YouTube blocks the server). Raises ValueError for a bad URL, and
    TranscriptUnavailable when no transcript can be produced.
    """
    video_id = extract_video_id(url_or_id)
    if pasted_text:
        return from_pasted_text(video_id, pasted_text)
    return fetch_from_youtube(video_id, languages=languages)
