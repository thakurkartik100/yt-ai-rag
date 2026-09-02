"""Unit tests for app.ingestion -- no network calls needed."""

import pytest

from app.ingestion import (
    TranscriptUnavailable,
    Segment,
    Transcript,
    extract_video_id,
    format_timestamp,
    from_pasted_text,
    _clean,
)


class TestExtractVideoId:
    def test_bare_id(self):
        assert extract_video_id("TgF-uMvhNmg") == "TgF-uMvhNmg"

    def test_watch_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=TgF-uMvhNmg") == "TgF-uMvhNmg"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/TgF-uMvhNmg") == "TgF-uMvhNmg"

    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/TgF-uMvhNmg") == "TgF-uMvhNmg"

    def test_shorts_url(self):
        assert extract_video_id("https://www.youtube.com/shorts/TgF-uMvhNmg") == "TgF-uMvhNmg"

    def test_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=TgF-uMvhNmg" + "&t=60s"
        assert extract_video_id(url) == "TgF-uMvhNmg"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("https://vimeo.com/12345678")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("")

    def test_strips_whitespace(self):
        assert extract_video_id("  TgF-uMvhNmg  ") == "TgF-uMvhNmg"


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "0:00"

    def test_seconds_only(self):
        assert format_timestamp(45) == "0:45"

    def test_one_minute(self):
        assert format_timestamp(60) == "1:00"

    def test_minutes_and_seconds(self):
        assert format_timestamp(372) == "6:12"

    def test_hours(self):
        assert format_timestamp(3723) == "1:02:03"

    def test_float_input(self):
        assert format_timestamp(372.9) == "6:12"


class TestClean:
    def test_removes_music_tag(self):
        assert _clean("[Music]") == ""

    def test_removes_applause_tag(self):
        assert _clean("[Applause] hello") == "hello"

    def test_collapses_whitespace(self):
        assert _clean("hello   world") == "hello world"

    def test_strips_newlines(self):
        assert _clean("hello\nworld") == "hello world"

    def test_empty_string(self):
        assert _clean("") == ""


class TestFromPastedText:
    def test_creates_transcript(self):
        t = from_pasted_text("abc12345678", "Hello world")
        assert t.video_id == "abc12345678"
        assert t.source == "manual"
        assert len(t.segments) == 1
        assert "Hello world" in t.segments[0].text

    def test_empty_text_raises(self):
        with pytest.raises(TranscriptUnavailable):
            from_pasted_text("abc12345678", "   ")

    def test_only_tags_raises(self):
        with pytest.raises(TranscriptUnavailable):
            from_pasted_text("abc12345678", "[Music][Applause]")


class TestTranscript:
    def _make(self, segments):
        return Transcript(video_id="test123456x", language="en", source="youtube", segments=segments)

    def test_full_text(self):
        t = self._make([Segment("hello", 0.0, 1.0), Segment("world", 1.0, 1.0)])
        assert t.full_text == "hello world"

    def test_duration(self):
        t = self._make([Segment("a", 0.0, 2.0), Segment("b", 5.0, 3.0)])
        assert t.duration == 8.0

    def test_empty_transcript_duration(self):
        t = self._make([])
        assert t.duration == 0.0
