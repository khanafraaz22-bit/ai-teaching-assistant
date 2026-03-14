"""
Unit tests — YouTubePlaylistParser

Tests URL parsing and metadata extraction with mocked yt-dlp.
No real network calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.youtube.playlist_parser import YouTubePlaylistParser


class TestPlaylistIdExtraction:
    """Test the static _extract_playlist_id method."""

    @pytest.fixture
    def parser(self):
        return YouTubePlaylistParser()

    def test_standard_playlist_url(self, parser):
        url = "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"
        assert parser._extract_playlist_id(url) == "PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"

    def test_url_with_extra_params(self, parser):
        url = "https://youtube.com/watch?v=abc123&list=PLtest123&index=2"
        assert parser._extract_playlist_id(url) == "PLtest123"

    def test_short_url_format(self, parser):
        url = "https://youtu.be/abc?list=PLshorttest"
        assert parser._extract_playlist_id(url) == "PLshorttest"

    def test_invalid_url_returns_none(self, parser):
        assert parser._extract_playlist_id("https://youtube.com/watch?v=abc") is None
        assert parser._extract_playlist_id("not a url") is None
        assert parser._extract_playlist_id("") is None


class TestPlaylistParsing:
    """Test full playlist parsing with mocked yt-dlp."""

    @pytest.fixture
    def parser(self):
        return YouTubePlaylistParser()

    @pytest.fixture
    def mock_ydl_info(self):
        """Simulated yt-dlp response for a playlist."""
        return {
            "_type": "playlist",
            "title": "Test Course Playlist",
            "description": "A test course",
            "uploader": "Test Channel",
            "thumbnails": [{"url": "https://thumb.jpg", "width": 1280, "height": 720}],
            "entries": [
                {
                    "id": "video001",
                    "title": "Lecture 1 — Introduction",
                    "duration": 600,
                    "thumbnails": [{"url": "https://thumb1.jpg", "width": 640, "height": 360}],
                },
                {
                    "id": "video002",
                    "title": "Lecture 2 — Core Concepts",
                    "duration": 900,
                    "thumbnails": [],
                },
                None,   # Simulates an unavailable video
            ],
        }

    def test_returns_playlist_meta(self, parser, mock_ydl_info):
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = mock_ydl_info
            result = parser.parse("https://youtube.com/playlist?list=PLtest123")

        assert result.title == "Test Course Playlist"
        assert result.playlist_id == "PLtest123"
        assert result.channel_name == "Test Channel"

    def test_skips_unavailable_videos(self, parser, mock_ydl_info):
        """None entries (unavailable videos) should be skipped."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = mock_ydl_info
            result = parser.parse("https://youtube.com/playlist?list=PLtest123")

        assert result.total_videos == 2   # 3 entries, 1 None → 2 videos

    def test_video_positions_are_sequential(self, parser, mock_ydl_info):
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = mock_ydl_info
            result = parser.parse("https://youtube.com/playlist?list=PLtest123")

        positions = [v.position for v in result.videos]
        assert positions == [1, 2]

    def test_video_urls_are_well_formed(self, parser, mock_ydl_info):
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = mock_ydl_info
            result = parser.parse("https://youtube.com/playlist?list=PLtest123")

        for video in result.videos:
            assert video.url.startswith("https://www.youtube.com/watch?v=")
            assert video.video_id in video.url

    def test_raises_for_non_playlist(self, parser):
        non_playlist_info = {"_type": "video", "title": "Just a video"}
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = non_playlist_info
            with pytest.raises(ValueError, match="does not point to a playlist"):
                parser.parse("https://youtube.com/playlist?list=PLtest123")

    def test_raises_for_invalid_url(self, parser):
        with pytest.raises(ValueError, match="Could not extract playlist ID"):
            parser.parse("https://youtube.com/watch?v=notaplaylist")
