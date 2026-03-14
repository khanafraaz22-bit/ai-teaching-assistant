"""
YouTube Playlist Parser
-----------------------
Extracts playlist metadata and video list from a YouTube playlist URL.
Uses yt-dlp for robust metadata extraction (handles age-restricted, private, etc.)

Returns structured data — no DB writes happen here (that's the ingestion service).
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass, field

import yt_dlp

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class VideoMeta:
    video_id: str
    title: str
    url: str
    position: int                        # 1-indexed order in the playlist
    duration_seconds: Optional[int]
    thumbnail_url: Optional[str]


@dataclass
class PlaylistMeta:
    playlist_id: str
    playlist_url: str
    title: str
    description: Optional[str]
    channel_name: Optional[str]
    thumbnail_url: Optional[str]
    videos: list[VideoMeta] = field(default_factory=list)

    @property
    def total_videos(self) -> int:
        return len(self.videos)


# ── Parser ────────────────────────────────────────────────────────────────────

class YouTubePlaylistParser:
    """
    Parses a YouTube playlist URL and returns structured metadata.

    Usage:
        parser = YouTubePlaylistParser()
        playlist = await parser.parse("https://youtube.com/playlist?list=XXXXX")
    """

    # yt-dlp options — extract info only, no downloads
    _YDL_OPTS = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",   # Fast: gets metadata without fetching each video page
        "skip_download": True,
        "ignoreerrors": True,            # Skip unavailable videos gracefully
    }

    def parse(self, playlist_url: str) -> PlaylistMeta:
        """
        Extract full playlist + video metadata.
        Raises ValueError for invalid URLs or private playlists.
        """
        playlist_id = self._extract_playlist_id(playlist_url)
        if not playlist_id:
            raise ValueError(f"Could not extract playlist ID from URL: {playlist_url}")

        logger.info(f"Parsing playlist: {playlist_id}")

        with yt_dlp.YoutubeDL(self._YDL_OPTS) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        if not info:
            raise ValueError(f"Could not fetch playlist info for: {playlist_url}")

        if info.get("_type") != "playlist":
            raise ValueError("URL does not point to a playlist.")

        videos = self._extract_videos(info)
        if not videos:
            raise ValueError("Playlist is empty or all videos are unavailable.")

        logger.info(f"✅ Parsed '{info.get('title')}' — {len(videos)} videos")

        return PlaylistMeta(
            playlist_id=playlist_id,
            playlist_url=playlist_url,
            title=info.get("title") or "Untitled Playlist",
            description=info.get("description"),
            channel_name=info.get("uploader") or info.get("channel"),
            thumbnail_url=self._best_thumbnail(info.get("thumbnails") or []),
            videos=videos,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_playlist_id(url: str) -> Optional[str]:
        """Pull the list= parameter from any YouTube playlist URL format."""
        patterns = [
            r"[?&]list=([A-Za-z0-9_-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_videos(info: dict) -> list[VideoMeta]:
        entries = info.get("entries") or []
        videos = []
        for position, entry in enumerate(entries, start=1):
            if not entry:
                continue                  # Skip unavailable entries
            video_id = entry.get("id") or entry.get("url", "").split("v=")[-1]
            if not video_id:
                continue
            videos.append(VideoMeta(
                video_id=video_id,
                title=entry.get("title") or f"Video {position}",
                url=f"https://www.youtube.com/watch?v={video_id}",
                position=position,
                duration_seconds=entry.get("duration"),
                thumbnail_url=YouTubePlaylistParser._best_thumbnail(
                    entry.get("thumbnails") or []
                ),
            ))
        return videos

    @staticmethod
    def _best_thumbnail(thumbnails: list[dict]) -> Optional[str]:
        """Return the highest-resolution thumbnail URL."""
        if not thumbnails:
            return None
        # Sort by resolution if available, else take last (usually highest res)
        sorted_thumbs = sorted(
            thumbnails,
            key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
            reverse=True,
        )
        return sorted_thumbs[0].get("url")
