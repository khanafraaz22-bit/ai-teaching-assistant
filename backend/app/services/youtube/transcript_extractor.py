"""
Transcript Extractor
--------------------
Fetches transcripts for YouTube videos.

Strategy (in order):
  1. YouTube's native transcript (free, fast, has timestamps)
  2. Auto-generated captions (also from YouTube, slightly lower quality)
  3. Whisper fallback — downloads audio and transcribes locally
     (slower, used only when YouTube has no transcript at all)

Output is always a list of TranscriptSegment with normalized timestamps.
"""

import logging
import tempfile
import os
from typing import Optional
from dataclasses import dataclass

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    text: str
    start: float        # seconds from video start
    duration: float     # seconds this segment lasts

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class RawTranscript:
    video_id: str
    segments: list[TranscriptSegment]
    language: str
    source: str         # "youtube_manual" | "youtube_auto" | "whisper"

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


# ── Extractor ─────────────────────────────────────────────────────────────────

class TranscriptExtractor:
    """
    Extracts transcripts from YouTube videos with fallback logic.

    Usage:
        extractor = TranscriptExtractor(use_whisper_fallback=True)
        transcript = extractor.extract("dQw4w9WgXcQ")
    """

    PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]

    def __init__(self, use_whisper_fallback: bool = False):
        """
        Args:
            use_whisper_fallback: If True, download audio and use OpenAI Whisper
                                  when YouTube has no transcript. Slower but thorough.
        """
        self.use_whisper_fallback = use_whisper_fallback

    def extract(self, video_id: str) -> Optional[RawTranscript]:
        """
        Extract transcript for a single video.
        Returns None if no transcript is available and fallback is disabled.
        """
        # Strategy 1 & 2: YouTube API (manual first, then auto-generated)
        transcript = self._fetch_from_youtube(video_id)
        if transcript:
            return transcript

        # Strategy 3: Whisper fallback
        if self.use_whisper_fallback:
            logger.info(f"[{video_id}] Trying Whisper fallback...")
            return self._transcribe_with_whisper(video_id)

        logger.warning(f"[{video_id}] No transcript available.")
        return None

    # ── YouTube API ───────────────────────────────────────────────────────────

    def _fetch_from_youtube(self, video_id: str) -> Optional[RawTranscript]:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except (VideoUnavailable, TranscriptsDisabled) as e:
            logger.warning(f"[{video_id}] YouTube transcript unavailable: {e}")
            return None
        except Exception as e:
            logger.error(f"[{video_id}] Unexpected error listing transcripts: {e}")
            return None

        # Prefer manually created transcripts in English
        transcript_obj = None
        source = "youtube_auto"

        try:
            transcript_obj = transcript_list.find_manually_created_transcript(
                self.PREFERRED_LANGUAGES
            )
            source = "youtube_manual"
            logger.info(f"[{video_id}] Found manual transcript ({transcript_obj.language_code})")
        except NoTranscriptFound:
            pass

        # Fall back to auto-generated
        if not transcript_obj:
            try:
                transcript_obj = transcript_list.find_generated_transcript(
                    self.PREFERRED_LANGUAGES
                )
                source = "youtube_auto"
                logger.info(f"[{video_id}] Found auto-generated transcript ({transcript_obj.language_code})")
            except NoTranscriptFound:
                pass

        # Try translating any available transcript to English as last resort
        if not transcript_obj:
            try:
                available = list(transcript_list)
                if available:
                    transcript_obj = available[0].translate("en")
                    source = "youtube_translated"
                    logger.info(f"[{video_id}] Using translated transcript from {available[0].language_code}")
            except Exception:
                pass

        if not transcript_obj:
            return None

        try:
            raw_data = transcript_obj.fetch()
        except Exception as e:
            logger.error(f"[{video_id}] Failed to fetch transcript data: {e}")
            return None

        segments = [
            TranscriptSegment(
                text=entry.get("text", "").strip(),
                start=float(entry.get("start", 0)),
                duration=float(entry.get("duration", 0)),
            )
            for entry in raw_data
            if entry.get("text", "").strip()
        ]

        if not segments:
            logger.warning(f"[{video_id}] Transcript fetched but had no content.")
            return None

        return RawTranscript(
            video_id=video_id,
            segments=segments,
            language=transcript_obj.language_code,
            source=source,
        )

    # ── Whisper fallback ──────────────────────────────────────────────────────

    def _transcribe_with_whisper(self, video_id: str) -> Optional[RawTranscript]:
        """
        Download audio with yt-dlp, transcribe with OpenAI Whisper.
        Requires: pip install openai-whisper and ffmpeg installed.
        """
        try:
            import whisper          # type: ignore  (optional dependency)
            import yt_dlp
        except ImportError:
            logger.error("Whisper fallback requires: pip install openai-whisper")
            return None

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = os.path.join(tmpdir, f"{video_id}.mp3")

                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": audio_path,
                    "quiet": True,
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                    }],
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

                # Use base model for speed; swap to "small" or "medium" for quality
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, verbose=False)

                segments = [
                    TranscriptSegment(
                        text=seg["text"].strip(),
                        start=seg["start"],
                        duration=seg["end"] - seg["start"],
                    )
                    for seg in result.get("segments", [])
                    if seg.get("text", "").strip()
                ]

                logger.info(f"[{video_id}] ✅ Whisper transcription complete. {len(segments)} segments.")
                return RawTranscript(
                    video_id=video_id,
                    segments=segments,
                    language="en",
                    source="whisper",
                )

        except Exception as e:
            logger.error(f"[{video_id}] Whisper transcription failed: {e}")
            return None
