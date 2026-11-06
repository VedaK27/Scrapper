import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)


def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from common YouTube URL formats.
    """

    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise ValueError("Could not extract YouTube video ID from URL.")


def fetch_transcript(video_id: str):
    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(
            video_id,
            languages=["mr"]
        )

        print("✓ Marathi transcript found.")

        return transcript

    except NoTranscriptFound:
        raise RuntimeError(
            "No Marathi transcript was found for this video."
        )

    except TranscriptsDisabled:
        raise RuntimeError(
            "Transcripts are disabled for this video."
        )


