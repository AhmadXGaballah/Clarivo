"""Whisper model loading and audio transcription helpers."""

from typing import Optional, Tuple

import streamlit as st
from faster_whisper import WhisperModel

from .config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE
from .text_utils import clean_text


@st.cache_resource(show_spinner=False)
def load_whisper(whisper_size: str):
    return WhisperModel(
        whisper_size,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


def transcribe_audio(
    whisper_model: WhisperModel,
    audio_path: str,
    language: Optional[str] = None,
) -> Tuple[str, str]:
    segments, info = whisper_model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    segments = list(segments)
    parts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
    transcript = clean_text(" ".join(parts))

    if not transcript:
        raise RuntimeError("No transcript could be generated from this audio.")

    detected_language = getattr(info, "language", "") or ""
    return transcript, detected_language
