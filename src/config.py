"""Shared configuration and runtime helpers for Clarivo."""

from typing import Tuple
import torch

APP_NAME = "Clarivo"
TAGLINE = "Turn long content into clear decisions."

MODEL_OPTIONS = {
    "Instant": "philschmid/flan-t5-base-samsum",
    "Performance": "stacked-summaries/flan-t5-large-samsum",
    "Long Video": "pszemraj/long-t5-tglobal-base-16384-book-summary",
}

WHISPER_SIZE = "base"
WHISPER_LANGUAGE = None  # Auto-detect


def get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_whisper_runtime() -> Tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


TORCH_DEVICE = get_torch_device()
WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = get_whisper_runtime()
