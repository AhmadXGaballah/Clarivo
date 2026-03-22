import os
import re
import sys
import uuid
import shutil
import tempfile
import subprocess
from typing import List, Optional, Tuple

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from faster_whisper import WhisperModel


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Clarivo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar completely
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Header
# =========================================================
st.title("✦ Clarivo")
st.caption("Turn long content into clear decisions.")

MODEL_OPTIONS = {
    "Instant": "philschmid/flan-t5-base-samsum",
    "Performance": "stacked-summaries/flan-t5-large-samsum",
    "Long Video": "pszemraj/long-t5-tglobal-base-16384-book-summary",
    
}

mode_label = st.selectbox(
    "Summarization mode",
    list(MODEL_OPTIONS.keys()),
    index=0,
)

MODEL_NAME = MODEL_OPTIONS[mode_label]

# =========================================================
# Defaults 
# =========================================================
WHISPER_SIZE = "base"
WHISPER_LANGUAGE = None  # Auto-detect


# =========================================================
# Session state init
# =========================================================
def init_state() -> None:
    defaults = {
        "yt_transcript": "",
        "yt_summary": "",
        "yt_detected_language": "",
        "last_url": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# =========================================================
# Device helpers
# =========================================================
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


# =========================================================
# Cached model loaders
# =========================================================
@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if TORCH_DEVICE in {"cuda", "mps"}:
        model = model.to(TORCH_DEVICE)

    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_whisper(whisper_size: str = WHISPER_SIZE):
    return WhisperModel(
        whisper_size,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


# =========================================================
# Text utilities
# =========================================================
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("<n>", "\n")
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+!", "!", text)
    text = re.sub(r"\s+\?", "?", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])


def split_long_text(text: str, tokenizer, max_input_len: int = 768) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current = ""

    def push_chunk(chunk: str):
        chunk = clean_text(chunk)
        if chunk:
            chunks.append(chunk)

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para

        if token_count(tokenizer, candidate) <= max_input_len:
            current = candidate
            continue

        if current:
            push_chunk(current)
            current = ""

        if token_count(tokenizer, para) <= max_input_len:
            current = para
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        sent_buf = ""

        for sent in sentences:
            sent_candidate = f"{sent_buf} {sent}".strip() if sent_buf else sent

            if token_count(tokenizer, sent_candidate) <= max_input_len:
                sent_buf = sent_candidate
            else:
                if sent_buf:
                    push_chunk(sent_buf)
                sent_buf = sent

        if sent_buf:
            push_chunk(sent_buf)

    if current:
        push_chunk(current)

    return chunks


# =========================================================
# Summary length presets
# =========================================================
def get_summary_settings(mode: str) -> dict:
    presets = {
        "Short": {
            "min_new_tokens": 20,
            "max_new_tokens": 45,
            "length_penalty": 0.8,
        },
        "Medium": {
            "min_new_tokens": 35,
            "max_new_tokens": 70,
            "length_penalty": 1.0,
        },
        "Long": {
            "min_new_tokens": 55,
            "max_new_tokens": 95,
            "length_penalty": 1.05,
        },
        "Detailed": {
            "min_new_tokens": 70,
            "max_new_tokens": 120,
            "length_penalty": 1.1,
        },
    }
    return presets[mode]

def build_summary_prompt(text: str, mode: str) -> str:
    instructions = {
        "Short": (
            "Summarize the following educational transcript in 2 concise sentences. "
            "Include only information explicitly mentioned in the transcript. "
            "Do not add outside information.\n\nTranscript:\n"
        ),
        "Medium": (
            "Summarize the following educational transcript in 3 to 4 concise sentences. "
            "Focus only on the key concepts explicitly mentioned. "
            "Do not add outside information.\n\nTranscript:\n"
        ),
        "Long": (
            "Summarize the following educational transcript in a clear paragraph. "
            "Cover the main concepts and definitions mentioned in the transcript only. "
            "Do not add outside information.\n\nTranscript:\n"
        ),
        "Detailed": (
            "Summarize the following educational transcript in 5 bullet points. "
            "Use only facts explicitly stated in the transcript. "
            "Do not add outside information, examples, or assumptions.\n\nTranscript:\n"
        ),
    }
    return instructions[mode] + text

def render_summary_length_control(key: str):
    st.markdown("#### Summary length")
    mode = st.radio(
        "Summary length",
        ["Short", "Medium", "Long", "Detailed"],
        index=1,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )
    return mode, get_summary_settings(mode)


# =========================================================
# Summarization
# =========================================================
def summarize_chunk(
    tokenizer,
    model,
    text: str,
    mode: str = "Medium",
    max_input_len: int = 768,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    min_new_tokens: int = 35,
    max_new_tokens: int = 70,
) -> str:
    prompt = build_summary_prompt(text, mode)

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=max_input_len,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,
            length_penalty=length_penalty,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return clean_text(summary)


def summarize_text(
    tokenizer,
    model,
    text: str,
    mode: str = "Medium",
    max_input_len: int = 768,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    min_new_tokens: int = 35,
    max_new_tokens: int = 70,
) -> str:
    text = clean_text(text)
    if not text:
        return ""

    current_text = text

    for _ in range(3):
        chunks = split_long_text(current_text, tokenizer, max_input_len=max_input_len)

        if not chunks:
            return ""

        if len(chunks) == 1:
            return summarize_chunk(
                tokenizer=tokenizer,
                model=model,
                text=chunks[0],
                mode=mode,
                max_input_len=max_input_len,
                num_beams=num_beams,
                length_penalty=length_penalty,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
            )

        partial_summaries = [
            summarize_chunk(
                tokenizer=tokenizer,
                model=model,
                text=chunk,
                mode=mode,
                max_input_len=max_input_len,
                num_beams=num_beams,
                length_penalty=length_penalty,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
            )
            for chunk in chunks
        ]

        current_text = "\n".join(partial_summaries)

        if token_count(tokenizer, current_text) <= max_input_len:
            return summarize_chunk(
                tokenizer=tokenizer,
                model=model,
                text=current_text,
                mode=mode,
                max_input_len=max_input_len,
                num_beams=num_beams,
                length_penalty=length_penalty,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
            )

    return summarize_chunk(
        tokenizer=tokenizer,
        model=model,
        text=current_text,
        mode=mode,
        max_input_len=max_input_len,
        num_beams=num_beams,
        length_penalty=length_penalty,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
    )


# =========================================================
# YouTube download + transcription
# =========================================================
def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not available on PATH.")


def download_youtube_audio(url: str, work_dir: str) -> str:
    require_ffmpeg()
    os.makedirs(work_dir, exist_ok=True)

    base = f"yt_{uuid.uuid4().hex}"
    out_template = os.path.join(work_dir, f"{base}.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bestaudio",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        out_template,
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"yt-dlp failed.\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
        ) from e

    for name in os.listdir(work_dir):
        if name.startswith(base) and name.endswith(".mp3"):
            return os.path.join(work_dir, name)

    raise RuntimeError(f"Audio download finished but MP3 was not found.\n\n{result.stdout}")


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




# =========================================================
# Load models
# =========================================================
try:
    with st.spinner("Loading models..."):
        tokenizer, summarizer_model = load_summarizer(MODEL_NAME)
        whisper_model = load_whisper(WHISPER_SIZE)
except Exception as e:
    st.error("Model loading failed.")
    st.exception(e)
    st.stop()

# =========================================================
# Tabs
# =========================================================
tab_text, tab_youtube = st.tabs(["Summarize Text", "Summarize YouTube Video"])


# =========================================================
# Tab 1: Text
# =========================================================
with tab_text:
    text_mode, text_settings = render_summary_length_control("text_summary_length")
    
    
    st.subheader("Paste your text")
    input_text = st.text_area(
        "Input text",
        height=260,
        placeholder="Paste article, notes, transcript, or report here...",
    )

    if st.button("Summarize Text", use_container_width=True):
        if not input_text.strip():
            st.warning("Please paste some text first.")
        else:
            try:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(
                        tokenizer=tokenizer,
                        model=summarizer_model,
                        text=input_text,
                        mode=text_mode,
                        length_penalty=text_settings["length_penalty"],
                        min_new_tokens=text_settings["min_new_tokens"],
                        max_new_tokens=text_settings["max_new_tokens"],
                    )
                    
                st.success("Done.")
                st.markdown("### Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")


# =========================================================
# Tab 2: YouTube
# =========================================================
with tab_youtube:
    yt_mode, yt_settings = render_summary_length_control("yt_summary_length")

    st.subheader("Enter a YouTube URL")
    url = st.text_input(
        "YouTube link",
        placeholder="https://www.youtube.com/watch?v=...",
        value=st.session_state["last_url"],
    )
    st.session_state["last_url"] = url

    col1, col2, col3 = st.columns(3)
    transcribe_btn = col1.button("Transcribe Audio", use_container_width=True)
    full_btn = col2.button("Transcribe then Summarize", use_container_width=True)
    

    
    if transcribe_btn or full_btn:
        if not url.strip():
            st.warning("Please paste a valid YouTube URL.")
        else:
            try:
                with st.spinner("Transcribing..."):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        audio_path = download_youtube_audio(url, tmpdir)
                        transcript, detected_lang = transcribe_audio(
                            whisper_model=whisper_model,
                            audio_path=audio_path,
                            language=WHISPER_LANGUAGE,
                        )

                    st.session_state["yt_transcript"] = transcript
                    st.session_state["yt_detected_language"] = detected_lang
                    st.session_state["yt_summary"] = ""

                if full_btn:
                    with st.spinner("Summarizing transcript..."):
                        st.session_state["yt_summary"] = summarize_text(
                            tokenizer=tokenizer,
                            model=summarizer_model,
                            text=st.session_state["yt_transcript"],
                            mode=yt_mode,
                            length_penalty=yt_settings["length_penalty"],
                            min_new_tokens=yt_settings["min_new_tokens"],
                            max_new_tokens=yt_settings["max_new_tokens"],
                        )   

                st.success("Done.")

            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    if st.session_state["yt_detected_language"]:
        st.caption(f"Detected language: {st.session_state['yt_detected_language']}")

    st.markdown("### Transcript")
    edited_transcript = st.text_area(
        "Transcript",
        height=280,
        value=st.session_state["yt_transcript"],
    )
    st.session_state["yt_transcript"] = edited_transcript

    if st.button("Summarize Transcript", use_container_width=True):
        if not edited_transcript.strip():
            st.warning("Transcript is empty.")
        else:
            try:
                with st.spinner("Summarizing transcript..."):
                   st.session_state["yt_summary"] = summarize_text(
                        tokenizer=tokenizer,
                        model=summarizer_model,
                        text=st.session_state["yt_transcript"],
                        mode=yt_mode,
                        length_penalty=yt_settings["length_penalty"],
                        min_new_tokens=yt_settings["min_new_tokens"],
                        max_new_tokens=yt_settings["max_new_tokens"],
                    )
                st.success("Summary ready.")
            except Exception as e:
                st.error(f"Summarization failed: {e}")

    if st.session_state["yt_summary"]:
        st.markdown("### Summary")
        st.write(st.session_state["yt_summary"])

        st.download_button(
            "Download Summary as TXT",
            data=st.session_state["yt_summary"],
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


st.divider()
st.caption("Powered by FLAN-T5 + Faster-Whisper")
st.caption("Developed by Ahmad Gaballah")