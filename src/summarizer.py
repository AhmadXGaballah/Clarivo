"""Summarization model loading and inference helpers."""

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import TORCH_DEVICE
from .text_utils import clean_text, split_long_text, token_count


@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if TORCH_DEVICE in {"cuda", "mps"}:
        model = model.to(TORCH_DEVICE)

    model.eval()
    return tokenizer, model


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
