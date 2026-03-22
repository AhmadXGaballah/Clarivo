"""Text cleaning, token counting, and chunking helpers."""

import re
from typing import List


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

    def push_chunk(chunk: str) -> None:
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
