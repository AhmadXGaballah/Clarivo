"""Reusable Streamlit UI helpers."""

import streamlit as st

from .summarizer import get_summary_settings


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
