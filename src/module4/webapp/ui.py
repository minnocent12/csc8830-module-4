"""Shared Streamlit presentation helpers for a consistent Module 4 application."""
from __future__ import annotations

import streamlit as st

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def pending_experiment_banner(text: str) -> None:
    """Show a neutral informational prompt for a normal pre-run waiting state."""
    st.info(text)


def bundled_sample_notice(detail: str) -> None:
    """Show the standard 'showing a bundled real sample; upload your own to override' notice."""
    st.info(detail)

