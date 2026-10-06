"""Shared Streamlit presentation helpers for a consistent Module 4 application."""
from __future__ import annotations

import streamlit as st

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def pending_experiment_banner(detail: str | None = None) -> None:
    """Display an honest notice for work that needs later implementation or user data."""
    message = "**Status: Pending user data**\n\nNo empirical result or reference mask is available yet."
    st.warning(message if detail is None else f"{message}\n\n{detail}")


def bundled_sample_notice(detail: str) -> None:
    """Show the standard 'showing a bundled real sample; upload your own to override' notice."""
    st.info(detail)

