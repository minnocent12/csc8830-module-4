"""Shared Streamlit presentation helpers for a consistent Module 4 application."""
from __future__ import annotations

import streamlit as st

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def page_header(
    title: str,
    *,
    assignment_label: str,
    summary: str,
    input_hint: str,
    status: str = "Implemented",
) -> None:
    """Render the common assignment mapping, purpose, input, and implementation status."""
    st.header(f"{assignment_label} — {title}")
    st.info(summary)
    st.caption(f"Expected input: {input_hint}")
    st.caption(f"Implementation status: **{status}**")


def status_message(label: str, state: str, detail: str) -> None:
    """Render a consistent, text-labeled status without relying on color alone."""
    normalized = state.strip().lower()
    message = f"**{label}: {state}**\n\n{detail}"
    if normalized in {"failed", "error"}:
        st.error(message)
    elif normalized in {"pending", "unavailable"}:
        st.warning(message)
    elif normalized in {"implemented", "available", "completed"}:
        st.success(message)
    else:
        st.info(message)


def pending_experiment_banner(detail: str | None = None) -> None:
    """Display an honest notice for work that needs later implementation or user data."""
    message = "**Status: Pending user data**\n\nNo empirical result or reference mask is available yet."
    st.warning(message if detail is None else f"{message}\n\n{detail}")


def bundled_sample_notice(detail: str) -> None:
    """Show the standard 'showing a bundled real sample; upload your own to override' notice."""
    st.info(detail)


def foundation_page(title: str, scope: str) -> None:
    """Render a pending-safe placeholder for any future page."""
    st.header(title)
    st.write(scope)
    pending_experiment_banner(
        "This page is structurally available now. Its computer-vision processing is scheduled "
        "for a later approved phase."
    )
