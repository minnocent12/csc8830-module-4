"""Streamlit rendering shell for the Module 4 page provider."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import streamlit as st

from module4.webapp._page import PageSpec

_PAGE_CONTEXT = {
    "RGB Human Boundary": "Question 1 · classical OpenCV RGB segmentation",
    "Thermal Human Boundary": "Question 2 · classical OpenCV thermal segmentation",
    "Comparison and Evaluation": "Supporting comparison · classical mask versus reference mask",
    "Fourier Theory": "Question 3 · Fourier Parts A–F and demonstrations",
}


def render_app(pages: Sequence[PageSpec], *, title: str = "CSc 8830 - Module 4") -> None:
    """Render a sidebar page selector and dispatch the selected page."""
    st.set_page_config(page_title=title, layout="wide")
    if not pages:
        st.error("No pages registered.")
        return

    by_module: dict[str, list[PageSpec]] = defaultdict(list)
    for page in pages:
        by_module[page.module_label].append(page)

    with st.sidebar:
        st.title(title)
        st.caption("CSc 8830 · Module 4")
        module_label = st.selectbox("Module", list(by_module))
        module_pages = by_module[module_label]
        page_label = st.radio("Page", [page.page_label for page in module_pages])
        st.divider()
        st.caption(_PAGE_CONTEXT.get(page_label, "Module 4 assignment component"))

    selected = next(page for page in module_pages if page.page_label == page_label)
    selected.render()
