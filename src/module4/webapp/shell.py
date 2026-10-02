"""Streamlit shell for the Module 4 app: a thin adapter over the shared csc8830-ui shell.

Navigation, identity, breadcrumbs, and footer come from ``module4.webapp.design.shell``,
a vendored copy of the course design kit. This module only states what is specific here.
"""
from __future__ import annotations

from collections.abc import Sequence

from module4.webapp._page import PageSpec
from module4.webapp.design.shell import render_shell

_PAGE_CONTEXT = {
    "RGB Human Boundary": "Question 1 · classical OpenCV RGB segmentation",
    "Thermal Human Boundary": "Question 2 · classical OpenCV thermal segmentation",
    "Comparison and Evaluation": "Supporting comparison · classical mask versus reference mask",
    "Fourier Theory": "Question 3 · Fourier Parts A-F and demonstrations",
}


def _page_context(page: PageSpec) -> str:
    """Sidebar caption naming the assignment question a page answers."""
    return _PAGE_CONTEXT.get(page.page_label, "Module 4 assignment component")


def render_app(pages: Sequence[PageSpec], *, title: str = "CSc 8830 - Module 4") -> None:
    """Render the standalone Module 4 app around the selected page."""
    render_shell(pages, page_title=title, standalone=True, page_context=_page_context)
