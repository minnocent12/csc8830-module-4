"""Module 4 Streamlit pages and the get_pages provider.

The foundation phase exposes the final navigation shape with honest pending notices. The
segmentation and theory implementations are intentionally added in later approved phases.
"""
from __future__ import annotations

from module4.webapp._page import PageSpec
from module4.webapp.ui import foundation_page

_MODULE = "Module 4"


def _rgb_page() -> None:
    foundation_page(
        "RGB Human Boundary",
        "RGB image upload, classical preprocessing, human-region isolation, contour extraction, "
        "and reference comparison will be added in later phases.",
    )


def _thermal_page() -> None:
    foundation_page(
        "Thermal Human Boundary",
        "Thermal image loading, normalization, classical polarity handling, contour extraction, "
        "and reference comparison will be added in later phases.",
    )


def _comparison_page() -> None:
    foundation_page(
        "Comparison and Evaluation",
        "Mask alignment, IoU, Dice, precision, recall, visual comparisons, and result downloads "
        "will be added after the independent pipelines exist.",
    )


def _theory_page() -> None:
    foundation_page(
        "Fourier Theory",
        "The Parts A-F Fourier-domain derivation and optional demonstrations will be added in a "
        "later approved phase.",
    )


def get_pages() -> list[PageSpec]:
    """Return the Module 4 pages for standalone or shared-dashboard hosts."""
    return [
        PageSpec(_MODULE, "RGB Human Boundary", 10, _rgb_page),
        PageSpec(_MODULE, "Thermal Human Boundary", 20, _thermal_page),
        PageSpec(_MODULE, "Comparison and Evaluation", 30, _comparison_page),
        PageSpec(_MODULE, "Fourier Theory", 40, _theory_page),
    ]
