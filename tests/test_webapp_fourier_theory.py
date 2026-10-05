"""Fourier Theory page: theory-first structure, exact mathematics, and unchanged demonstration.

The LaTeX list below is the page's mathematics verbatim; any change to an equation, its
order, or its count fails here. No test uploads a file, so the suite runs on every supported
Streamlit version.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

SRC = str(Path(__file__).resolve().parents[1] / "src")
PAGE_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages
pages._theory_page()
"""

EQUATIONS = [
    r"F(u,v)=\int\!\!\int f(x,y)e^{-j2\pi(ux+vy)}\,dx\,dy",
    r"f(x,y)=\int\!\!\int F(u,v)e^{j2\pi(ux+vy)}\,du\,dv",
    r"F[k,l]=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}f[m,n]e^{-j2\pi(km/M+ln/N)}",
    r"f[m,n]=\frac{1}{MN}\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}F[k,l]e^{j2\pi(km/M+ln/N)}",
    r"G(u,v)=H(u,v)F(u,v),\qquad g(x,y)=\mathcal{F}^{-1}\{G(u,v)\}",
    r"H_{HP}(u,v)=1-H_{LP}(u,v)",
    r"G(u,v)=H_{HP}(u,v)F(u,v),\qquad g=\mathcal{F}^{-1}\{G\}",
    r"\mathcal{F}\{\partial f/\partial x\}=j2\pi uF(u,v)",
    r"\mathcal{F}\{\partial f/\partial y\}=j2\pi vF(u,v)",
    r"\nabla^2f=\partial^2f/\partial x^2+\partial^2f/\partial y^2",
    r"\mathcal{F}\{\nabla^2f\}=-4\pi^2(u^2+v^2)F(u,v)",
    r"F_k(u,v)=\mathcal{F}\{f_k(x,y)\}",
]
PARTS = [
    "Part A: 2D Fourier representation",
    "Part B: Why edges are high frequency",
    "Part C: Gaussian high-pass filtering",
    "Part D: Fourier derivative property",
    "Part E: Frequency-domain Laplacian",
    "Part F: Fourier-domain region segmentation",
]


@pytest.fixture(scope="module")
def page() -> AppTest:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    assert not app.exception
    return app


def _latex(app: AppTest) -> list[str]:
    # st.latex wraps the source as "$$\n...\n$$"; compare the source itself.
    return [e.value.removeprefix("$$\n").removesuffix("\n$$") for e in app.latex]


def test_header_is_canonical_and_shown_once(page: AppTest) -> None:
    assert [h.value for h in page.header] == [
        "Question 3: Fourier-Domain Edge Detection and Region Segmentation"
    ]
    assert "Module 4" in " ".join(e.proto.body for e in page.get("html"))


def test_theory_comes_first_then_demonstration_results_and_interpretation(page: AppTest) -> None:
    assert [s.value for s in page.subheader] == [
        "Theory", *PARTS, "Demonstration", "Results", "Interpretation",
    ]


def test_every_equation_is_unchanged_and_in_order(page: AppTest) -> None:
    assert _latex(page) == EQUATIONS


def test_demonstration_controls_keep_keys_labels_and_defaults(page: AppTest) -> None:
    assert [(s.key, s.label, s.value) for s in page.slider] == [
        ("fourier_sigma", "Gaussian low-pass sigma (cycles per pixel)", 0.08),
        ("fourier_cutoff", "Selected high-frequency cutoff (cycles per pixel)", 0.15),
        ("fourier_energy_threshold", "Educational display threshold as fraction of maximum energy", 0.5),
    ]
    (window,) = page.select_slider
    assert (window.key, window.label, window.value, list(window.options)) == (
        "fourier_window", "Local analysis window size", 8, ["4", "8", "16", "32"],
    )
    assert not page.button  # the demonstration has no run gate; it updates live as before


def test_outputs_are_grouped_by_part_in_the_original_order(page: AppTest) -> None:
    # The image element is "image" on newer Streamlit and "imgs" on older; both are ImageLists.
    elements = page.get("image") or page.get("imgs")
    captions = [img.proto.imgs[0].caption for img in elements]
    assert captions == [
        "Scalar image used for Fourier analysis",
        "Centered log magnitude spectrum: log(1 + |F|)",
        "Gaussian low-pass reconstruction",
        "Gaussian high-pass response",
        "Fourier x derivative",
        "Fourier y derivative",
        "Fourier-domain Laplacian response",
        "Local high-frequency energy map",
        "Thresholded educational frequency-region map",
    ]


def test_synthetic_input_notice_is_an_explanation_not_a_status_badge(page: AppTest) -> None:
    assert any("Educational Demonstration" in item.value for item in page.info)
    assert not page.success
    assert not any("Demonstration status" in item.value for item in page.info)


def test_interpretation_is_ordinary_text(page: AppTest) -> None:
    assert any(
        "not a semantic segmentation result or empirical assignment measurement" in m.value
        for m in page.markdown
    )


def test_parameter_change_still_recomputes_live() -> None:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    app.slider(key="fourier_sigma").set_value(0.2).run()
    assert not app.exception
    assert _latex(app) == EQUATIONS
