"""Pre-run banner is neutral info; real warnings still render as warnings."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


# ---------------------------------------------------------------------------
# Helper unit-level checks (no AppTest page render needed)
# ---------------------------------------------------------------------------

def test_pending_banner_uses_info_not_warning() -> None:
    """The helper must render st.info, which AppTest captures in app.info."""
    from streamlit.testing.v1 import AppTest as _AT
    script = (
        "import sys; sys.path.insert(0, r'" + str(SRC) + "')\n"
        "from module4.webapp.ui import pending_experiment_banner\n"
        "pending_experiment_banner('Upload to begin.')\n"
    )
    app = _AT.from_string(script).run()
    assert not app.exception
    assert any("Upload to begin." in item.value for item in app.info)
    assert not any("Upload to begin." in item.value for item in app.warning)


def test_pending_banner_text_has_no_status_prefix() -> None:
    """The old 'Status: Pending user data' prefix must be gone."""
    from module4.webapp.ui import pending_experiment_banner as _b
    import inspect
    source = inspect.getsource(_b)
    assert "Status: Pending" not in source
    assert "Pending user data" not in source


def test_pending_banner_text_has_no_reference_mask_copy() -> None:
    """The old generic 'No empirical result or reference mask is available yet.' copy must be gone."""
    from module4.webapp.ui import pending_experiment_banner as _b
    import inspect
    source = inspect.getsource(_b)
    assert "empirical result" not in source


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def _skip_if_no_upload_api() -> None:
    import streamlit
    ver = tuple(int(x) for x in streamlit.__version__.split(".")[:2])
    if ver < (1, 56):
        pytest.skip("this Streamlit version's AppTest cannot simulate file uploads")


def _encoded_png(image: np.ndarray) -> tuple[str, bytes, str]:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return ("test.png", buf.tobytes(), "image/png")


def _small_rgb() -> np.ndarray:
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[12:36, 16:32] = (40, 120, 220)
    return img


def _small_thermal() -> np.ndarray:
    img = np.zeros((48, 48), dtype=np.uint8)
    img[12:36, 16:32] = 220
    return img


# ---------------------------------------------------------------------------
# RGB: uploaded but Run not yet clicked
# ---------------------------------------------------------------------------

def test_rgb_uploaded_before_run_shows_info_not_warning() -> None:
    _skip_if_no_upload_api()
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("RGB Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(_small_rgb())).run()

    assert not app.exception
    # run gate is holding: no run button was clicked, so no Processing Results section
    subheaders = [item.value for item in app.subheader]
    assert "Processing Results" not in subheaders
    # the waiting guidance is info, not a warning
    assert not any("Pending user data" in item.value for item in app.warning)
    assert any("Run" in item.value for item in app.info)


def test_rgb_uploaded_before_run_has_no_pipeline_output() -> None:
    _skip_if_no_upload_api()
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("RGB Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(_small_rgb())).run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert "Processing Results" not in subheaders
    assert "Interpretation" not in subheaders


# ---------------------------------------------------------------------------
# Thermal: uploaded but Run not yet clicked
# ---------------------------------------------------------------------------

def test_thermal_uploaded_before_run_shows_info_not_warning() -> None:
    _skip_if_no_upload_api()
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Thermal Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(_small_thermal())).run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert "Processing Results" not in subheaders
    assert not any("Pending user data" in item.value for item in app.warning)
    assert any("Run" in item.value for item in app.info)


def test_thermal_uploaded_before_run_has_no_pipeline_output() -> None:
    _skip_if_no_upload_api()
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Thermal Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(_small_thermal())).run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert "Processing Results" not in subheaders
    assert "Interpretation" not in subheaders


# ---------------------------------------------------------------------------
# Comparison: uploaded but Run not yet clicked
# ---------------------------------------------------------------------------

def test_comparison_uploaded_before_run_shows_info_not_warning() -> None:
    _skip_if_no_upload_api()
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Comparison and Evaluation").run()
    app.file_uploader[0].set_value(_encoded_png(_small_rgb())).run()

    assert not app.exception
    assert not any("Pending user data" in item.value for item in app.warning)
    assert any("Run" in item.value for item in app.info)


# ---------------------------------------------------------------------------
# Real warnings must stay warnings
# ---------------------------------------------------------------------------

def test_thermal_false_color_palette_warning_is_still_a_warning() -> None:
    """BGR false-color palette input must still produce a warning, not become info."""
    _skip_if_no_upload_api()
    # A 3-channel image triggers the false-color-palette warning in the thermal pipeline.
    three_channel = np.zeros((48, 48, 3), dtype=np.uint8)
    three_channel[12:36, 16:32] = (40, 120, 220)
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Thermal Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(three_channel)).run()

    assert not app.exception
    warnings = [item.value for item in app.warning]
    assert any("false-color" in w or "palette" in w.lower() for w in warnings), (
        "False-color palette warning must still be a warning after the banner change"
    )
