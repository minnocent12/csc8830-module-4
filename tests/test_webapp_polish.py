from __future__ import annotations

import cv2
import numpy as np

from streamlit.testing.v1 import AppTest


def _encoded_png(image: np.ndarray) -> tuple[str, bytes, str]:
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    return ("input.png", encoded.tobytes(), "image/png")


def test_rgb_page_explains_question_mapping_and_pending_input_state() -> None:
    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("RGB Human Boundary").run()

    assert not app.exception
    assert any("Question 1" in item.value for item in app.header)
    assert any("classical OpenCV" in item.value for item in app.info)
    assert any("Pending user data" in item.value for item in app.warning)


def test_thermal_page_explains_dual_polarity_classical_processing() -> None:
    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("Thermal Human Boundary").run()

    assert not app.exception
    assert any("Question 2" in item.value for item in app.header)
    assert any("bright and dark" in item.value for item in app.info)


def test_comparison_page_labels_pending_reference_without_zero_metrics() -> None:
    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("Comparison and Evaluation").run()

    assert not app.exception
    assert any("Supporting evaluation" in item.value for item in app.header)
    assert any("Pending user data" in item.value for item in app.warning)
    assert not app.metric


def test_fourier_page_separates_theory_and_educational_demonstration() -> None:
    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("Fourier Theory").run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert "Theory" in subheaders
    assert "Demonstration" in subheaders
    assert any("Part A" in value for value in subheaders)
    assert any("Part F" in value for value in subheaders)
    assert any("Educational Demonstration" in item.value for item in app.info)


def test_rgb_and_thermal_pages_run_their_major_ui_sections() -> None:
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[12:36, 16:32] = (40, 120, 220)
    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("RGB Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(rgb)).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Processing sequence" in item.value for item in app.subheader)

    thermal = np.zeros((48, 48), dtype=np.uint8)
    thermal[12:36, 16:32] = 220
    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("Thermal Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(thermal)).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Classical thermal intermediate results" in item.value for item in app.subheader)
