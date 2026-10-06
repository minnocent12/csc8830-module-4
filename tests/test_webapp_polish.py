from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from streamlit.testing.v1 import AppTest

# Absolute, so the path means the same thing on every supported Streamlit version: 1.47
# resolves a relative path against the working directory first, 1.64 against this file.
APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def _encoded_png(image: np.ndarray) -> tuple[str, bytes, str]:
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    return ("input.png", encoded.tobytes(), "image/png")


def test_rgb_page_explains_question_mapping_and_runs_bundled_sample() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("RGB Human Boundary").run()

    assert not app.exception
    assert any("Question 1" in item.value for item in app.header)
    # The explanation is the page description (ordinary text), not an alert.
    assert any("classical OpenCV" in item.value for item in app.markdown)
    # No upload: the committed AAU VAP frame is processed live instead of a pending banner.
    assert any("bundled real frame" in item.value for item in app.info)
    assert not any("Pending user data" in item.value for item in app.warning)
    assert any("aau_vap_scene1_00085.jpg" in item.value for item in app.caption)
    # ROI is prefilled with the real predefined Phase 8 location, and the run gate is skipped.
    roi = {item.label: item.value for item in app.number_input}
    assert roi == {"x": 120, "y": 20, "width": 450, "height": 460}
    assert not app.button
    # The pipeline summary is kept as a labeled subsection inside Processing Results.
    assert "Processing Results" in [item.value for item in app.subheader]
    assert "**Processing sequence**" in [item.value for item in app.markdown]


def test_thermal_page_explains_dual_polarity_classical_processing() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Thermal Human Boundary").run()

    assert not app.exception
    assert any("Question 2" in item.value for item in app.header)
    # The explanation is the page description (ordinary text), not an alert.
    assert any("bright and dark" in item.value for item in app.markdown)


def test_comparison_page_evaluates_bundled_ground_truth_with_real_metrics() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Comparison and Evaluation").run()

    assert not app.exception
    assert any("Supporting evaluation" in item.value for item in app.header)
    # No upload: the committed frame and its real ground-truth mask are used as the reference.
    assert any("bundled real frame" in item.value for item in app.info)
    assert any("bundled real ground-truth mask" in item.value for item in app.info)
    assert not any("Pending user data" in item.value for item in app.warning)
    assert app.radio(key="comparison_reference_source").value == "Uploaded reference mask"
    assert not app.button
    # Metrics come from a real comparison, so each one is present and a genuine ratio, never
    # a placeholder zero.
    metrics = {item.label: float(item.value) for item in app.metric}
    assert {"IoU", "Dice", "Precision", "Recall"} <= set(metrics)
    for label in ("IoU", "Dice", "Precision", "Recall"):
        assert 0.0 < metrics[label] <= 1.0


def test_fourier_page_separates_theory_and_educational_demonstration() -> None:
    app = AppTest.from_file(APP_PATH).run()
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
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("RGB Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(rgb)).run()
    app.button[0].click().run()
    assert not app.exception
    assert "**Processing sequence**" in [item.value for item in app.markdown]

    thermal = np.zeros((48, 48), dtype=np.uint8)
    thermal[12:36, 16:32] = 220
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Thermal Human Boundary").run()
    app.file_uploader[0].set_value(_encoded_png(thermal)).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Processing Results" in item.value for item in app.subheader)
