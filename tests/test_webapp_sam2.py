from __future__ import annotations

import cv2
import numpy as np

from streamlit.testing.v1 import AppTest


def test_comparison_page_reports_sam2_unavailable_without_metrics() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 12:20] = (10, 100, 200)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok

    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("Comparison and Evaluation").run()
    app.file_uploader[0].set_value(("input.png", encoded.tobytes(), "image/png")).run()
    app.radio[1].set_value("SAM2 reference").run()
    app.button[0].click().run()

    assert not app.exception
    assert any("SAM2 reference is unavailable" in item.value for item in app.warning)
    assert not any(item.label == "IoU" for item in app.metric)
