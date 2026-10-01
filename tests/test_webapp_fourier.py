from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from streamlit.testing.v1 import AppTest

# Absolute, so the path means the same thing on every supported Streamlit version: 1.47
# resolves a relative path against the working directory first, 1.64 against this file.
APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def test_fourier_page_renders_parts_a_to_f_with_synthetic_demo() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Fourier Theory").run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert any("Part A" in value for value in subheaders)
    assert any("Part F" in value for value in subheaders)


def test_fourier_page_accepts_uploaded_rgb_input() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, 16:] = (40, 120, 220)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok

    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Fourier Theory").run()
    app.file_uploader[0].set_value(("fourier.png", encoded.tobytes(), "image/png")).run()

    assert not app.exception
