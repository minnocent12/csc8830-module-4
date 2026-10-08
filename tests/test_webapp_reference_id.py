"""Reference image ID ownership on Comparison and Evaluation.

The keyed "Reference image ID" field follows the current comparison image while it is
auto-managed, keeps a user's edit once the user owns it, and never relaxes validation: a real
ID mismatch is still refused. Upload steps need AppTest's file_uploader, which Streamlit 1.49
lacks, so those tests skip there; the runtime behavior does not depend on it.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = str(REPO_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from module4.io_utils import decode_image_bgr  # noqa: E402
from module4.metrics import evaluate_masks  # noqa: E402
from module4.rgb import run_rgb_segmentation  # noqa: E402
from module4.types import ROI, RGBPipelineConfig  # noqa: E402
from module4.validation import prepare_reference  # noqa: E402
from module4.webapp import pages  # noqa: E402

PAGE_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages
pages._comparison_page()
"""
ID_KEY = "comparison_reference_image_id"
OWNER_KEY = "_comparison_reference_image_id_owner"
BUNDLED_ID = pages._SAMPLE_RGB_IMAGE.name
UPLOAD_NAME = "my_frame.jpg"
UPLOAD_BYTES = pages._SAMPLE_RGB_IMAGE.read_bytes()


def _reference_png() -> tuple[str, bytes, str]:
    mask = cv2.imread(str(pages._SAMPLE_REFERENCE_RGB_MASK), cv2.IMREAD_UNCHANGED)
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    return ("my_frame_mask.png", encoded.tobytes(), "image/png")


def _page() -> AppTest:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    assert not app.exception
    return app


def _uploader(app: AppTest, key: str):
    if not hasattr(app, "file_uploader"):
        pytest.skip("this Streamlit version's AppTest cannot simulate file uploads")
    return app.file_uploader(key=key)


def _upload_image(app: AppTest, name: str = UPLOAD_NAME) -> None:
    _uploader(app, "comparison_input").set_value((name, UPLOAD_BYTES, "image/jpeg")).run()
    assert not app.exception


def _reference_id(app: AppTest) -> str:
    return app.text_input(key=ID_KEY).value


def _run(app: AppTest) -> None:
    app.button[0].click().run()
    assert not app.exception


def _expected_uploaded_metrics():
    """Recompute what the page shows for the uploaded frame with its default ROI, on a fresh
    thread (OpenCV's GrabCut RNG is thread-local, as in each Streamlit script run)."""
    out = {}

    def compute() -> None:
        source = decode_image_bgr(UPLOAD_BYTES, source_name=UPLOAD_NAME)
        height, width = source.shape[:2]
        roi = ROI(width // 4, height // 8, min(max(2, width // 2), width), min(max(2, (height * 3) // 4), height))
        result = run_rgb_segmentation(source, roi, config=RGBPipelineConfig(
            grabcut_iterations=5, opening_kernel_size=3, closing_kernel_size=5))
        mask = cv2.imdecode(
            __import__("numpy").frombuffer(_reference_png()[1], dtype="uint8"), cv2.IMREAD_UNCHANGED
        )
        prepared = prepare_reference(
            mask, expected_shape=result.final_mask.shape, reference_status="available",
            reference_type="ground_truth", image_id=UPLOAD_NAME, reference_image_id=UPLOAD_NAME,
        )
        out["metrics"] = evaluate_masks(result.final_mask, prepared.mask)

    worker = threading.Thread(target=compute)
    worker.start()
    worker.join()
    return out["metrics"]


def test_bundled_image_initializes_the_reference_id() -> None:
    app = _page()
    assert _reference_id(app) == BUNDLED_ID
    assert app.session_state[OWNER_KEY] == "auto"


def test_uploading_a_new_image_updates_an_auto_managed_id() -> None:
    app = _page()
    _upload_image(app)
    assert _reference_id(app) == UPLOAD_NAME
    assert app.session_state[OWNER_KEY] == "auto"


def test_previously_broken_workflow_evaluates_without_manual_id_correction() -> None:
    # bundled frame -> upload a different image -> upload its matching reference -> run
    app = _page()
    assert _reference_id(app) == BUNDLED_ID
    _upload_image(app)
    _uploader(app, "comparison_reference").set_value(_reference_png()).run()
    _run(app)
    assert not app.error
    expected = _expected_uploaded_metrics()
    assert [(m.label, m.value) for m in app.metric] == [
        ("IoU", f"{expected.iou:.4f}"),
        ("Dice", f"{expected.dice:.4f}"),
        ("Precision", f"{expected.precision:.4f}"),
        ("Recall", f"{expected.recall:.4f}"),
    ]
    table = app.dataframe[-1]  # the recorded SAM2 results and timing tables come first
    assert dict(zip(table.value["Count"], table.value["Pixels"])) == {
        "TP": expected.tp, "FP": expected.fp, "FN": expected.fn, "TN": expected.tn,
    }


def test_manual_edit_becomes_user_owned_and_survives_ordinary_reruns() -> None:
    app = _page()
    app.text_input(key=ID_KEY).set_value("custom_id.jpg").run()
    assert app.session_state[OWNER_KEY] == "user"
    assert _reference_id(app) == "custom_id.jpg"
    # ordinary reruns: a parameter change, then a plain rerun
    app.slider(key="comparison_rgb_iterations").set_value(4).run()
    assert _reference_id(app) == "custom_id.jpg"
    app.run()
    assert _reference_id(app) == "custom_id.jpg"
    assert app.session_state[OWNER_KEY] == "user"


def test_genuine_reference_id_mismatch_is_still_rejected() -> None:
    app = _page()
    _upload_image(app)
    _uploader(app, "comparison_reference").set_value(_reference_png()).run()
    app.text_input(key=ID_KEY).set_value("another_image.jpg").run()
    _run(app)
    assert not app.metric
    assert any("identities do not match" in item.value for item in app.error)


def test_user_owned_id_is_kept_when_the_image_is_replaced() -> None:
    app = _page()
    app.text_input(key=ID_KEY).set_value("custom_id.jpg").run()
    _upload_image(app)
    assert _reference_id(app) == "custom_id.jpg"
    assert app.session_state[OWNER_KEY] == "user"


def test_restoring_the_current_id_returns_the_field_to_auto_management() -> None:
    app = _page()
    _upload_image(app)
    app.text_input(key=ID_KEY).set_value("custom_id.jpg").run()
    app.text_input(key=ID_KEY).set_value(UPLOAD_NAME).run()
    assert app.session_state[OWNER_KEY] == "auto"
    _upload_image(app, "second_frame.jpg")
    assert _reference_id(app) == "second_frame.jpg"


def test_sam2_reference_has_no_reference_id_field() -> None:
    # The bundled frame is a recorded dataset frame, so without a live SAM2 runtime its
    # recorded official SAM2 output is the reference; no reference image ID is asked for.
    app = _page()
    app.radio(key="comparison_reference_source").set_value("SAM2 reference").run()
    assert not app.exception
    assert not any(w.key == ID_KEY for w in app.text_input)
    html = " ".join(e.proto.body for e in app.get("html"))
    assert "Source: recorded official SAM2 run on this frame" in html


def test_reference_widgets_keep_labels_types_and_defaults() -> None:
    app = _page()
    field = app.text_input(key=ID_KEY)
    assert field.label == "Reference image ID"
    # The initial value comes from session state (so the app can update it), not ``value=``.
    assert field.value == BUNDLED_ID
    assert not any("Session State API" in item.value for item in app.warning)
    assert field.proto.help == "Must match the input image ID exactly; this prevents cross-image comparisons."
    assert app.radio(key="comparison_reference_source").value == "Uploaded reference mask"
    assert app.selectbox(key="comparison_reference_type").value == "ground_truth"
    assert app.checkbox(key="comparison_reference_alignment").value is False
    # No internal ownership metadata is shown anywhere on the page.
    visible = [str(e.value) for kind in ("markdown", "caption", "info", "warning", "error") for e in getattr(app, kind)]
    assert not any("_comparison_reference_image_id" in text or "auto-managed" in text for text in visible)
