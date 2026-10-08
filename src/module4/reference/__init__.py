"""SAM2 comparison method for Module 4.

The assignment requires comparing the classical results with SAM2. SAM2 is a deep-learning
model, so it is kept isolated from the classical RGB and thermal pipelines (which must not use
ML/DL). Importing the package does not import SAM2, PyTorch, checkpoints, or GPU-specific
code: live inference loads them lazily, and the recorded official SAM2 evidence is read from
JSON/PNG files.
"""

from module4.reference.recorded import (
    RecordedSAM2Case,
    find_recorded_sam2_case,
    load_sam2_records,
    load_timing_records,
)
from module4.reference.sam2_reference import (
    SAM2ReferenceError,
    bgr_to_sam2_rgb,
    check_sam2_availability,
    pending_sam2_reference,
    run_sam2_reference,
    select_sam2_mask,
    thermal_to_sam2_rgb,
    validate_box_prompt,
)

__all__ = [
    "RecordedSAM2Case",
    "find_recorded_sam2_case",
    "load_sam2_records",
    "load_timing_records",
    "SAM2ReferenceError",
    "bgr_to_sam2_rgb",
    "check_sam2_availability",
    "pending_sam2_reference",
    "run_sam2_reference",
    "select_sam2_mask",
    "thermal_to_sam2_rgb",
    "validate_box_prompt",
]
