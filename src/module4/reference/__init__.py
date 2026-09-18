"""Optional reference-model integrations for Module 4.

The package is deliberately isolated from the classical RGB and thermal pipelines. Importing
the package does not import SAM2, PyTorch, checkpoints, or GPU-specific code.
"""

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
    "SAM2ReferenceError",
    "bgr_to_sam2_rgb",
    "check_sam2_availability",
    "pending_sam2_reference",
    "run_sam2_reference",
    "select_sam2_mask",
    "thermal_to_sam2_rgb",
    "validate_box_prompt",
]
