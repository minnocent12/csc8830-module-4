"""Small, dependency-light types shared by Module 4 foundation code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColorOrder = Literal["BGR", "RGB", "GRAY", "THERMAL"]
ReferenceStatusValue = Literal["pending", "completed", "failed", "unavailable"]


@dataclass(frozen=True)
class ImageMetadata:
    """Describes an image array without changing or copying its pixels."""

    height: int
    width: int
    channels: int
    dtype: str
    color_order: ColorOrder
    source_name: str | None = None


@dataclass(frozen=True)
class ReferenceStatus:
    """Reports whether an optional comparison reference is available."""

    status: ReferenceStatusValue
    message: str
    source: str | None = None
