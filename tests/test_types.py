from __future__ import annotations

from module4 import ImageMetadata, ReferenceStatus


def test_foundation_types_are_immutable_dataclasses() -> None:
    metadata = ImageMetadata(10, 20, 3, "uint8", "BGR", "image.png")
    status = ReferenceStatus("pending", "No reference has been run.")

    assert metadata.width == 20
    assert status.status == "pending"
