from __future__ import annotations

import pytest

from module4.webapp._page import PageSpec
from module4.webapp.pages import get_pages
from module4.webapp.registry import collect_pages


def _fake_provider() -> list[PageSpec]:
    return [
        PageSpec("Module X", "Overview", 10, lambda: None),
        PageSpec("Module X", "Details", 20, lambda: None),
    ]


def test_module4_provider_returns_stable_pages() -> None:
    pages = collect_pages([get_pages])
    assert [page.page_label for page in pages] == [
        "RGB Human Boundary",
        "Thermal Human Boundary",
        "Comparison and Evaluation",
        "Fourier Theory",
    ]
    assert {page.module_label for page in pages} == {"Module 4"}


def test_registry_merges_and_orders_multiple_providers() -> None:
    pages = collect_pages([get_pages, _fake_provider])
    assert [(page.module_label, page.page_label) for page in pages] == [
        ("Module 4", "RGB Human Boundary"),
        ("Module 4", "Thermal Human Boundary"),
        ("Module 4", "Comparison and Evaluation"),
        ("Module 4", "Fourier Theory"),
        ("Module X", "Overview"),
        ("Module X", "Details"),
    ]


def test_later_provider_overrides_duplicate_page() -> None:
    marker = object()

    def override() -> list[PageSpec]:
        return [PageSpec("Module 4", "Fourier Theory", 40, lambda: marker)]

    theory = next(
        page for page in collect_pages([get_pages, override]) if page.page_label == "Fourier Theory"
    )
    assert theory.render() is marker


def test_registry_rejects_non_pagespec() -> None:
    with pytest.raises(TypeError):
        collect_pages([lambda: ["not a page"]])
