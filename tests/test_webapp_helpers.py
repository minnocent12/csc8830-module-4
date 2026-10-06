"""Module 4 pages use the shared design kit's page header; the old local helpers are gone."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from module4.webapp import pages, ui  # noqa: E402
from module4.webapp.design.components import page_header as kit_page_header  # noqa: E402

PAGES_SOURCE = SRC / "module4" / "webapp" / "pages.py"


def test_obsolete_local_helpers_are_removed() -> None:
    for name in ("page_header", "status_message", "foundation_page"):
        assert not hasattr(ui, name), name


def test_pages_import_only_live_local_helpers_and_the_kit_header() -> None:
    tree = ast.parse(PAGES_SOURCE.read_text(encoding="utf-8"))
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imports["module4.webapp.ui"] == {"IMAGE_TYPES", "bundled_sample_notice", "pending_experiment_banner"}
    assert "page_header" in imports["module4.webapp.design.components"]


def test_every_module_4_page_calls_the_kit_header() -> None:
    assert pages.kit_page_header is kit_page_header
    tree = ast.parse(PAGES_SOURCE.read_text(encoding="utf-8"))
    callers = {
        function.name
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "kit_page_header"
    }
    page_renderers = {page.render.__name__ for page in pages.get_pages()}
    assert page_renderers <= callers
