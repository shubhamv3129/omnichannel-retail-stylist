"""
Unit tests for main.py application entry points.
Validates entry point wiring and delegation to feature_storefront.
"""

from unittest.mock import patch
import main


def test_example_retailnext_store_invokes_render():
    """Verify example_retailnext_store triggers render_retailnext_app."""
    with patch.object(main, "render_retailnext_app") as mock_render:
        main.example_retailnext_store()
        mock_render.assert_called_once()


def test_main_invokes_example_retailnext_store():
    """Verify main invokes example_retailnext_store."""
    with patch.object(main, "example_retailnext_store") as mock_example:
        main.main()
        mock_example.assert_called_once()
