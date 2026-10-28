"""
Main entry point for the RetailNext Smart Store application.
"""
from feature_storefront import render_retailnext_app


def example_retailnext_store() -> None:
    """
    Example function launching the RetailNext Smart Store Streamlit UI.
    """
    render_retailnext_app()


def main() -> None:
    """
    Main entry point.
    """
    example_retailnext_store()


if __name__ == "__main__":
    main()

