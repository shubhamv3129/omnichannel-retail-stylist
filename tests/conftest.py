"""
Pytest configuration and environment fixtures for omnichannel-retail-stylist.
Ensures external dependencies are mocked gracefully if not installed in the test environment.
"""

import math
import sys
from unittest.mock import MagicMock
import pytest

# Ensure missing third-party modules can be imported during tests
MOCK_MODULES = [
    "dotenv",
    "openai",
    "pandas",
    "numpy",
    "pydantic",
    "streamlit",
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            mock_obj = MagicMock()
            sys.modules[mod] = mock_obj
            if mod == "streamlit":
                mock_obj.secrets = {}
                mock_obj.session_state = {}
                mock_obj.cache_data = lambda *args, **kwargs: (lambda fn: fn)
            elif mod == "numpy":
                mock_obj.dot = lambda a, b: sum(x * y for x, y in zip(a, b))
                mock_obj.linalg = MagicMock()
                mock_obj.linalg.norm = lambda a: math.sqrt(sum(x * x for x in a))


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit UI calls so tests run cleanly in headless mode."""
    import streamlit as st

    st.spinner = MagicMock(
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
    )
    st.write = MagicMock()
    st.markdown = MagicMock()
    st.error = MagicMock()
    st.info = MagicMock()
    st.success = MagicMock()
    st.balloons = MagicMock()
    st.title = MagicMock()
    st.set_page_config = MagicMock()
    st.columns = MagicMock(
        return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    )
    st.container = MagicMock(
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
    )
    st.chat_message = MagicMock(
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
    )
    st.chat_input = MagicMock(return_value=None)
    st.file_uploader = MagicMock(return_value=None)
    st.audio_input = MagicMock(return_value=None)
    st.audio = MagicMock()
    st.image = MagicMock()
    st.button = MagicMock(return_value=False)
    st.text_input = MagicMock(return_value="")
    st.radio = MagicMock(return_value="option1")
    st.divider = MagicMock()
    st.rerun = MagicMock()
    st.stop = MagicMock(side_effect=SystemExit(0))
    st.session_state = {}
    st.secrets = {}
    yield st
