"""Tests for the web UI module."""
from __future__ import annotations

from lmscan.web import check_streamlit


class TestWebUI:
    def test_check_streamlit_returns_bool(self) -> None:
        result = check_streamlit()
        assert isinstance(result, bool)

    def test_launch_without_streamlit(self) -> None:
        if check_streamlit():
            return  # skip if streamlit is actually installed
        import pytest

        with pytest.raises(RuntimeError, match="Streamlit"):
            from lmscan.web import launch
            launch()
