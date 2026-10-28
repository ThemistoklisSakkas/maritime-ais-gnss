"""Headless check that every dashboard page renders without an exception."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DASH = Path(__file__).resolve().parents[1] / "outputs" / "dashboard"
PAGES = ["Overview", "Dark activity", "Ship-to-ship encounters", "GNSS interference",
         "ETA prediction", "INS/GNSS resilience"]


@pytest.mark.skipif(not (DASH / "dark_gaps.parquet").exists(),
                    reason="dashboard bundle not exported (python -m maritime dashboard-data)")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
