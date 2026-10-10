"""Screenshots of upload-preview and rejected states with synthetic fixtures (no writes). Not a pytest module."""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".frontend_redesign" / "screenshots"
URL = "http://127.0.0.1:8501/?sid=sess-screens-0001"


def settle(page, ms=2000):
    page.wait_for_timeout(600)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=240000)
    except Exception:  # noqa: BLE001
        pass
    page.wait_for_timeout(ms)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1280, "height": 2600})
    page.goto(URL)
    settle(page, 3000)
    page.locator('[data-testid="stSidebar"] label').filter(has_text="Upload Dataset (CSV, JSON)").first.click()
    settle(page)
    page.locator('input[type="file"]').set_input_files(str(ROOT / "tests/frontend/fixtures/synthetic_sales.csv"))
    settle(page)
    page.get_by_role("button", name="Preview (no writes)").click()
    settle(page, 2500)
    page.screenshot(path=str(OUT / "state_upload_preview_1280.png"))
    page.locator('[data-testid="stSidebar"] label').filter(has_text="Demo Database").first.click()
    settle(page)
    page.get_by_label("Slow SQL query (single read-only SELECT)").fill("SELECT * FROM table_that_does_not_exist WHERE id = 1")
    page.keyboard.press("Control+Enter")
    settle(page)
    page.get_by_role("button", name="Run optimisation pipeline").click()
    settle(page, 3000)
    page.screenshot(path=str(OUT / "state_rejected_1280.png"))
    b.close()
