"""Screenshot every detail view after a demo-scenario run (all nine views). Not a pytest module."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[2] / ".frontend_redesign" / "screenshots"
URL = "http://127.0.0.1:8501/?sid=sess-screens-0001"


def settle(page, ms=2000):
    page.wait_for_timeout(600)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=240000)
    except Exception:  # noqa: BLE001
        pass
    page.wait_for_timeout(ms)


def main(tag: str, scenario_index: int = 1) -> None:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1280, "height": 7000})
        page.goto(URL)
        settle(page, 3000)
        sb = page.locator('[data-testid="stSidebar"]')
        sb.get_by_text("Demo mode (5 core tabs)").click()
        settle(page)
        sb.get_by_role("combobox", name="Demonstration scenario").click()
        page.get_by_role("option").nth(scenario_index).click()
        sb.get_by_role("button", name="Load scenario").click()
        settle(page)
        page.get_by_role("button", name="Run optimisation pipeline").click()
        settle(page, 3000)
        details = page.locator(".st-key-qs_glass_details")
        tabs = details.get_by_role("tab")
        for i in range(tabs.count()):
            name = tabs.nth(i).inner_text().strip()
            tabs.nth(i).click()
            settle(page, 1500)
            details.screenshot(path=str(OUT / f"{tag}_view_{i + 1}_{name.split(' ')[0].lower()}.png"))
            print("captured", name)
        b.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
