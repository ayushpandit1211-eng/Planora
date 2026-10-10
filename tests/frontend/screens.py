"""Capture dashboard screenshots with Playwright (V3 redesign QA). Not a pytest module.

    python tests/frontend/screens.py <tag> [scenario ...]
Scenarios: initial, report (loads demo scenario 1 and runs the pipeline), upload, dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[2] / ".frontend_redesign" / "screenshots"
URL = "http://127.0.0.1:8501/?sid=sess-screens-0001"
SIZES = [(1280, 820), (1440, 900), (1024, 768), (768, 1024), (390, 844)]


def settle(page, ms=2500):
    page.wait_for_timeout(600)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=180000)
    except Exception:  # noqa: BLE001
        pass
    page.wait_for_timeout(ms)


def main(tag: str, scenarios: list[str], sizes=SIZES) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for sc in scenarios:
            for w, h in sizes:
                page = b.new_page(viewport={"width": w, "height": h})
                page.goto(URL)
                page.wait_for_selector('[data-testid="stApp"]')
                settle(page)
                if sc == "report":
                    sb = page.locator('[data-testid="stSidebar"]')
                    if w < 900:
                        page.locator('[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button').first.click()
                        page.wait_for_timeout(800)
                    sb.get_by_role("combobox", name="Demonstration scenario").click()
                    page.get_by_role("option").nth(1).click()
                    sb.get_by_role("button", name="Load scenario").click()
                    settle(page, 1500)
                    if w < 900:
                        page.keyboard.press("Escape")
                        sb.locator('[data-testid="stSidebarCollapseButton"] button').first.click(timeout=3000) if sb.locator('[data-testid="stSidebarCollapseButton"] button').count() else None
                        page.wait_for_timeout(600)
                    page.get_by_role("button", name="Run optimisation pipeline").click()
                    settle(page, 3000)
                elif sc == "upload":
                    if w < 900:
                        page.locator('[data-testid="stExpandSidebarButton"]').first.click()
                        page.wait_for_timeout(800)
                    page.locator('[data-testid="stSidebar"] label').filter(has_text="Upload Dataset (CSV, JSON)").first.click()
                    if w < 900:
                        page.wait_for_timeout(1500)
                        page.locator('[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"]').first.click()
                    settle(page)
                elif sc == "askdb":
                    if w < 900:
                        page.locator('[data-testid="stExpandSidebarButton"]').first.click()
                        page.wait_for_timeout(800)
                    page.locator('[data-testid="stSidebar"]').get_by_role("combobox", name="Active source").click()
                    page.get_by_role("option").filter(has_text="enterprise_survey2").first.click()
                    settle(page)
                    if w < 900:
                        page.locator('[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"]').first.click()
                        page.wait_for_timeout(800)
                    page.get_by_label("Ask your database").fill("total value by industry name anzsic in 2025")
                    page.keyboard.press("Enter")
                    settle(page)
                    page.get_by_role("button", name="Ask", exact=True).click()
                    settle(page, 3000)
                page.screenshot(path=str(OUT / f"{tag}_{sc}_{w}x{h}.png"), full_page=False)
                total = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
                                      " return m ? m.scrollHeight : document.body.scrollHeight }")
                page.set_viewport_size({"width": w, "height": min(max(total, h), 9000)})
                page.wait_for_timeout(1200)
                page.screenshot(path=str(OUT / f"{tag}_{sc}_{w}x{h}_full.png"), full_page=True)
                page.close()
        b.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["initial"])
