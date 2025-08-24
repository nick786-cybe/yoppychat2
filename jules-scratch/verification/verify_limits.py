
from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()

    # Log in as a test user
    page.goto("http://localhost:5000/dev/login")

    # Navigate to the channel page
    page.goto("http://localhost:5000/channel")
    page.screenshot(path="jules-scratch/verification/channel_page.png")

    # Navigate to the ask page
    page.goto("http://localhost:5000/ask")
    # Open the profile menu
    page.click("#user-profile-trigger")
    page.screenshot(path="jules-scratch/verification/ask_page_profile_menu.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
