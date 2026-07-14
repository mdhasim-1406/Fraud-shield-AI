"""Quick Selenium smoke test for FraudShield"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import subprocess, sys, time

# Find chromedriver
find_result = subprocess.run(
    ["find", "/snap/chromium", "-name", "chromedriver"],
    capture_output=True, text=True
)
paths = [p for p in find_result.stdout.strip().split("\n") if p]
if not paths:
    print("NO CHROMEDRIVER FOUND")
    sys.exit(1)

chromedriver_path = paths[0]

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-software-rasterizer")
options.binary_location = "/snap/bin/chromium"
options.page_load_strategy = "eager"

service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.implicitly_wait(3)

try:
    driver.get("data:text/html,<h1>Hello FraudShield</h1><p>Test page</p>")
    print(f"Title: {driver.title}")
    h1 = driver.find_element("tag name", "h1")
    print(f"H1 text: {h1.text}")
    print("SELENIUM SMOKE TEST PASSED")
finally:
    driver.quit()
