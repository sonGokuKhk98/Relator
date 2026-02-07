"""
Google Maps Coordinate Scraper for Dubai Areas
Selenium-based approach (inspired by USC campus explorer)

Searches Google Maps for each Dubai area/landmark and extracts
lat/lon from the resulting URL.

Parallelized with multiple browser instances for speed.

Usage:
    python scraper_layer/scrape_coordinates.py
"""

import time
import json
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

NUM_WORKERS = 4  # Number of parallel browser instances

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'dubai_areas_coordinates.json')

# Dubai areas & landmarks to scrape coordinates for
DUBAI_LOCATIONS = [
    # Major residential areas
    "Dubai Marina",
    "Palm Jumeirah",
    "Downtown Dubai",
    "Business Bay",
    "Jumeirah Beach Residence JBR",
    "Dubai Hills Estate",
    "Arabian Ranches",
    "The Springs Dubai",
    "The Meadows Dubai",
    "Emirates Hills",
    "Jumeirah Village Circle JVC",
    "Jumeirah Lake Towers JLT",
    "Dubai Sports City",
    "Motor City Dubai",
    "International City Dubai",
    "Discovery Gardens Dubai",
    "Al Barsha Dubai",
    "Deira Dubai",
    "Bur Dubai",
    "Karama Dubai",
    "Al Quoz Dubai",
    "Jumeirah Dubai",
    "Umm Suqeim Dubai",
    "Al Nahda Dubai",
    "Al Rashidiya Dubai",
    "Al Warqaa Dubai",
    "Mirdif Dubai",
    "Silicon Oasis Dubai",
    "Academic City Dubai",
    "Dubai Production City IMPZ",
    "Al Furjan Dubai",
    "Town Square Dubai",
    "Damac Hills Dubai",
    "Tilal Al Ghaf Dubai",
    "Dubai Creek Harbour",
    "Sobha Hartland Dubai",
    "MBR City District One Dubai",
    "City Walk Dubai",
    "La Mer Dubai",
    "Bluewaters Island Dubai",
    "The Greens Dubai",
    "The Views Dubai",
    "DIFC Dubai",
    "World Trade Centre Dubai",
    "Al Satwa Dubai",
    "Oud Metha Dubai",
    "Healthcare City Dubai",
    "Festival City Dubai",
    "Al Jaddaf Dubai",
    "Culture Village Dubai",
    # Major malls & landmarks
    "Dubai Mall",
    "Mall of the Emirates Dubai",
    "Ibn Battuta Mall Dubai",
    "Burj Khalifa",
    "Burj Al Arab",
    "Dubai Frame",
    "Museum of the Future Dubai",
    "Dubai Expo City",
    "Dubai Airport Terminal 3",
    "Al Maktoum International Airport",
    # Metro stations (supplementary)
    "Rashidiya Metro Station Dubai",
    "Airport Terminal 1 Metro Dubai",
    "GGICO Metro Station Dubai",
    "Deira City Centre Metro Dubai",
    "Union Metro Station Dubai",
    "BurJuman Metro Station Dubai",
    "ADCB Metro Station Dubai",
    "Sharaf DG Metro Station Dubai",
    "Mall of the Emirates Metro Dubai",
    "Nakheel Metro Station Dubai",
    "Ibn Battuta Metro Station Dubai",
    "UAE Exchange Metro Station Dubai",
    "Expo 2020 Metro Station Dubai",
    "Jebel Ali Metro Station Dubai",
]


def get_lat_lon_from_url(url):
    """Extract lat/lon from Google Maps URL using regex patterns."""
    # Try exact place coordinates: ...!3d34.0192383!4d-118.2869462...
    lat_match = re.search(r'!3d(-?\d+\.\d+)', url)
    lon_match = re.search(r'!4d(-?\d+\.\d+)', url)

    if lat_match and lon_match:
        return float(lat_match.group(1)), float(lon_match.group(1))

    # Fallback to viewport center: .../@25.0657,55.1713,17z/...
    match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', url)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def load_existing_results():
    """Load previously scraped results to avoid re-scraping."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def _create_driver():
    """Create a headless Chrome driver instance."""
    options = webdriver.ChromeOptions()
    options.add_argument('--lang=en')
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-images')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=options
    )
    driver.set_page_load_timeout(15)
    return driver


def _dismiss_consent(driver):
    """Handle Google consent/cookie dialog."""
    try:
        for btn in driver.find_elements(By.CSS_SELECTOR, "button"):
            if btn.text.lower().strip() in ("accept all", "accept", "agree", "i agree", "consent"):
                btn.click()
                time.sleep(1)
                return
    except Exception:
        pass
    try:
        for btn in driver.find_elements(By.CSS_SELECTOR, "form button, form input[type='submit']"):
            if "accept" in btn.text.lower() or "agree" in btn.text.lower():
                btn.click()
                time.sleep(1)
                return
    except Exception:
        pass


def _find_search_box(driver, timeout=10):
    """Try multiple selectors to find the Google Maps search box."""
    selectors = [
        (By.ID, "searchboxinput"),
        (By.CSS_SELECTOR, "input#searchboxinput"),
        (By.CSS_SELECTOR, "input[name='q']"),
        (By.CSS_SELECTOR, "input[aria-label='Search Google Maps']"),
    ]
    for by, selector in selectors:
        try:
            return WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
        except Exception:
            continue
    return None


def _wait_for_coordinates(driver, timeout=8):
    """Wait for URL to contain coordinates instead of using fixed sleep."""
    start = time.time()
    while time.time() - start < timeout:
        url = driver.current_url
        lat, lng = get_lat_lon_from_url(url)
        if lat and lng:
            return lat, lng, url
        # If on search results page, click first result
        if "/search/" in url:
            try:
                first_result = driver.find_element(By.CSS_SELECTOR, "a.hfpxzc")
                first_result.click()
            except Exception:
                pass
        time.sleep(0.5)
    return None, None, driver.current_url


def _scrape_worker(worker_id, locations):
    """Worker function: each runs its own browser and scrapes a chunk of locations."""
    results = []
    driver = None
    try:
        driver = _create_driver()
        driver.get("https://www.google.com/maps")
        time.sleep(2)
        _dismiss_consent(driver)

        search_box = _find_search_box(driver, timeout=15)
        if not search_box:
            print(f"  [W{worker_id}] Could not find search box, aborting worker")
            return [{"name": loc, "error": "Search box not found"} for loc in locations]

        for i, location in enumerate(locations):
            try:
                # Re-find search box (can go stale)
                search_box = _find_search_box(driver, timeout=5)
                if not search_box:
                    driver.get("https://www.google.com/maps")
                    time.sleep(2)
                    search_box = _find_search_box(driver, timeout=10)
                    if not search_box:
                        results.append({"name": location, "error": "Search box not found"})
                        continue

                search_box.click()
                search_box.clear()
                search_box.send_keys(Keys.COMMAND + "a")
                search_box.send_keys(Keys.DELETE)
                time.sleep(0.1)
                search_box.send_keys(location)
                search_box.send_keys(Keys.RETURN)

                lat, lng, url = _wait_for_coordinates(driver, timeout=8)

                if lat and lng:
                    print(f"  [W{worker_id}] {location} -> {lat}, {lng}")
                    results.append({"name": location, "lat": lat, "lng": lng, "url": url})
                else:
                    print(f"  [W{worker_id}] {location} -> NOT FOUND")
                    results.append({"name": location, "error": "Coordinates not found"})

            except Exception as e:
                print(f"  [W{worker_id}] {location} -> ERROR: {e}")
                results.append({"name": location, "error": str(e)})

    finally:
        if driver:
            driver.quit()

    return results


# Thread-safe lock for file writes
_file_lock = threading.Lock()


def scrape_coordinates():
    """Main scraping function - parallel workers with Selenium + Google Maps."""
    existing_results = load_existing_results()
    completed_map = {item['name']: item for item in existing_results if 'lat' in item}
    to_scrape = [loc for loc in DUBAI_LOCATIONS if loc not in completed_map]

    if not to_scrape:
        print("All locations have been successfully scraped!")
        print(f"Total: {len(completed_map)} locations with coordinates")
        return existing_results

    print(f"Found {len(completed_map)} existing successful results.")
    print(f"Scraping {len(to_scrape)} locations with {NUM_WORKERS} parallel browsers...")

    # Split locations evenly across workers
    chunks = [[] for _ in range(NUM_WORKERS)]
    for i, loc in enumerate(to_scrape):
        chunks[i % NUM_WORKERS].append(loc)

    all_results = list(completed_map.values())

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_scrape_worker, wid, chunk): wid
            for wid, chunk in enumerate(chunks) if chunk
        }
        for future in as_completed(futures):
            worker_results = future.result()
            all_results.extend(worker_results)
            # Save progress as each worker finishes
            with _file_lock:
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {OUTPUT_FILE}")

    successful = [r for r in all_results if 'lat' in r]
    failed = [r for r in all_results if 'error' in r]
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    return all_results


if __name__ == "__main__":
    print("=== Dubai Areas Google Maps Coordinate Scraper ===")
    print("Using Selenium to extract lat/lon from Google Maps URLs")
    print()
    scrape_coordinates()
