"""
DOM Scraper - AI-powered intelligent page parsing.

Uses LLM to analyze page structure and generate CSS selectors,
then extracts structured data from repeating elements.

Inspired by the Zillow scraper approach:
- Anti-detection measures
- Smart selector discovery
- Pagination handling
- Data parsing (prices, dates, numbers)
"""

import json
import time
import re
import random
from typing import List, Dict, Any, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WDM = True
except ImportError:
    HAS_WDM = False


class DOMScraper:
    """
    Intelligent DOM scraper that uses AI to figure out CSS selectors.

    Usage:
        scraper = DOMScraper(llm_model=gemini_model)
        data = scraper.scrape("https://example.com/products")
        scraper.close()
    """

    def __init__(self, llm_model=None, headless: bool = True):
        self.llm = llm_model
        self.headless = headless
        self.driver = None

    def scrape(
        self,
        url: str,
        description: str = "",
        max_pages: int = 3,
        selectors: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Scrape structured data from a URL.

        If selectors are provided, use them directly.
        Otherwise, use AI to discover the right selectors.
        """
        if not HAS_SELENIUM:
            print("  [dom_scraper] Selenium not installed. Using HTML-only mode.")
            return self._html_only_scrape(url, description)

        self.driver = self._setup_driver()
        all_data: List[Dict] = []

        try:
            # Navigate to page
            print(f"  [dom_scraper] Navigating to {url}")
            self.driver.get(url)
            time.sleep(random.uniform(3, 5))

            # Check for blocking
            self._check_for_blocks()

            # Scroll to load content
            self._scroll_page()

            # Get page HTML for AI analysis
            page_html = self.driver.page_source
            page_text = self._get_visible_text()

            # Discover selectors with AI (or use provided)
            if not selectors:
                selectors = self._ai_discover_selectors(url, page_html, page_text, description)

            if not selectors:
                print("  [dom_scraper] Could not discover selectors. Trying text extraction.")
                return self._extract_from_text(page_text, description)

            # Extract data using selectors
            page_data = self._extract_with_selectors(selectors)
            all_data.extend(page_data)
            print(f"  [dom_scraper] Page 1: extracted {len(page_data)} records")

            # Handle pagination
            for page_num in range(2, max_pages + 1):
                next_url = self._find_next_page(selectors)
                if not next_url:
                    break

                print(f"  [dom_scraper] Navigating to page {page_num}...")
                self.driver.get(next_url)
                time.sleep(random.uniform(2, 4))
                self._scroll_page()

                page_data = self._extract_with_selectors(selectors)
                if not page_data:
                    break

                all_data.extend(page_data)
                print(f"  [dom_scraper] Page {page_num}: extracted {len(page_data)} records")

        except Exception as e:
            print(f"  [dom_scraper] Error: {e}")

        return all_data

    def close(self):
        """Close the browser."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_driver(self):
        """Setup Chrome with anti-detection (like Zillow scraper)."""
        options = webdriver.ChromeOptions()

        if self.headless:
            options.add_argument("--headless=new")

        # Anti-detection settings
        options.add_argument("--lang=en-US")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        if HAS_WDM:
            driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=options,
            )
        else:
            driver = webdriver.Chrome(options=options)

        # Remove webdriver flag
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
        )

        return driver

    def _check_for_blocks(self):
        """Check for CAPTCHA or blocking pages."""
        if not self.driver:
            return

        source = self.driver.page_source.lower()
        blocked_keywords = ["captcha", "blocked", "access denied", "rate limit", "forbidden"]

        for keyword in blocked_keywords:
            if keyword in source:
                print(f"  [dom_scraper] WARNING: Possible block detected ({keyword})")
                time.sleep(5)  # Wait and hope it clears
                break

    def _scroll_page(self):
        """Scroll to trigger lazy loading."""
        if not self.driver:
            return

        try:
            for _ in range(6):
                self.driver.execute_script("window.scrollBy(0, 600)")
                time.sleep(random.uniform(0.5, 1.0))
            self.driver.execute_script("window.scrollTo(0, 0)")
            time.sleep(1)
        except Exception:
            pass

    def _get_visible_text(self) -> str:
        """Get all visible text on the page."""
        if not self.driver:
            return ""
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            return ""

    def _ai_discover_selectors(
        self,
        url: str,
        page_html: str,
        page_text: str,
        description: str,
    ) -> Optional[Dict[str, str]]:
        """
        Use AI to analyze the page and generate CSS selectors for data extraction.
        This is the MAGIC - AI figures out the page structure automatically.
        """
        if not self.llm:
            return self._heuristic_selectors()

        # Truncate HTML (keep the interesting parts)
        html_snippet = page_html[:10000]

        prompt = f"""You are a web scraping expert. Analyze this webpage and generate CSS selectors to extract structured data.

URL: {url}
User wants: {description or 'All repeating structured data (listings, products, items, etc.)'}

PAGE HTML (truncated):
{html_snippet}

VISIBLE TEXT (first 2000 chars):
{page_text[:2000]}

Your task:
1. Identify the REPEATING elements (product cards, list items, table rows, etc.)
2. For each data field, provide a CSS selector RELATIVE to the card/item container

Respond with ONLY valid JSON:
{{
    "card_selector": "CSS selector for the repeating container element",
    "fields": {{
        "field_name": "CSS selector relative to card (or 'TEXT' for card text)",
        "field_name_2": "CSS selector"
    }},
    "next_page_selector": "CSS selector for next page button/link (or null)",
    "data_type": "what kind of data is this (products, listings, articles, etc.)"
}}

IMPORTANT:
- card_selector should match ALL repeating items
- field selectors are RELATIVE to each card
- Use 'TEXT' if the data is in the card's text content and needs regex extraction
- Be specific with selectors (data attributes > class names > tag names)"""

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()

            # Extract JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                selectors = json.loads(json_match.group())
                print(f"  [dom_scraper] AI generated selectors: {json.dumps(selectors, indent=2)[:200]}")
                return selectors
        except Exception as e:
            print(f"  [dom_scraper] AI selector generation failed: {e}")

        return self._heuristic_selectors()

    def _heuristic_selectors(self) -> Optional[Dict[str, str]]:
        """
        Fall back to heuristic selector detection.
        Try common patterns for listing pages.
        """
        if not self.driver:
            return None

        common_card_selectors = [
            "article[data-test*='card']",
            "[data-testid*='card']",
            "[class*='card']",
            "[class*='listing']",
            "[class*='item']",
            "article",
            ".product",
            "li[class]",
            "tr[class]",
        ]

        for selector in common_card_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if len(elements) >= 3:  # Need at least 3 repeating items
                    print(f"  [dom_scraper] Heuristic found cards with: {selector} ({len(elements)} items)")
                    return {
                        "card_selector": selector,
                        "fields": {"TEXT": "TEXT"},
                        "next_page_selector": None,
                    }
            except Exception:
                continue

        return None

    def _extract_with_selectors(self, selectors: Dict) -> List[Dict]:
        """Extract data from page using AI-generated selectors."""
        if not self.driver or not selectors:
            return []

        data: List[Dict] = []
        card_selector = selectors.get("card_selector", "")
        fields = selectors.get("fields", {})

        if not card_selector:
            return []

        try:
            cards = self.driver.find_elements(By.CSS_SELECTOR, card_selector)
            print(f"  [dom_scraper] Found {len(cards)} cards with '{card_selector}'")

            for idx, card in enumerate(cards):
                record = {}

                if "TEXT" in fields and fields["TEXT"] == "TEXT":
                    # Use card text content and parse with regex
                    record = self._parse_card_text(card.text)
                else:
                    for field_name, field_selector in fields.items():
                        if field_selector == "TEXT":
                            record[field_name] = card.text.strip()
                            continue

                        try:
                            elem = card.find_element(By.CSS_SELECTOR, field_selector)
                            value = elem.text.strip()

                            # Try to get href for links
                            if not value:
                                value = elem.get_attribute("href") or elem.get_attribute("src") or ""

                            record[field_name] = value
                        except NoSuchElementException:
                            record[field_name] = None
                        except Exception:
                            record[field_name] = None

                # Only add if we got some data
                if record and any(v for v in record.values() if v):
                    # Clean up values
                    record = self._clean_record(record)
                    data.append(record)

        except Exception as e:
            print(f"  [dom_scraper] Extraction error: {e}")

        return data

    def _parse_card_text(self, text: str) -> Dict:
        """Parse unstructured card text into fields using regex patterns."""
        record = {"raw_text": text}

        if not text:
            return record

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Price patterns
        price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', text)
        if price_match:
            price_text = price_match.group()
            record["price"] = price_text
            try:
                record["price_numeric"] = float(
                    price_text.replace("$", "").replace(",", "")
                )
            except ValueError:
                pass

        # Beds/baths/sqft patterns
        beds_match = re.search(r'(\d+)\s*(?:bd|bds|bed|beds|br|bedroom)', text, re.I)
        if beds_match:
            record["bedrooms"] = int(beds_match.group(1))

        baths_match = re.search(r'(\d+\.?\d*)\s*(?:ba|baths?|bathroom)', text, re.I)
        if baths_match:
            record["bathrooms"] = float(baths_match.group(1))

        sqft_match = re.search(r'([\d,]+)\s*(?:sqft|sq\s*ft|square\s*feet)', text, re.I)
        if sqft_match:
            record["sqft"] = int(sqft_match.group(1).replace(",", ""))

        # Address pattern (line with comma + state abbreviation)
        for line in lines:
            if re.search(r',\s*[A-Z]{2}\s*\d{5}', line):
                record["address"] = line
                break
            elif re.search(r',\s*[A-Z]{2}\b', line) and len(line) > 10:
                record["address"] = line
                break

        # Rating patterns
        rating_match = re.search(r'(\d+\.?\d*)\s*(?:stars?|rating|/5|out of)', text, re.I)
        if rating_match:
            record["rating"] = float(rating_match.group(1))

        # Date patterns
        date_match = re.search(
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})', text
        )
        if date_match:
            record["date"] = date_match.group(1)

        # Use first non-price, non-address line as title
        for line in lines:
            if line and not line.startswith("$") and line != record.get("address"):
                record["title"] = line
                break

        return record

    def _clean_record(self, record: Dict) -> Dict:
        """Clean up extracted record values."""
        cleaned = {}
        for key, value in record.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
                # Try to parse numbers
                if re.match(r'^\$[\d,]+\.?\d*$', value):
                    try:
                        cleaned[key + "_numeric"] = float(
                            value.replace("$", "").replace(",", "")
                        )
                    except ValueError:
                        pass
                cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned

    def _find_next_page(self, selectors: Dict) -> Optional[str]:
        """Find and return the URL for the next page."""
        if not self.driver:
            return None

        next_selector = selectors.get("next_page_selector")

        if next_selector:
            try:
                next_btn = self.driver.find_element(By.CSS_SELECTOR, next_selector)
                href = next_btn.get_attribute("href")
                if href:
                    return href
                # Try clicking the button
                next_btn.click()
                time.sleep(3)
                return self.driver.current_url
            except Exception:
                pass

        # Try common next page selectors
        common_next_selectors = [
            "a[rel='next']",
            "a[aria-label='Next page']",
            "a[title='Next page']",
            "[class*='next'] a",
            "[class*='pagination'] a:last-child",
            "nav a:last-child",
        ]

        for selector in common_next_selectors:
            try:
                elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                href = elem.get_attribute("href")
                if href:
                    return href
            except Exception:
                continue

        return None

    def _extract_from_text(self, text: str, description: str) -> List[Dict]:
        """Last resort: extract data from raw page text using AI."""
        if not self.llm or not text:
            return []

        prompt = f"""Extract structured data from this webpage text.

User wants: {description or 'All structured/repeating data'}

PAGE TEXT (first 5000 chars):
{text[:5000]}

Return ONLY a JSON array of objects. Each object is one data record.
[{{"field1": "value1", ...}}, ...]"""

        try:
            response = self.llm.generate_content(prompt)
            text_resp = response.text.strip()
            json_match = re.search(r'\[.*\]', text_resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    return data
        except Exception:
            pass

        return []

    # ------------------------------------------------------------------
    # HTML-only mode (no Selenium)
    # ------------------------------------------------------------------

    def _html_only_scrape(self, url: str, description: str) -> List[Dict]:
        """Scrape without Selenium - fetch HTML and parse with regex + AI."""
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Try to parse with BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")

                # Remove script/style tags
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                # Strip HTML tags with regex
                text = re.sub(r'<[^>]+>', '\n', html)
                text = re.sub(r'\s+', ' ', text).strip()

            return self._extract_from_text(text, description)

        except Exception as e:
            print(f"  [dom_scraper] HTML-only scrape failed: {e}")
            return []
