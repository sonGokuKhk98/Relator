"""
API Sniffer - Discover hidden APIs by intercepting network traffic.

Opens a URL in headless Chrome, uses Chrome DevTools Protocol (CDP)
to capture all XHR/fetch requests, identifies JSON API responses,
and extracts structured data.

Inspired by the Repliers scraper approach:
- Browser-based API interception
- Rate limit handling
- Pagination detection
- Session/cookie management
"""

import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WDM = True
except ImportError:
    HAS_WDM = False

from .smart_scraper import DiscoveredAPI


class APISniffer:
    """
    Intercept network requests from a web page to discover hidden APIs.

    Usage:
        sniffer = APISniffer(headless=True)
        apis, data = sniffer.sniff("https://example.com/listings")
        sniffer.close()
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self._intercepted_requests: List[Dict] = []
        self._intercepted_responses: List[Dict] = []

    def sniff(
        self,
        url: str,
        wait_seconds: int = 8,
        scroll: bool = True,
    ) -> Tuple[List[DiscoveredAPI], List[Dict]]:
        """
        Open URL, intercept network traffic, discover APIs.

        Returns:
            (discovered_apis, dom_extracted_data)
        """
        if not HAS_SELENIUM:
            print("  [api_sniffer] Selenium not installed. Using lightweight mode.")
            return self._lightweight_sniff(url)

        self.driver = self._setup_driver()
        discovered_apis: List[DiscoveredAPI] = []
        dom_data: List[Dict] = []

        try:
            # Enable network interception via CDP
            self.driver.execute_cdp_cmd("Network.enable", {})

            # Set up request/response capture
            self._intercepted_requests = []
            self._intercepted_responses = []

            # Navigate to URL
            print(f"  [api_sniffer] Navigating to {url}")
            self.driver.get(url)
            time.sleep(3)

            # Scroll to trigger lazy loading
            if scroll:
                self._scroll_page()

            # Wait for XHR/fetch requests to complete
            print(f"  [api_sniffer] Waiting {wait_seconds}s for API calls...")
            time.sleep(wait_seconds)

            # Capture network traffic via performance logs
            discovered_apis = self._analyze_network_traffic()

            # Also try to extract data from the page DOM
            dom_data = self._extract_dom_data()

        except Exception as e:
            print(f"  [api_sniffer] Error: {e}")

        return discovered_apis, dom_data

    def paginate_api(
        self, api: DiscoveredAPI, max_pages: int = 5
    ) -> List[Dict]:
        """
        Follow pagination on a discovered API.
        """
        if not self.driver or not api.is_paginated:
            return []

        all_data: List[Dict] = []
        page_param = api.page_param or "page"

        for page in range(2, max_pages + 1):
            print(f"  [api_sniffer] Fetching page {page}...")

            # Build paginated URL
            url = api.url
            if "?" in url:
                url += f"&{page_param}={page}"
            else:
                url += f"?{page_param}={page}"

            # Execute fetch from within browser context
            headers_json = json.dumps(api.headers)
            script = f"""
            try {{
                const response = await fetch("{url}", {{
                    method: '{api.method}',
                    headers: {headers_json}
                }});
                if (response.status === 429) return {{ rate_limited: true }};
                if (!response.ok) return {{ error: true, status: response.status }};
                return await response.json();
            }} catch(e) {{
                return {{ error: true, message: e.message }};
            }}
            """

            try:
                result = self.driver.execute_script(script)

                if not result:
                    break

                if result.get("rate_limited"):
                    print(f"  [api_sniffer] Rate limited! Waiting 5s...")
                    time.sleep(5)
                    continue

                if result.get("error"):
                    print(f"  [api_sniffer] API error: {result}")
                    break

                # Extract data from response
                data = self._find_data_array(result)
                if not data:
                    print(f"  [api_sniffer] No more data on page {page}")
                    break

                all_data.extend(data)
                print(f"  [api_sniffer] Page {page}: got {len(data)} records (total: {len(all_data)})")

                time.sleep(1)  # Rate limiting

            except Exception as e:
                print(f"  [api_sniffer] Pagination error: {e}")
                break

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
    # Internal methods
    # ------------------------------------------------------------------

    def _setup_driver(self):
        """Setup Chrome with CDP and performance logging."""
        options = webdriver.ChromeOptions()

        if self.headless:
            options.add_argument("--headless=new")

        options.add_argument("--lang=en-US")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )

        # Enable performance logging to capture network traffic
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

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

    def _scroll_page(self):
        """Scroll page to trigger lazy loading."""
        print("  [api_sniffer] Scrolling page to trigger lazy loading...")
        try:
            for _ in range(5):
                self.driver.execute_script("window.scrollBy(0, 800)")
                time.sleep(0.5)
            self.driver.execute_script("window.scrollTo(0, 0)")
            time.sleep(1)
        except Exception:
            pass

    def _analyze_network_traffic(self) -> List[DiscoveredAPI]:
        """
        Parse Chrome performance logs to find API calls.
        This is the magic - we intercept ALL network requests.
        """
        apis: List[DiscoveredAPI] = []
        seen_urls = set()

        try:
            logs = self.driver.get_log("performance")
        except Exception:
            logs = []

        print(f"  [api_sniffer] Analyzing {len(logs)} network events...")

        for entry in logs:
            try:
                log_data = json.loads(entry["message"])
                message = log_data.get("message", {})
                method = message.get("method", "")
                params = message.get("params", {})

                # Look for Network.responseReceived events
                if method != "Network.responseReceived":
                    continue

                response = params.get("response", {})
                url = response.get("url", "")
                mime = response.get("mimeType", "")
                status = response.get("status", 0)
                request_id = params.get("requestId", "")

                # Filter: only JSON API responses
                if status != 200:
                    continue
                if not ("json" in mime or "javascript" in mime):
                    continue

                # Skip static assets, analytics, tracking
                skip_patterns = [
                    "google-analytics", "gtag", "facebook", "doubleclick",
                    "cloudflare", "sentry", "hotjar", "segment",
                    ".css", ".js", ".png", ".jpg", ".gif", ".svg",
                    "favicon", "manifest", "robots.txt",
                    "google.com/recaptcha", "gstatic.com",
                ]
                if any(p in url.lower() for p in skip_patterns):
                    continue

                # Deduplicate
                url_base = url.split("?")[0]
                if url_base in seen_urls:
                    continue
                seen_urls.add(url_base)

                # Try to get the response body
                response_body = None
                try:
                    body_result = self.driver.execute_cdp_cmd(
                        "Network.getResponseBody", {"requestId": request_id}
                    )
                    body_text = body_result.get("body", "")
                    if body_text:
                        response_body = json.loads(body_text)
                except Exception:
                    pass

                # Analyze response to determine if it's useful data
                record_count = 0
                data_path = ""
                is_paginated = False
                page_param = ""
                total_pages = 0

                if response_body:
                    if isinstance(response_body, list):
                        record_count = len(response_body)
                    elif isinstance(response_body, dict):
                        # Find the data array
                        for key, value in response_body.items():
                            if isinstance(value, list) and value and isinstance(value[0], dict):
                                record_count = len(value)
                                data_path = key
                                break

                        # Check for pagination indicators
                        pagination_keys = ["page", "pageNum", "currentPage", "offset", "nextPage"]
                        total_keys = ["total", "totalCount", "count", "totalPages", "totalResults"]
                        for pk in pagination_keys:
                            if pk in response_body or pk in str(response_body.get("meta", {})):
                                is_paginated = True
                                page_param = pk if pk in url else "page"
                                break
                        for tk in total_keys:
                            if tk in response_body:
                                total_count = response_body[tk]
                                if isinstance(total_count, (int, float)) and total_count > 0:
                                    is_paginated = True
                                    if record_count > 0:
                                        total_pages = int(total_count / record_count) + 1

                # Extract request headers
                req_headers = {}
                for key, value in response.get("headers", {}).items():
                    if key.lower() in ("authorization", "api-key", "x-api-key", "token"):
                        req_headers[key] = value

                api = DiscoveredAPI(
                    url=url,
                    method="GET",
                    headers=req_headers,
                    response_type="json",
                    data_path=data_path,
                    sample_response=response_body,
                    record_count=record_count,
                    is_paginated=is_paginated,
                    page_param=page_param,
                    total_pages=total_pages,
                )

                # Only add if it has meaningful data
                if record_count > 0 or (response_body and len(str(response_body)) > 200):
                    apis.append(api)
                    print(
                        f"  [api_sniffer] FOUND API: {url_base[:80]} "
                        f"({record_count} records, paginated={is_paginated})"
                    )

            except Exception:
                continue

        print(f"  [api_sniffer] Total discovered APIs: {len(apis)}")
        return apis

    def _extract_dom_data(self) -> List[Dict]:
        """
        Try to extract structured data from the page DOM as fallback.
        Looks for JSON-LD, meta tags, and table data.
        """
        data: List[Dict] = []

        if not self.driver:
            return data

        try:
            # Method 1: Look for JSON-LD structured data
            json_ld_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 'script[type="application/ld+json"]'
            )
            for elem in json_ld_elements:
                try:
                    ld_data = json.loads(elem.get_attribute("textContent"))
                    if isinstance(ld_data, list):
                        data.extend([d for d in ld_data if isinstance(d, dict)])
                    elif isinstance(ld_data, dict):
                        # Check for @graph array
                        if "@graph" in ld_data:
                            graph = ld_data["@graph"]
                            if isinstance(graph, list):
                                data.extend([d for d in graph if isinstance(d, dict)])
                        else:
                            data.append(ld_data)
                except Exception:
                    continue

            if data:
                print(f"  [api_sniffer] Found {len(data)} JSON-LD records in DOM")
                return data

            # Method 2: Look for __NEXT_DATA__ or similar hydration data
            hydration_scripts = [
                '__NEXT_DATA__',
                '__NUXT__',
                'window.__data',
                'window.__INITIAL_STATE__',
                'window.__PRELOADED_STATE__',
            ]
            for var_name in hydration_scripts:
                try:
                    result = self.driver.execute_script(
                        f"return window.{var_name.replace('window.', '')} || null;"
                    )
                    if result and isinstance(result, dict):
                        # Find data arrays in the hydration data
                        arrays = self._find_all_arrays(result, max_depth=5)
                        if arrays:
                            best = max(arrays, key=len)
                            if len(best) > 0 and isinstance(best[0], dict):
                                data = best
                                print(f"  [api_sniffer] Found {len(data)} records in {var_name}")
                                return data
                except Exception:
                    continue

            # Method 3: Look for HTML tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                try:
                    headers = [
                        th.text.strip()
                        for th in table.find_elements(By.TAG_NAME, "th")
                        if th.text.strip()
                    ]
                    if not headers:
                        continue

                    rows = table.find_elements(By.TAG_NAME, "tr")
                    for row in rows[1:]:  # Skip header row
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) == len(headers):
                            record = {}
                            for h, c in zip(headers, cells):
                                record[h] = c.text.strip()
                            data.append(record)

                    if data:
                        print(f"  [api_sniffer] Found {len(data)} records in HTML table")
                        return data
                except Exception:
                    continue

        except Exception as e:
            print(f"  [api_sniffer] DOM extraction error: {e}")

        return data

    def _find_all_arrays(self, obj: Any, max_depth: int = 5, _depth: int = 0) -> List[List]:
        """Recursively find all arrays of dicts in a nested object."""
        results: List[List] = []
        if _depth > max_depth:
            return results

        if isinstance(obj, list):
            dict_items = [item for item in obj if isinstance(item, dict)]
            if len(dict_items) > 2:
                results.append(dict_items)
        elif isinstance(obj, dict):
            for value in obj.values():
                results.extend(self._find_all_arrays(value, max_depth, _depth + 1))

        return results

    def _find_data_array(self, response: Any) -> List[Dict]:
        """Find the main data array in an API response."""
        if isinstance(response, list):
            return [r for r in response if isinstance(r, dict)]

        if isinstance(response, dict):
            # Look for common data keys
            data_keys = [
                "data", "results", "items", "records", "listings",
                "entries", "rows", "objects", "hits", "content",
            ]
            for key in data_keys:
                if key in response and isinstance(response[key], list):
                    return [r for r in response[key] if isinstance(r, dict)]

            # Find largest array
            best: List[Dict] = []
            for value in response.values():
                if isinstance(value, list) and len(value) > len(best):
                    dict_items = [v for v in value if isinstance(v, dict)]
                    if dict_items:
                        best = dict_items
            return best

        return []

    # ------------------------------------------------------------------
    # Lightweight mode (no Selenium)
    # ------------------------------------------------------------------

    def _lightweight_sniff(self, url: str) -> Tuple[List[DiscoveredAPI], List[Dict]]:
        """
        Lightweight API discovery without Selenium.
        Fetches the page HTML and looks for embedded API URLs and data.
        """
        import urllib.request

        apis: List[DiscoveredAPI] = []
        data: List[Dict] = []

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Find API URLs in the HTML source
            api_patterns = [
                r'(?:fetch|axios\.get|\.get|\.post)\s*\(\s*[\'"]([^"\']+)[\'"]',
                r'(?:api|endpoint|baseUrl|apiUrl)\s*[:=]\s*[\'"]([^"\']+)[\'"]',
                r'https?://[^"\'>\s]+/api/[^"\'>\s]+',
                r'https?://[^"\'>\s]+\.json[^"\'>\s]*',
            ]

            found_urls = set()
            for pattern in api_patterns:
                matches = re.findall(pattern, html)
                found_urls.update(matches)

            for api_url in found_urls:
                if api_url.startswith("http"):
                    apis.append(DiscoveredAPI(url=api_url))

            # Look for JSON-LD data
            json_ld_matches = re.findall(
                r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
                html,
                re.DOTALL,
            )
            for match in json_ld_matches:
                try:
                    ld = json.loads(match)
                    if isinstance(ld, list):
                        data.extend([d for d in ld if isinstance(d, dict)])
                    elif isinstance(ld, dict):
                        data.append(ld)
                except Exception:
                    continue

            # Look for embedded JSON data
            json_patterns = [
                r'window\.__NEXT_DATA__\s*=\s*(\{.*?\});?\s*</script>',
                r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});?\s*</script>',
            ]
            for pattern in json_patterns:
                match = re.search(pattern, html, re.DOTALL)
                if match:
                    try:
                        embedded = json.loads(match.group(1))
                        arrays = self._find_all_arrays(embedded)
                        if arrays:
                            best = max(arrays, key=len)
                            data.extend(best)
                    except Exception:
                        continue

        except Exception as e:
            print(f"  [api_sniffer] Lightweight sniff error: {e}")

        return apis, data
