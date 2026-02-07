"""
SmartScraper - AI-Powered Universal Data Extractor

Give it ANY URL and it will:
1. Analyze the page structure
2. Discover hidden APIs via network interception
3. Fall back to intelligent DOM scraping
4. Use AI to structure the raw data into clean schemas
5. Load directly into your database

The WOW: User says "scrape this website" → AI does everything automatically.
"""

import json
import time
import re
import os
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote


class ScrapeStrategy(str, Enum):
    API_SNIFF = "api_sniff"         # Intercept hidden APIs via CDP
    DOM_PARSE = "dom_parse"         # Parse page DOM with AI-generated selectors
    DIRECT_API = "direct_api"       # Direct API calls (user provided endpoint)
    HYBRID = "hybrid"               # API sniff + DOM parse combined
    AI_ANALYZED = "ai_analyzed"     # Fully AI-driven (LLM reads page and extracts)


@dataclass
class DiscoveredAPI:
    """An API endpoint discovered by sniffing network traffic."""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    response_type: str = "json"
    data_path: str = ""              # JSONPath to the data array in response
    sample_response: Optional[Dict] = None
    record_count: int = 0
    is_paginated: bool = False
    page_param: str = ""
    total_pages: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ScrapeResult:
    """Result of a scraping operation."""
    success: bool
    strategy_used: ScrapeStrategy
    url: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)
    discovered_apis: List[DiscoveredAPI] = field(default_factory=list)
    table_name: str = ""
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    ai_analysis: str = ""
    warnings: List[str] = field(default_factory=list)
    execution_log: List[Dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    scraped_at: str = ""

    def to_dict(self):
        d = asdict(self)
        d["strategy_used"] = self.strategy_used.value
        return d


@dataclass
class EmbeddedAPIConfig:
    """API configuration extracted from a URL's query parameters."""
    api_url: str = ""
    api_key: str = ""
    api_key_header: str = ""     # Header name for the key
    fields: List[str] = field(default_factory=list)
    endpoint: str = ""
    extra_params: Dict[str, str] = field(default_factory=dict)
    is_valid: bool = False

    def to_dict(self):
        return asdict(self)


class SmartScraper:
    """
    AI-Powered Universal Web Scraper.

    Usage:
        scraper = SmartScraper(llm_model=gemini_model)
        result = scraper.scrape("https://example.com/products")
        # result.data = [{...}, {...}, ...]  structured records
        # result.schema = {"columns": [...], "types": [...]}
    """

    # Known SPA / playground patterns where data comes from APIs, not HTML
    SPA_PATTERNS = [
        "playgrounds.repliers.com",
        "playground.",
        "swagger.",
        "api-explorer.",
        "developer.",
        "sandbox.",
    ]

    # Known API key parameter names in URLs
    API_KEY_PARAMS = [
        "apiKey", "api_key", "apikey", "key", "token",
        "access_token", "accessToken", "auth",
    ]

    # Known API URL parameter names
    API_URL_PARAMS = [
        "apiUrl", "api_url", "apiurl", "baseUrl", "base_url",
        "endpoint", "server", "host",
    ]

    def __init__(self, llm_model=None, db_connection=None, headless: bool = True):
        self.llm = llm_model
        self.db = db_connection
        self.headless = headless
        self._log: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(
        self,
        url: str,
        description: str = "",
        strategy: Optional[ScrapeStrategy] = None,
        max_pages: int = 5,
        load_to_db: bool = True,
    ) -> ScrapeResult:
        """
        Main entry point.  Give it a URL → get structured data back.

        Args:
            url: The website URL to scrape
            description: Optional plain-English description of what data to extract
            strategy: Force a strategy, or None for auto-detect
            max_pages: Max pagination pages to follow
            load_to_db: If True, auto-create SQLite table from results
        """
        start = time.time()
        self._log = []
        self._log_event("start", f"Starting scrape of {url}")

        # Step 0: Check if URL contains embedded API credentials
        embedded_api = self._extract_embedded_api(url)
        if embedded_api and embedded_api.is_valid:
            self._log_event(
                "api_discovered",
                f"Embedded API found: {embedded_api.api_url} (key: {embedded_api.api_key[:8]}...)"
            )
            # Override strategy to direct API
            strategy = ScrapeStrategy.DIRECT_API

        # Step 1: AI analyzes the URL and decides strategy
        if strategy is None:
            strategy = self._ai_select_strategy(url, description)
            self._log_event("strategy", f"AI selected strategy: {strategy.value}")
        else:
            self._log_event("strategy", f"Strategy: {strategy.value}")

        # Step 2: Execute the chosen strategy
        result = ScrapeResult(
            success=False,
            strategy_used=strategy,
            url=url,
            scraped_at=datetime.utcnow().isoformat() + "Z",
        )

        try:
            if strategy == ScrapeStrategy.DIRECT_API and embedded_api and embedded_api.is_valid:
                # Use the extracted API config to call the API directly
                result = self._execute_direct_api(embedded_api, description, max_pages, result)
            elif strategy == ScrapeStrategy.DIRECT_API:
                # No embedded API but URL looks like a raw JSON endpoint
                result = self._try_direct_json_fetch(url, result)
            elif strategy == ScrapeStrategy.API_SNIFF:
                result = self._execute_api_sniff(url, description, max_pages, result)
            elif strategy == ScrapeStrategy.DOM_PARSE:
                result = self._execute_dom_parse(url, description, max_pages, result)
            elif strategy == ScrapeStrategy.AI_ANALYZED:
                result = self._execute_ai_analysis(url, description, result)
            elif strategy == ScrapeStrategy.HYBRID:
                # Step A: Try direct JSON fetch first (cheapest, fastest)
                result = self._try_direct_json_fetch(url, result)
                # Step B: If no JSON, try API sniff
                if not result.data:
                    self._log_event("fallback", "Not a raw JSON URL, trying API sniff...")
                    result = self._execute_api_sniff(url, description, max_pages, result)
                # Step C: If still nothing, try DOM parse
                if not result.data:
                    self._log_event("fallback", "API sniff found nothing, falling back to DOM parse")
                    result = self._execute_dom_parse(url, description, max_pages, result)
                # Step D: Last resort, AI analysis on raw HTML
                if not result.data:
                    self._log_event("fallback", "DOM parse found nothing, trying AI analysis")
                    result = self._execute_ai_analysis(url, description, result)
        except Exception as e:
            self._log_event("error", f"Strategy failed: {str(e)}")
            result.warnings.append(f"Primary strategy failed: {str(e)}")

            # Auto-fallback chain
            if not result.data:
                self._log_event("fallback", "Trying direct JSON fetch as fallback...")
                try:
                    result = self._try_direct_json_fetch(url, result)
                except Exception:
                    pass
            if not result.data and strategy != ScrapeStrategy.AI_ANALYZED:
                self._log_event("fallback", "Falling back to AI analysis")
                try:
                    result = self._execute_ai_analysis(url, description, result)
                except Exception as e2:
                    result.warnings.append(f"Fallback also failed: {str(e2)}")

        # Step 3: AI generates clean schema from raw data
        if result.data:
            result = self._ai_generate_schema(result, description)
            result.success = True
            result.row_count = len(result.data)
            self._log_event("success", f"Extracted {result.row_count} records with {len(result.columns)} columns")

            # Step 4: Load into database if requested
            if load_to_db and self.db:
                self._load_to_database(result)
                self._log_event("db_loaded", f"Loaded into table: {result.table_name}")

        result.duration_seconds = round(time.time() - start, 2)
        result.execution_log = self._log
        return result

    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Pre-analysis: AI examines a URL and tells you what it can extract.
        No actual scraping happens - just intelligence gathering.
        """
        self._log_event("analyze", f"Pre-analyzing {url}")

        analysis = {
            "url": url,
            "suggested_strategy": "hybrid",
            "estimated_data_types": [],
            "confidence": 0.0,
            "recommendations": [],
        }

        if not self.llm:
            analysis["recommendations"] = ["LLM not available - will use pattern matching"]
            return analysis

        prompt = f"""Analyze this URL and predict what data can be scraped from it.

URL: {url}

Respond in JSON:
{{
    "site_type": "ecommerce|real_estate|news|social|api|government|other",
    "likely_data": ["list of data fields you expect to find"],
    "has_api": true/false,
    "has_pagination": true/false,
    "anti_scraping": "none|basic|moderate|aggressive",
    "best_strategy": "api_sniff|dom_parse|hybrid|ai_analyzed",
    "confidence": 0.0-1.0,
    "recommendations": ["list of tips for scraping this site"],
    "similar_sites": ["known similar sites for reference"]
}}"""

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            analysis = json.loads(text)
            analysis["url"] = url
        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    # ------------------------------------------------------------------
    # URL Parameter Analysis - Extract embedded API configs
    # ------------------------------------------------------------------

    def _extract_embedded_api(self, url: str) -> Optional[EmbeddedAPIConfig]:
        """
        Parse a URL to detect embedded API credentials and configuration.

        Many developer playgrounds / API explorers embed the API URL and key
        right in the URL query parameters.  For example:
            https://playgrounds.repliers.com/?apiUrl=https://api.repliers.io&apiKey=ABC123

        This method extracts those and returns a ready-to-use API config.
        """
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)

            # Flatten single-value lists
            flat_params: Dict[str, str] = {}
            for k, v in params.items():
                flat_params[k] = v[0] if len(v) == 1 else ",".join(v)

            config = EmbeddedAPIConfig()

            # Find API URL
            for param_name in self.API_URL_PARAMS:
                if param_name in flat_params:
                    raw = flat_params[param_name]
                    config.api_url = unquote(raw)
                    self._log_event("url_parse", f"Found API URL param '{param_name}': {config.api_url}")
                    break

            # Find API Key
            for param_name in self.API_KEY_PARAMS:
                if param_name in flat_params:
                    config.api_key = flat_params[param_name]
                    config.api_key_header = self._guess_api_key_header(parsed.netloc, param_name)
                    self._log_event("url_parse", f"Found API key param '{param_name}': {config.api_key[:8]}...")
                    break

            # Find requested fields
            if "fields" in flat_params:
                config.fields = [f.strip() for f in flat_params["fields"].split(",") if f.strip()]
                self._log_event("url_parse", f"Found {len(config.fields)} requested fields")

            # Find endpoint
            if "endpoint" in flat_params:
                config.endpoint = flat_params["endpoint"]

            # Collect other potentially useful params
            skip_keys = set(self.API_KEY_PARAMS + self.API_URL_PARAMS + [
                "fields", "endpoint", "tab", "zoom", "lng", "lat", "center",
                "dynamicClustering", "dynamicClusterPrecision", "stats",
                "clusterLimit", "clusterPrecision", "nlpVersion",
            ])
            for k, v in flat_params.items():
                if k not in skip_keys:
                    config.extra_params[k] = v

            # Valid if we have at least an API URL or key
            config.is_valid = bool(config.api_url and config.api_key)

            # Even without explicit apiUrl, check if the domain itself is a known playground
            if not config.is_valid and config.api_key:
                for pattern in self.SPA_PATTERNS:
                    if pattern in parsed.netloc:
                        # Try to infer API URL from the playground domain
                        config.api_url = self._infer_api_url(parsed.netloc, config.api_key)
                        if config.api_url:
                            config.is_valid = True
                        break

            return config if config.is_valid else None

        except Exception as e:
            self._log_event("url_parse_error", f"Could not parse URL params: {e}")
            return None

    def _guess_api_key_header(self, domain: str, param_name: str) -> str:
        """Guess the correct HTTP header name for the API key based on domain."""
        domain_lower = domain.lower()

        # Known platform-specific headers
        if "repliers" in domain_lower:
            return "REPLIERS-API-KEY"
        if "stripe" in domain_lower:
            return "Authorization"  # Bearer token
        if "openai" in domain_lower:
            return "Authorization"

        # Generic mappings
        if param_name.lower() in ("token", "access_token", "accesstoken"):
            return "Authorization"

        # Default: X-API-Key
        return "X-API-Key"

    def _infer_api_url(self, playground_domain: str, api_key: str) -> str:
        """Try to infer the API URL from a known playground domain."""
        if "repliers" in playground_domain:
            return "https://api.repliers.io"
        return ""

    # ------------------------------------------------------------------
    # Strategy: Direct API Calling (from embedded credentials)
    # ------------------------------------------------------------------

    def _execute_direct_api(
        self,
        api_config: EmbeddedAPIConfig,
        description: str,
        max_pages: int,
        result: ScrapeResult,
    ) -> ScrapeResult:
        """
        Call a discovered API directly using extracted credentials.
        Handles pagination, rate limiting, and data extraction.

        This is the WOW feature: user pastes a playground URL →
        we extract the API key → call the API → get ALL the data.
        """
        import urllib.request
        import urllib.error

        self._log_event("direct_api", f"Calling API: {api_config.api_url}")

        # Determine endpoints to try - prioritize data-rich endpoints first
        endpoints_to_try = []

        # Common data-rich endpoints first (these tend to have the most useful data)
        priority_endpoints = ["/listings", "/properties", "/products", "/items", "/data"]
        for ep in priority_endpoints:
            endpoints_to_try.append(ep)

        # Then the user-specified endpoint
        if api_config.endpoint:
            ep = f"/{api_config.endpoint}" if not api_config.endpoint.startswith("/") else api_config.endpoint
            if ep not in endpoints_to_try:
                endpoints_to_try.append(ep)

        # Then other common endpoints
        other_endpoints = ["/locations", "/search", "/results", "/records"]
        for ep in other_endpoints:
            if ep not in endpoints_to_try:
                endpoints_to_try.append(ep)

        all_data: List[Dict] = []
        discovered_apis: List[DiscoveredAPI] = []

        for endpoint in endpoints_to_try:
            api_url = api_config.api_url.rstrip("/") + endpoint

            # Build query parameters
            params_parts = []
            if api_config.fields:
                params_parts.append(f"fields={','.join(api_config.fields)}")
            params_parts.append("resultsPerPage=100")
            for k, v in api_config.extra_params.items():
                params_parts.append(f"{k}={v}")

            full_url = f"{api_url}?{'&'.join(params_parts)}" if params_parts else api_url

            self._log_event("api_call", f"Trying endpoint: {endpoint}")

            try:
                # Build request with API key header
                req = urllib.request.Request(full_url)
                req.add_header("Content-Type", "application/json")
                req.add_header("User-Agent", "SmartScraper/1.0")

                if api_config.api_key:
                    header_name = api_config.api_key_header or "X-API-Key"
                    if header_name == "Authorization":
                        req.add_header(header_name, f"Bearer {api_config.api_key}")
                    else:
                        req.add_header(header_name, api_config.api_key)

                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    response_data = json.loads(body)

                # Extract data from response
                records = self._extract_data_from_api_response(response_data, "")
                total_count = 0

                # Check for total count in response
                if isinstance(response_data, dict):
                    for key in ["count", "total", "totalCount", "totalResults"]:
                        if key in response_data and isinstance(response_data[key], (int, float)):
                            total_count = int(response_data[key])
                            break

                if records:
                    self._log_event(
                        "api_success",
                        f"Endpoint {endpoint}: {len(records)} records"
                        + (f" (total available: {total_count:,})" if total_count else "")
                    )

                    # Track discovered API
                    discovered = DiscoveredAPI(
                        url=full_url,
                        method="GET",
                        headers={api_config.api_key_header: api_config.api_key[:8] + "..."},
                        response_type="json",
                        record_count=len(records),
                        is_paginated=total_count > len(records),
                        page_param="pageNum",
                        total_pages=(total_count // 100 + 1) if total_count else 0,
                    )
                    discovered_apis.append(discovered)

                    all_data.extend(records)

                    # Handle pagination
                    if total_count > len(records) and max_pages > 1:
                        self._log_event("pagination", f"Paginating: {total_count:,} total records, fetching up to {max_pages} pages")
                        more_data = self._paginate_direct_api(
                            api_config, endpoint, params_parts, len(records), total_count, max_pages
                        )
                        all_data.extend(more_data)

                    # If we found good data on this endpoint, stop trying others
                    if len(all_data) >= 5:
                        break

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self._log_event("rate_limit", f"Rate limited on {endpoint}, waiting 3s...")
                    time.sleep(3)
                elif e.code == 404:
                    self._log_event("api_404", f"Endpoint {endpoint} not found, trying next...")
                elif e.code == 401 or e.code == 403:
                    self._log_event("api_auth_error", f"Auth failed on {endpoint} (HTTP {e.code})")
                else:
                    self._log_event("api_error", f"HTTP {e.code} on {endpoint}")
                continue
            except Exception as e:
                self._log_event("api_error", f"Error on {endpoint}: {str(e)[:80]}")
                continue

        result.data = all_data
        result.discovered_apis = discovered_apis
        if all_data:
            self._log_event("direct_api_done", f"Total extracted: {len(all_data)} records from {len(discovered_apis)} endpoints")
        else:
            self._log_event("direct_api_empty", "No data extracted from any endpoint")
            result.warnings.append("Direct API calls returned no data")

        return result

    def _paginate_direct_api(
        self,
        api_config: EmbeddedAPIConfig,
        endpoint: str,
        base_params: List[str],
        first_page_count: int,
        total_count: int,
        max_pages: int,
    ) -> List[Dict]:
        """Fetch additional pages from a paginated API."""
        import urllib.request
        import urllib.error

        all_data: List[Dict] = []
        results_per_page = max(first_page_count, 100)

        for page in range(2, max_pages + 1):
            api_url = api_config.api_url.rstrip("/") + endpoint
            params = list(base_params) + [f"pageNum={page}"]
            full_url = f"{api_url}?{'&'.join(params)}"

            self._log_event("pagination", f"Page {page}/{max_pages}...")

            try:
                req = urllib.request.Request(full_url)
                req.add_header("Content-Type", "application/json")
                req.add_header("User-Agent", "SmartScraper/1.0")

                if api_config.api_key:
                    header_name = api_config.api_key_header or "X-API-Key"
                    if header_name == "Authorization":
                        req.add_header(header_name, f"Bearer {api_config.api_key}")
                    else:
                        req.add_header(header_name, api_config.api_key)

                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    response_data = json.loads(body)

                records = self._extract_data_from_api_response(response_data, "")
                if not records:
                    self._log_event("pagination", f"Page {page}: empty response, stopping")
                    break

                all_data.extend(records)
                self._log_event("pagination", f"Page {page}: +{len(records)} records (total: {first_page_count + len(all_data)})")

                # Rate limit protection
                time.sleep(1)

                if len(records) < results_per_page:
                    self._log_event("pagination", "Partial page received, likely last page")
                    break

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self._log_event("rate_limit", f"Rate limited on page {page}, waiting 5s...")
                    time.sleep(5)
                    continue  # Retry
                else:
                    self._log_event("pagination_error", f"HTTP {e.code} on page {page}, stopping")
                    break
            except Exception as e:
                self._log_event("pagination_error", f"Error on page {page}: {str(e)[:60]}")
                break

        return all_data

    # ------------------------------------------------------------------
    # Strategy: Direct JSON Fetch (for raw API URLs)
    # ------------------------------------------------------------------

    def _try_direct_json_fetch(self, url: str, result: ScrapeResult) -> ScrapeResult:
        """
        Try to fetch the URL directly and parse as JSON.
        Works for raw API endpoints like https://fakestoreapi.com/products.
        This is the fastest and simplest strategy - just GET the URL and parse.
        """
        self._log_event("direct_fetch", f"Trying direct JSON fetch: {url}")

        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                },
            )

            with urllib.request.urlopen(req, timeout=20) as resp:
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8", errors="replace")

            # Try to parse as JSON
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                # Not JSON - check if it looks like JSON that's wrapped in something
                # (e.g., JSONP callback)
                jsonp_match = re.search(r'(?:callback\s*\()?\s*(\[.*\]|\{.*\})\s*\)?;?\s*$', body, re.DOTALL)
                if jsonp_match:
                    data = json.loads(jsonp_match.group(1))
                else:
                    self._log_event("direct_fetch", "Response is not JSON, skipping direct fetch")
                    return result

            # Process the parsed JSON
            if isinstance(data, list):
                # Direct array of records
                records = [r for r in data if isinstance(r, dict)]
                if records:
                    result.data = records
                    result.strategy_used = ScrapeStrategy.DIRECT_API
                    self._log_event(
                        "direct_fetch_success",
                        f"Direct JSON fetch: {len(records)} records (array response)"
                    )
                    return result

            elif isinstance(data, dict):
                # Object response - find the data array inside
                records = self._extract_data_from_api_response(data, "")
                if records:
                    result.data = records
                    result.strategy_used = ScrapeStrategy.DIRECT_API
                    self._log_event(
                        "direct_fetch_success",
                        f"Direct JSON fetch: {len(records)} records (object response)"
                    )
                    return result

                # Maybe the dict itself is a single record with useful data
                if len(data) >= 3:
                    result.data = [data]
                    result.strategy_used = ScrapeStrategy.DIRECT_API
                    self._log_event(
                        "direct_fetch_success",
                        "Direct JSON fetch: 1 record (single object response)"
                    )
                    return result

            self._log_event("direct_fetch", "JSON parsed but no structured records found")

        except urllib.error.HTTPError as e:
            self._log_event("direct_fetch", f"HTTP error {e.code}")
        except Exception as e:
            self._log_event("direct_fetch", f"Not a direct JSON endpoint: {str(e)[:60]}")

        return result

    # ------------------------------------------------------------------
    # Strategy: API Sniffing (CDP Network Interception)
    # ------------------------------------------------------------------

    def _execute_api_sniff(
        self, url: str, description: str, max_pages: int, result: ScrapeResult
    ) -> ScrapeResult:
        """
        Open page in headless Chrome, intercept ALL network requests,
        find JSON API responses, and extract data from them.
        """
        self._log_event("api_sniff", "Starting browser with CDP network interception")

        from .api_sniffer import APISniffer

        sniffer = APISniffer(headless=self.headless)
        try:
            discovered_apis, raw_data = sniffer.sniff(url, wait_seconds=8)

            result.discovered_apis = discovered_apis
            self._log_event("apis_found", f"Discovered {len(discovered_apis)} APIs")

            if discovered_apis:
                # Use AI to pick the best API (the one with the most data)
                best_api = self._ai_pick_best_api(discovered_apis, description)

                if best_api and best_api.sample_response:
                    # Extract data from the API response
                    data = self._extract_data_from_api_response(
                        best_api.sample_response, best_api.data_path
                    )
                    if data:
                        result.data = data
                        self._log_event("extracted", f"Extracted {len(data)} records from API: {best_api.url[:80]}")

                        # Handle pagination
                        if best_api.is_paginated and max_pages > 1:
                            more_data = sniffer.paginate_api(
                                best_api, max_pages=max_pages
                            )
                            if more_data:
                                result.data.extend(more_data)
                                self._log_event("paginated", f"Fetched {len(more_data)} more records via pagination")

            # If no API data, try to extract from DOM while browser is open
            if not result.data and raw_data:
                result.data = raw_data
                self._log_event("dom_fallback", f"Using DOM-extracted data: {len(raw_data)} records")

        finally:
            sniffer.close()

        return result

    # ------------------------------------------------------------------
    # Strategy: DOM Parsing (Selenium + AI selectors)
    # ------------------------------------------------------------------

    def _execute_dom_parse(
        self, url: str, description: str, max_pages: int, result: ScrapeResult
    ) -> ScrapeResult:
        """
        Open page, use AI to generate CSS selectors, extract structured data.
        """
        self._log_event("dom_parse", "Starting intelligent DOM parsing")

        from .dom_scraper import DOMScraper

        scraper = DOMScraper(llm_model=self.llm, headless=self.headless)
        try:
            data = scraper.scrape(url, description=description, max_pages=max_pages)
            result.data = data
            self._log_event("dom_extracted", f"DOM parsing extracted {len(data)} records")
        finally:
            scraper.close()

        return result

    # ------------------------------------------------------------------
    # Strategy: AI Analysis (LLM reads raw HTML and extracts data)
    # ------------------------------------------------------------------

    def _execute_ai_analysis(
        self, url: str, description: str, result: ScrapeResult
    ) -> ScrapeResult:
        """
        Fetch page HTML, send to LLM, ask it to extract structured data.
        Works for any page but limited by context window.
        """
        self._log_event("ai_analysis", "Using AI to directly analyze page content")

        if not self.llm:
            result.warnings.append("LLM not available for AI analysis")
            return result

        # Fetch HTML (lightweight, no browser needed)
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
        except Exception as e:
            self._log_event("fetch_error", f"Could not fetch URL: {e}")
            result.warnings.append(f"Could not fetch URL: {e}")
            return result

        # Truncate HTML to fit in context window (keep first 15k chars)
        html_truncated = html[:15000]

        prompt = f"""You are a data extraction expert. Extract ALL structured data from this webpage.

URL: {url}
User wants: {description or 'All available structured data'}

HTML (truncated):
{html_truncated}

INSTRUCTIONS:
1. Identify ALL repeating data items (products, listings, records, etc.)
2. Extract each item as a JSON object with consistent keys
3. Use clean, snake_case field names
4. Parse prices as numbers, dates as ISO strings
5. Return ONLY a JSON array of objects

Respond with ONLY the JSON array, no explanation:
[
  {{"field1": "value1", "field2": "value2", ...}},
  ...
]"""

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()

            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list) and len(data) > 0:
                    result.data = data
                    result.ai_analysis = f"AI extracted {len(data)} records directly from HTML"
                    self._log_event("ai_extracted", f"AI extracted {len(data)} records")
        except Exception as e:
            self._log_event("ai_error", f"AI extraction failed: {e}")
            result.warnings.append(f"AI extraction error: {e}")

        return result

    # ------------------------------------------------------------------
    # AI Helpers
    # ------------------------------------------------------------------

    def _ai_select_strategy(self, url: str, description: str) -> ScrapeStrategy:
        """AI picks the best scraping strategy for this URL."""
        if not self.llm:
            return ScrapeStrategy.HYBRID

        # Quick heuristics first
        url_lower = url.lower()
        parsed = urlparse(url)

        # Detect raw JSON API endpoints
        # Patterns: /api/..., .json, known API domains, /v1/..., /v2/...
        api_indicators = [
            "/api/", "/api.", ".json", "/v1/", "/v2/", "/v3/",
            "/rest/", "/graphql", "/query",
        ]
        api_domain_indicators = [
            "api.", "api-", "jsonplaceholder", "fakestoreapi",
            "restcountries", "pokeapi", "swapi",
            "data.gov", "opendata",
        ]

        is_api_url = any(ind in url_lower for ind in api_indicators)
        is_api_domain = any(ind in parsed.netloc.lower() for ind in api_domain_indicators)

        if is_api_url or is_api_domain:
            self._log_event("heuristic", f"Detected raw API endpoint URL")
            return ScrapeStrategy.DIRECT_API

        # SPA/Playground detection - these need API sniffing, not DOM/HTML parsing
        for pattern in self.SPA_PATTERNS:
            if pattern in parsed.netloc.lower():
                self._log_event("heuristic", f"Detected SPA/playground pattern: {pattern}")
                return ScrapeStrategy.API_SNIFF

        prompt = f"""You are a web scraping expert. Pick the BEST strategy to extract data from this URL.

URL: {url}
Description: {description or 'Extract all structured data'}

Strategies:
1. "api_sniff" - Open in browser, intercept XHR/fetch API calls via Chrome DevTools Protocol.
   Best for: SPAs, React/Vue/Angular apps, sites that load data via AJAX.
2. "dom_parse" - Parse the HTML DOM with CSS selectors.
   Best for: Server-rendered pages, static HTML, simple listing pages.
3. "hybrid" - Try API sniffing first, fall back to DOM parsing.
   Best for: Unknown sites where you're not sure of the architecture.
4. "ai_analyzed" - Fetch raw HTML and use AI to extract data.
   Best for: Simple pages, small data sets, irregular HTML structures.

Respond with ONLY the strategy name (one of: api_sniff, dom_parse, hybrid, ai_analyzed):"""

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip().lower().replace('"', '').replace("'", "")
            strategy_map = {
                "api_sniff": ScrapeStrategy.API_SNIFF,
                "dom_parse": ScrapeStrategy.DOM_PARSE,
                "hybrid": ScrapeStrategy.HYBRID,
                "ai_analyzed": ScrapeStrategy.AI_ANALYZED,
            }
            return strategy_map.get(text, ScrapeStrategy.HYBRID)
        except Exception:
            return ScrapeStrategy.HYBRID

    def _ai_pick_best_api(
        self, apis: List[DiscoveredAPI], description: str
    ) -> Optional[DiscoveredAPI]:
        """AI picks which discovered API has the data we want."""
        if not apis:
            return None
        if len(apis) == 1:
            return apis[0]

        # Simple heuristic: pick the one with most records
        best = max(apis, key=lambda a: a.record_count)
        if best.record_count > 0:
            return best

        # If all zero, pick the first JSON one
        for api in apis:
            if api.response_type == "json":
                return api

        return apis[0]

    def _ai_generate_schema(self, result: ScrapeResult, description: str) -> ScrapeResult:
        """AI generates a clean schema from the extracted data."""
        if not result.data:
            return result

        # Flatten nested records before schema generation
        flat_data = [self._flatten_record(r) for r in result.data[:50] if isinstance(r, dict)]

        # Get columns from flattened data
        all_keys = set()
        for record in flat_data:
            all_keys.update(record.keys())

        result.columns = sorted(list(all_keys))

        # Generate table name from URL
        from urllib.parse import urlparse
        parsed = urlparse(result.url)
        domain = parsed.netloc.replace("www.", "").split(".")[0]
        path = parsed.path.strip("/").replace("/", "_").replace("-", "_")
        table_name = f"scraped_{domain}"
        if path:
            table_name += f"_{path[:30]}"
        # Clean up table name
        table_name = re.sub(r'[^a-zA-Z0-9_]', '', table_name).lower()
        result.table_name = table_name

        # AI-enhanced schema with types
        if self.llm and result.data:
            sample = json.dumps(result.data[:3], indent=2, default=str)
            prompt = f"""Given this scraped data, generate a clean schema.

Sample data (first 3 records):
{sample}

User wanted: {description or 'All data'}

Respond in JSON:
{{
    "table_name": "suggested_table_name",
    "description": "one line description",
    "columns": [
        {{"name": "col_name", "type": "TEXT|INTEGER|REAL|DATE", "description": "what this column contains"}}
    ]
}}"""
            try:
                response = self.llm.generate_content(prompt)
                text = response.text.strip()
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    schema = json.loads(json_match.group())
                    result.schema = schema
                    if schema.get("table_name"):
                        result.table_name = "scraped_" + re.sub(
                            r'[^a-zA-Z0-9_]', '', schema["table_name"]
                        ).lower()
            except Exception:
                pass

        return result

    def _extract_data_from_api_response(
        self, response: Any, data_path: str = ""
    ) -> List[Dict]:
        """Extract data array from an API response, auto-detecting the data path."""
        if isinstance(response, list):
            return [r for r in response if isinstance(r, dict)]

        if not isinstance(response, dict):
            return []

        # If data_path is given, use it
        if data_path:
            parts = data_path.split(".")
            current = response
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    break
            if isinstance(current, list):
                return [r for r in current if isinstance(r, dict)]

        # Auto-detect: find the largest array in the response
        best_key = ""
        best_list: List = []
        for key, value in response.items():
            if isinstance(value, list) and len(value) > len(best_list):
                # Check if items are dicts (structured data)
                if value and isinstance(value[0], dict):
                    best_key = key
                    best_list = value

        if best_list:
            self._log_event("data_path", f"Auto-detected data at key: '{best_key}' ({len(best_list)} items)")
            return best_list

        # Maybe the response itself is a single record
        if len(response) > 3:
            return [response]

        return []

    # ------------------------------------------------------------------
    # Database Integration
    # ------------------------------------------------------------------

    def _flatten_record(self, record: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """Flatten nested dicts into dot-notation keys.
        e.g. {"address": {"city": "LA"}} -> {"address_city": "LA"}
        """
        items: List[Tuple[str, Any]] = []
        for k, v in record.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_record(v, new_key, sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings
                items.append((new_key, json.dumps(v, default=str) if v else None))
            else:
                items.append((new_key, v))
        return dict(items)

    def _load_to_database(self, result: ScrapeResult) -> None:
        """Load scraped data into SQLite table."""
        if not self.db or not result.data:
            return

        import pandas as pd

        try:
            # Flatten nested dicts first (e.g. address.city -> address_city)
            flat_data = [self._flatten_record(r) for r in result.data if isinstance(r, dict)]

            df = pd.DataFrame(flat_data)

            # Convert any remaining complex types to JSON strings
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list)) else x
                    )

            df.to_sql(result.table_name, self.db, if_exists="replace", index=False)
            result.row_count = len(df)
            result.columns = list(df.columns)
            # Update data with flattened version for the preview
            result.data = flat_data
            self._log_event("db_loaded", f"Table '{result.table_name}' created with {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            result.warnings.append(f"DB load failed: {e}")
            self._log_event("db_error", str(e))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, message: str):
        """Add entry to execution log."""
        entry = {
            "time": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            "message": message,
        }
        self._log.append(entry)
        print(f"  [{event_type}] {message}")
