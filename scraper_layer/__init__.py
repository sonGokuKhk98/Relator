"""
AI-Powered Smart Scraper Layer

Automatic API discovery + intelligent web scraping.
Give it a URL, AI figures out the best way to extract data.

Three strategies (auto-selected):
1. API Sniffer   - Intercept network requests via CDP (like Repliers)
2. DOM Scraper   - Intelligent page parsing with AI selectors (like Zillow)
3. Direct API    - If user provides API docs/endpoints directly
"""

from .smart_scraper import SmartScraper, ScrapeResult, ScrapeStrategy
from .api_sniffer import APISniffer
from .dom_scraper import DOMScraper
from .schema_generator import SchemaGenerator

__all__ = [
    "SmartScraper",
    "ScrapeResult",
    "ScrapeStrategy",
    "APISniffer",
    "DOMScraper",
    "SchemaGenerator",
]
