"""
Web Search Integration

Provides web search capability using Google Custom Search API.
Allows the AI to look up current information during calls.
"""

import logging
import json
from typing import Optional, List, Dict, Any
import os

logger = logging.getLogger(__name__)

# Settings file path
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")


def _load_settings() -> dict:
    """Load settings from file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except:
            pass
    return {}


def get_search_config() -> tuple:
    """
    Get Google Custom Search configuration from settings.

    Returns:
        (api_key, cse_id) tuple, or (None, None) if not configured
    """
    settings = _load_settings()
    api_keys = settings.get("api_keys", {})

    api_key = api_keys.get("GOOGLE_API_KEY", "")
    cse_id = api_keys.get("GOOGLE_CSE_ID", "")

    if not api_key or not cse_id:
        return None, None

    return api_key, cse_id


def is_search_configured() -> bool:
    """Check if web search is properly configured"""
    api_key, cse_id = get_search_config()
    return bool(api_key and cse_id)


def search(
    query: str,
    num_results: int = 5,
    site_restrict: str = None
) -> List[Dict[str, Any]]:
    """
    Perform a Google Custom Search.

    Args:
        query: Search query
        num_results: Number of results to return (max 10)
        site_restrict: Optional site to restrict search to

    Returns:
        List of search results with title, link, snippet
    """
    api_key, cse_id = get_search_config()

    if not api_key or not cse_id:
        logger.warning("Google Custom Search not configured")
        return []

    try:
        import requests

        # Build search URL
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": min(num_results, 10)
        }

        if site_restrict:
            params["siteSearch"] = site_restrict

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logger.error(f"Search API error: {response.status_code}")
            return []

        data = response.json()

        # Parse results
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "displayLink": item.get("displayLink", "")
            })

        logger.info(f"Search for '{query}' returned {len(results)} results")
        return results

    except requests.exceptions.Timeout:
        logger.error("Search request timed out")
        return []
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


def search_and_summarize(query: str, num_results: int = 3) -> str:
    """
    Search and return a formatted summary of results.

    Args:
        query: Search query
        num_results: Number of results to include

    Returns:
        Formatted string with search results
    """
    results = search(query, num_results)

    if not results:
        return ""

    lines = [f"WEB SEARCH RESULTS for '{query}':"]

    for i, result in enumerate(results, 1):
        lines.append(f"\n{i}. {result['title']}")
        lines.append(f"   {result['snippet']}")
        lines.append(f"   Source: {result['displayLink']}")

    return "\n".join(lines)


def get_current_info(topic: str) -> str:
    """
    Get current information about a topic.

    Useful for looking up:
    - Current prices
    - Business hours
    - Recent news
    - Product availability

    Args:
        topic: Topic to search for

    Returns:
        Formatted search results or empty string
    """
    if not is_search_configured():
        return ""

    return search_and_summarize(topic, num_results=3)


# Integration with LLM for dynamic searches during calls
class WebSearchTool:
    """
    Web search tool for LLM integration.

    Can be used to dynamically search during calls when
    the AI needs current information.
    """

    def __init__(self):
        self.enabled = is_search_configured()
        self.cache = {}  # Simple in-memory cache

    def should_search(self, text: str) -> bool:
        """
        Determine if a search would be helpful.

        Looks for phrases like:
        - "let me look that up"
        - "I'm not sure about current..."
        - "what's the latest..."
        """
        search_triggers = [
            "look that up",
            "check on that",
            "current price",
            "latest info",
            "today's hours",
            "availability",
            "in stock"
        ]

        text_lower = text.lower()
        return any(trigger in text_lower for trigger in search_triggers)

    def search(self, query: str) -> str:
        """
        Perform a cached search.

        Args:
            query: Search query

        Returns:
            Search results string
        """
        if not self.enabled:
            return ""

        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Perform search
        result = search_and_summarize(query, num_results=3)

        # Cache result
        if result:
            self.cache[cache_key] = result

        return result


if __name__ == "__main__":
    # Test the search
    logging.basicConfig(level=logging.INFO)

    print("Testing Google Custom Search Integration")
    print("=" * 50)

    if not is_search_configured():
        print("\nSearch not configured.")
        print("Add GOOGLE_API_KEY and GOOGLE_CSE_ID in Settings > API Keys")
    else:
        print("\nSearch is configured. Testing...")

        # Test search
        results = search("Python programming", num_results=3)
        print(f"\nFound {len(results)} results:")
        for r in results:
            print(f"  - {r['title']}")
            print(f"    {r['snippet'][:100]}...")

        # Test summarize
        print("\n" + "=" * 50)
        print(search_and_summarize("weather forecast"))
