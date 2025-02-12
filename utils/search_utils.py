import logging
from typing import Dict, Any, List, Optional
from serpapi import GoogleSearch

logger = logging.getLogger(__name__)

def search_serpapi(query: str, serpapi_config: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Performs a Google search using SerpAPI and returns organic results.
    
    Args:
        query: Search query string
        serpapi_config: Configuration dictionary for SerpAPI
    
    Returns:
        List of dictionaries containing search results, each with 'title' and 'snippet',
        or None if the search fails
    """
    api_key = serpapi_config.get("api_key")
    if not api_key:
        logger.error("SerpAPI API key is missing in configuration.")
        return None

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": serpapi_config.get("num_results", 5),
            "gl": serpapi_config.get("gl", "mx"),
            "hl": serpapi_config.get("hl", "es"),
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        if not organic_results:
            return []

        formatted_results = []
        for result in organic_results:
            snippet = result.get("snippet")
            title = result.get("title")
            if snippet and title:
                formatted_results.append({
                    "title": title,
                    "snippet": snippet
                })

        return formatted_results

    except Exception as e:
        logger.error(f"Error performing SerpAPI search: {e}")
        return None