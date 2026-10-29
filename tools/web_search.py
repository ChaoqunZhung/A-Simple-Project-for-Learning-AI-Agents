"""Tavily search; optional credentials and recoverable upstream errors."""
import logging
import httpx

logger = logging.getLogger(__name__)


def web_search(query: str, api_key: str, max_results: int = 5) -> dict:
    if not api_key:
        return {"error": "尚未配置 TAVILY_API_KEY，无法搜索互联网。请在 .env 中配置后重启。", "results": []}
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("query must contain 1–2000 characters")
    if type(max_results) is not int or not 1 <= max_results <= 8:
        raise ValueError("max_results must be between 1 and 8")
    try:
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            response = client.post("https://api.tavily.com/search", json={
                "api_key": api_key, "query": query, "max_results": max_results,
                "search_depth": "basic", "include_answer": False,
            })
            response.raise_for_status()
            payload = response.json()
        results = [{"title": str(item.get("title", ""))[:300],
                    "url": str(item.get("url", ""))[:2000],
                    "content": str(item.get("content", ""))[:1800],
                    "score": item.get("score")}
                   for item in payload.get("results", [])[:max_results]]
        return {"query": query, "results": results}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Tavily request failed (%s)", type(exc).__name__)
        return {"error": "Web Search 请求失败，请检查 Tavily 配置、额度或网络后重试。", "results": []}
