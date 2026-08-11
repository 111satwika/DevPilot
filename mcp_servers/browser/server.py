"""DevPilot Browser / Documentation MCP Server.

Stage 5: two tools.
  - search_web: scrapes DuckDuckGo's HTML endpoint (no API key). Markup
    confirmed by fetching a real results page rather than assumed.
  - fetch_page: fetches a URL and extracts readable text, guarded against
    SSRF (internal/private/loopback hosts rejected before fetching).

Full rationale logged in DevPilot_AI_Implementation_Log.html Entry 9.
"""

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx2

from mcp.server import MCPServer

REQUEST_TIMEOUT_SECONDS = 10
MAX_PAGE_TEXT_CHARS = 5000
DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (DevPilot AI learning project)"

mcp = MCPServer("DevPilot Browser")


def _extract_real_url(redirect_href: str | None) -> str:
    """DuckDuckGo's result links wrap the real URL in a `uddg` query param."""
    if not redirect_href:
        return ""
    absolute = redirect_href if redirect_href.startswith("http") else f"https:{redirect_href}"
    query = parse_qs(urlparse(absolute).query)
    return unquote(query.get("uddg", [""])[0])


class _ResultParser(HTMLParser):
    """Extracts {title, url, snippet} from DuckDuckGo's HTML results markup."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._field: str | None = None
        self._parts: list[str] = []
        self._pending_href: str | None = None

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "")
        if tag == "a" and "result__a" in classes:
            self._field = "title"
            self._pending_href = dict(attrs).get("href")
            self._parts = []
        elif tag == "a" and "result__snippet" in classes:
            self._field = "snippet"
            self._parts = []

    def handle_data(self, data):
        if self._field:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag != "a" or not self._field:
            return
        text = "".join(self._parts).strip()
        if self._field == "title":
            self.results.append(
                {"title": text, "url": _extract_real_url(self._pending_href), "snippet": ""}
            )
        elif self._field == "snippet" and self.results:
            self.results[-1]["snippet"] = text
        self._field = None


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via DuckDuckGo's HTML endpoint (no API key required)."""
    try:
        with httpx2.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.get(
                DDG_HTML_ENDPOINT, params={"q": query}, headers={"User-Agent": USER_AGENT}
            )
    except httpx2.RequestError as exc:
        raise ConnectionError(f"Could not reach DuckDuckGo: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"DuckDuckGo search returned unexpected status {response.status_code}.")

    parser = _ResultParser()
    parser.feed(response.text)
    results = [r for r in parser.results if r["url"]]
    return results[:max_results]


class _TextExtractor(HTMLParser):
    """Strips tags/script/style and collects visible text."""

    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.chunks.append(text)


def _reject_unsafe_url(url: str) -> None:
    """SSRF guard: only http(s), and never an internal/private/loopback host."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Refusing to fetch '{url}': only http/https URLs are allowed.")

    hostname = (parsed.hostname or "").lower()
    if hostname in ("localhost",):
        raise ValueError(f"Refusing to fetch '{url}': internal host is not allowed.")

    try:
        resolved_ip = socket.gethostbyname(hostname)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host '{hostname}': {exc}") from exc

    ip = ipaddress.ip_address(resolved_ip)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError(
            f"Refusing to fetch '{url}': resolves to a private/internal address ({resolved_ip})."
        )


@mcp.tool()
def fetch_page(url: str) -> str:
    """Fetch a URL and return its readable text content, capped in length."""
    _reject_unsafe_url(url)

    try:
        with httpx2.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": USER_AGENT})
    except httpx2.RequestError as exc:
        raise ConnectionError(f"Could not fetch '{url}': {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Fetching '{url}' returned unexpected status {response.status_code}.")

    extractor = _TextExtractor()
    extractor.feed(response.text)
    text = re.sub(r"\s+", " ", " ".join(extractor.chunks)).strip()

    if len(text) > MAX_PAGE_TEXT_CHARS:
        text = text[:MAX_PAGE_TEXT_CHARS] + " [truncated]"

    return text


if __name__ == "__main__":
    mcp.run()
