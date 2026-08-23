"""Tests for DevPilot Browser MCP's SSRF guard, including the Entry 42
gap-fix: fetch_page used to validate only the original URL, then handed
the request to a client configured to auto-follow redirects, so a page
could 302 straight past the guard to an internal/metadata address.

Uses a fake httpx2.Client (no real network calls) so these are fast,
deterministic, and don't depend on any external site's availability.
"""

import pytest

from mcp_servers.browser import server as browser_server


class _FakeResponse:
    def __init__(self, status_code, text="", location=None):
        self.status_code = status_code
        self.text = text
        self.headers = {"location": location} if location else {}
        self.is_redirect = 300 <= status_code < 400 and location is not None


class _FakeClient:
    """Replays a scripted sequence of responses, one per call to .get()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requested_urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        self.requested_urls.append(url)
        return self._responses.pop(0)


def _install_fake_client(monkeypatch, responses):
    fake = _FakeClient(responses)
    monkeypatch.setattr(browser_server.httpx2, "Client", lambda **kwargs: fake)
    return fake


def test_fetch_page_returns_real_text(monkeypatch):
    _install_fake_client(monkeypatch, [_FakeResponse(200, text="<p>hello world</p>")])
    result = browser_server.fetch_page("https://example.com/docs")
    assert "hello world" in result


def test_redirect_to_metadata_endpoint_is_blocked(monkeypatch):
    """The core gap: a 302 pointing at the cloud metadata IP must be
    rejected on the SAME terms as if it had been the original URL --
    not silently followed."""
    fake = _install_fake_client(
        monkeypatch,
        [_FakeResponse(302, location="http://169.254.169.254/latest/meta-data/")],
    )
    with pytest.raises(ValueError, match="private/internal address"):
        browser_server.fetch_page("https://example.com/looks-safe")
    # Only the first hop was actually requested -- the unsafe redirect
    # target must never reach a real HTTP call.
    assert fake.requested_urls == ["https://example.com/looks-safe"]


def test_redirect_to_localhost_is_blocked(monkeypatch):
    _install_fake_client(
        monkeypatch,
        [_FakeResponse(302, location="http://localhost:8001/health")],
    )
    with pytest.raises(ValueError):
        browser_server.fetch_page("https://example.com/looks-safe")


def test_redirect_chain_within_limit_still_works(monkeypatch):
    _install_fake_client(
        monkeypatch,
        [
            _FakeResponse(302, location="https://example.com/step2"),
            _FakeResponse(200, text="<p>final content</p>"),
        ],
    )
    result = browser_server.fetch_page("https://example.com/step1")
    assert "final content" in result


def test_too_many_redirects_is_rejected(monkeypatch):
    responses = [
        _FakeResponse(302, location=f"https://example.com/step{i}")
        for i in range(browser_server.MAX_REDIRECTS + 2)
    ]
    _install_fake_client(monkeypatch, responses)
    with pytest.raises(RuntimeError, match="Too many redirects"):
        browser_server.fetch_page("https://example.com/step0")


def test_direct_localhost_url_is_still_blocked(monkeypatch):
    """Original (pre-Entry-42) protection: unaffected by this fix."""
    with pytest.raises(ValueError):
        browser_server.fetch_page("http://localhost/")
