"""Security regressions for feed URLs, redirects and model credentials."""
import socket
from types import SimpleNamespace

import pytest
import digest
from conftest import FeedResponse, rss
from digestlib import feeds


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "javascript:alert(1)",
    "http://127.0.0.1/latest/meta-data",
    "http://169.254.169.254/latest/meta-data",
    "http://10.0.0.1/rss", "http://[::1]/rss",
    "https://localhost/rss", "https://feeds.local/rss",
    "https://user:password@news.example/rss",
    "https://news.example:bad/rss",
    "https://news.example/rss\nInjected: true",
])
def test_unsafe_feed_urls_are_blocked(url):
    with pytest.raises(ValueError):
        feeds._public_feed_url(url)


def test_private_dns_answer_is_blocked(monkeypatch):
    def local_dns(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.9", port))]
    monkeypatch.setattr(socket, "getaddrinfo", local_dns)
    with pytest.raises(ValueError, match="private"):
        feeds._public_feed_url("https://rebind.example/rss")


def test_redirect_to_metadata_service_is_blocked_before_request(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=302,
                               headers={"Location": "http://169.254.169.254/latest/meta-data"})

    monkeypatch.setattr(digest.requests, "get", get)
    with pytest.raises(ValueError, match="private"):
        digest.fetch_feed("https://news.example/feed")
    assert [url for url, _ in calls] == ["https://news.example/feed"]
    assert calls[0][1]["allow_redirects"] is False


def test_public_redirect_is_validated(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url == "https://news.example/rss":
            return SimpleNamespace(status_code=302, headers={"Location": "/new.xml"})
        return FeedResponse(rss({"title": "Redirected"}))

    monkeypatch.setattr(digest.requests, "get", get)
    parsed = digest.fetch_feed("https://news.example/rss")
    assert parsed.entries[0].title == "Redirected"
    assert calls == ["https://news.example/rss", "https://news.example/new.xml"]


def test_feed_size_is_limited(monkeypatch):
    monkeypatch.setattr(digest.requests, "get", lambda url, **kwargs:
                        FeedResponse(b"x" * (feeds.MAX_FEED_BYTES + 1)))
    with pytest.raises(ValueError, match="size limit"):
        digest.fetch_feed("https://news.example/rss")


@pytest.mark.parametrize("url", [
    "https://user:password@news.example/story",
    "https://news.example/story\nEvil: yes",
    "https://news.example:bad/story", "javascript:alert(1)",
])
def test_untrusted_article_urls_are_not_links(url):
    assert digest._web_link(url) == ""


def test_model_calls_do_not_follow_redirects(transport):
    transport.script("gem-a", SimpleNamespace(status_code=302, text="", headers={}))
    from digestlib import models
    models._call_model("gemini", "gem-a", "prompt", "topic", "system", {})
    assert transport.calls[0]["allow_redirects"] is False
