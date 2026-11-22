"""Reading each topic's feeds: fetching them, cleaning the text, and choosing
which articles go to the model.
"""

import html
import ipaddress
import re
import socket
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlsplit
from concurrent.futures import ThreadPoolExecutor

import feedparser
import requests
from dateutil import parser as dateparser


# Most articles a topic can send to the model. This is well above a quiet
# topic's count, and _interleave_by_feed() decides what fits on busy days.
MAX_ARTICLES_PER_TOPIC = 100

# RSS summaries usually lead with the substance, so a short snippet leaves
# room for more articles in the prompt.
SNIPPET_CHARS = 300

FEED_FETCH_TIMEOUT = 20

# Feeds fetched at once per topic. Enough to keep a slow topic to a couple
# of timeout rounds without hitting any one host too hard.
FEED_FETCH_WORKERS = 8

USER_AGENT = (
    "daily-digest/1.0 (+https://github.com/Sathish-Rajmohan/AI-Daily-Digest; "
    "RSS reader for personal digest)"
)


def parse_entry_time(entry):
    """Best-effort publish time for a feedparser entry, in UTC.

    feedparser's parsed times come first because it understands zone
    abbreviations like EDT. dateutil treats those as UTC, which puts US feeds
    hours out. The raw strings are the fallback.
    """
    for key in ("published_parsed", "updated_parsed"):
        struct = entry.get(key)
        if struct:
            try:
                return datetime(*struct[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pass
    for key in ("published", "updated", "created"):
        if key in entry:
            try:
                dt = dateparser.parse(entry[key])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (ValueError, TypeError, OverflowError):
                pass
    return None


def _clean_text(text):
    """Plain text from a feed title or summary. Feeds send these as HTML, so
    strip the tags and decode entities like &amp;.
    """
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# Network restrictions apply to the original URL and every redirect target.
MAX_FEED_REDIRECTS = 5
MAX_FEED_BYTES = 5_000_000
_REDIRECT_CODES = {301, 302, 303, 307, 308}


def _web_link(url):
    """Only absolute, uncredentialed web links may appear in a digest."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url or any(ord(c) < 32 or ord(c) == 127 or c == "\\" for c in url):
        return ""
    try:
        parts = urlsplit(url)
        if (parts.scheme.lower() not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None):
            return ""
        _ = parts.port
    except ValueError:
        return ""
    return url


def _public_feed_url(url):
    """Block private/non-public addresses, invalid URLs and internal hosts.

    A malicious DNS server could still rebind between checking and connecting,
    so use trusted feed configurations on networks with privileged access.
    """
    if not _web_link(url):
        raise ValueError("invalid or unsafe feed URL")
    parts = urlsplit(url)
    host = parts.hostname.rstrip(".").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("local feed hostname blocked")
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        if not addresses:
            raise ValueError("feed hostname has no addresses")
        if any(not ipaddress.ip_address(info[4][0].split("%", 1)[0]).is_global
               for info in addresses):
            raise ValueError("private or non-public feed address blocked")
    except (OSError, UnicodeError) as exc:
        raise ValueError("could not resolve feed hostname") from exc
    return url


def _feed_label(url):
    """Show only hostnames in logs; feed URLs can contain API tokens."""
    try:
        return urlsplit(url).hostname or "(invalid host)"
    except (ValueError, TypeError):
        return "(invalid host)"


def fetch_feed(feed_url):
    """Fetch RSS with bounded redirects; never let feedparser fetch URLs."""
    current = feed_url
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }
    for redirect in range(MAX_FEED_REDIRECTS + 1):
        _public_feed_url(current)
        resp = requests.get(current, timeout=FEED_FETCH_TIMEOUT,
                            headers=headers, allow_redirects=False)
        if resp.status_code in _REDIRECT_CODES:
            location = resp.headers.get("Location")
            if not location or redirect == MAX_FEED_REDIRECTS:
                raise ValueError("feed redirect missing target or exceeded redirect limit")
            current = urljoin(current, location)
            continue
        resp.raise_for_status()
        data = resp.content
        if len(data) > MAX_FEED_BYTES:
            raise ValueError("feed response exceeds size limit")
        return feedparser.parse(data)
    raise ValueError("too many redirects")


def _safe_fetch_feed(feed_url):
    """fetch_feed(), but logs any error and returns None so one bad feed doesn't
    stop the others.
    """
    try:
        return fetch_feed(feed_url)
    except Exception as e:
        print(f"  [warn] failed to fetch feed from {_feed_label(feed_url)} ({type(e).__name__})",
              file=sys.stderr)
        return None


def fetch_topic_articles(topic, lookback_hours):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    feed_urls = topic.get("feeds", [])
    by_feed = []
    seen_links = set()

    # Fetched in parallel, since a long list of dead feeds could otherwise use the
    # whole run's time. Results are handled in the listed order, so duplicate
    # links resolve the same way on every run.
    with ThreadPoolExecutor(max_workers=FEED_FETCH_WORKERS) as pool:
        fetched = list(pool.map(_safe_fetch_feed, feed_urls))

    for feed_url, parsed in zip(feed_urls, fetched):
        if parsed is None:
            continue

        if getattr(parsed, "bozo", False) and not parsed.entries:
            print(f"  [warn] feed unreadable, skipping: {feed_url}", file=sys.stderr)
            continue

        # Use the host name when the feed's title is empty.
        source_name = (
            _clean_text(parsed.feed.get("title")) or urlsplit(feed_url).netloc or feed_url
        )
        from_this_feed = []

        for entry in parsed.entries:
            pub_time = parse_entry_time(entry)
            # Keep undated items. Some feeds that are otherwise fine don't date posts.
            if pub_time is not None and pub_time < cutoff:
                continue

            link = _web_link(entry.get("link"))
            if link and link in seen_links:
                continue
            if link:
                seen_links.add(link)

            summary = _clean_text(entry.get("summary", "") or entry.get("description", ""))

            from_this_feed.append({
                "title": _clean_text(entry.get("title")) or "(untitled)",
                "link": link,
                "summary": summary,
                "source": source_name,
                "published": pub_time.isoformat() if pub_time else None,
            })

        if from_this_feed:
            from_this_feed.sort(key=lambda a: a["published"] or "", reverse=True)
            by_feed.append(from_this_feed)

    return _interleave_by_feed(by_feed, MAX_ARTICLES_PER_TOPIC)


def _interleave_by_feed(by_feed, cap):
    """Take one article from each feed in turn, newest first within each feed,
    until the cap is reached.

    Cutting a time-sorted list lets one feed that posts every few minutes fill
    every slot. Taking turns keeps every outlet in, and a big story is one that
    many outlets cover.
    """
    picked = []
    round_index = 0
    while len(picked) < cap:
        added = False
        for feed_articles in by_feed:
            if round_index < len(feed_articles):
                picked.append(feed_articles[round_index])
                added = True
                if len(picked) >= cap:
                    break
        if not added:
            break  # every feed exhausted
        round_index += 1
    return picked
