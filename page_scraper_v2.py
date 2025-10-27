
#!/usr/bin/env python3
"""
Confluence Page Scraper (improved + robust pagination patch)

Adds protection against buggy or inconsistent `_links.next` in some Confluence DC builds
by validating the server-provided `next` against the actual number of items returned.
If the server advances `start` by the requested limit instead of the actual size, we
correct it to `prev_start + len(results)` and continue.

References:
- Atlassian REST pagination uses relative `_links.next` and start/limit paging.  See: https://developer.atlassian.com/server/confluence/pagination-in-the-rest-api/
- Inconsistent pagination behavior has been observed in some DC versions (e.g.,  attachments endpoint emitting incorrect next). See: https://jira.atlassian.com/browse/CONFSERVER-95396
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from typing import Dict, Iterable, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin, urlparse, parse_qs
from urllib3.util.retry import Retry

LOG = logging.getLogger("confluence_scraper")

# -----------------------------
# Helpers
# -----------------------------

def configure_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


def build_session(token: str, timeout: int = 30, max_retries: int = 5) -> requests.Session:
    """Create a requests.Session with retries and default timeout."""
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    })

    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        status=max_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    # Patch in a default timeout
    _request = s.request

    def _request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return _request(method, url, **kwargs)

    s.request = _request_with_timeout  # type: ignore[assignment]
    return s


def ensure_json_response(resp: requests.Response) -> None:
    ctype = resp.headers.get("Content-Type", "").lower()
    if "application/json" not in ctype:
        snippet = resp.text[:300].replace("", " ")
        raise RuntimeError(
            f"Expected JSON but got {ctype or 'unknown'} for {resp.url}. Snippet: {snippet}"
        )


def get_json(session: requests.Session, url: str, *, params: Optional[dict] = None) -> dict:
    resp = session.get(url, params=params)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        snippet = resp.text[:300].replace("", " ")
        LOG.error("HTTP %s for %s | %s", resp.status_code, resp.url, snippet)
        raise
    ensure_json_response(resp)
    return resp.json()


def clean_html_to_text(html: str) -> str:
    """Convert Confluence storage-format HTML to readable plain text."""
    soup = BeautifulSoup(html or "", "html.parser")

    # Code/pre blocks → fenced code blocks
    for pre in soup.find_all("pre"):
        code_text = pre.get_text("")
        pre.replace_with("```" + code_text.strip() + "```")

    # Links → text (href)
    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        href = a.get("href") or ""
        a.replace_with(f"{text} ({href})" if href else text)

    # Line breaks
    for br in soup.find_all("br"):
        br.replace_with("")

    # Paragraph spacing
    for p in soup.find_all("p"):
        p.insert_after("")

    text = soup.get_text(separator="")
    # Collapse excessive blank lines
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: List[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "".join(out).strip()


# -----------------------------
# Confluence REST API calls
# -----------------------------

def page_api(base_url: str, page_id: str) -> str:
    return f"{base_url.rstrip('/')}/rest/api/content/{page_id}"


def page_children_api(base_url: str, page_id: str) -> str:
    return f"{base_url.rstrip('/')}/rest/api/content/{page_id}/child/page"


def fetch_page(session: requests.Session, base_url: str, page_id: str) -> dict:
    url = page_api(base_url, page_id)
    params = {"expand": "body.storage,version,ancestors,_links"}
    return get_json(session, url, params=params)


# -----------------------------
# Robust pagination (Option B)
# -----------------------------

def _extract_start_from_url(u: str) -> Optional[int]:
    """Return the 'start' query parameter from a URL if present."""
    try:
        q = parse_qs(urlparse(u).query)
        if "start" in q and q["start"]:
            return int(q["start"][0])
    except Exception:
        return None
    return None


def iter_children(
    session: requests.Session,
    base_url: str,
    page_id: str,
    *,
    page_size: int = 250,
) -> Iterable[dict]:
    """
    Iterate all direct child pages of `page_id`, following _links.next.
    If the server's next uses an incorrect start (e.g., +requested limit instead of +actual size),
    correct it to start = prev_start + len(results). Also falls back to manual start= pagination
    when next is missing but the page looked full.
    """

    def url_with_params(start: Optional[int] = None):
        url = page_children_api(base_url, page_id)
        params = {"limit": page_size, "expand": "body.storage,version,ancestors,_links"}
        if start is not None:
            params["start"] = start
        return url, params

    # initial request (no explicit start)
    start = 0
    url, params = url_with_params(start=None)

    while True:
        data = get_json(session, url, params=params)
        results = data.get("results", [])
        limit = int(data.get("limit", page_size) or page_size)
        size = int(data.get("size", len(results)))
        start_from_resp = int(data.get("start", start) or start)
        links = (data.get("_links") or {})
        base_from_resp = links.get("base") or base_url.rstrip("/")
        next_rel = links.get("next")

        LOG.debug(
            "children page: start=%s size=%s limit=%s next=%s base=%s url=%s",
            start_from_resp, size, limit, bool(next_rel), base_from_resp, url,
        )

        # yield items
        for item in results:
            yield item

        # stop when empty
        if size == 0 or len(results) == 0:
            break

        expected_next_start = start_from_resp + len(results)

        if next_rel:
            candidate_next = urljoin(base_from_resp + "/", next_rel.lstrip("/"))
            next_start = _extract_start_from_url(candidate_next)

            if next_start is not None and next_start != expected_next_start:
                LOG.warning(
                    "Adjusting buggy next start: server=%s, expected=%s; overriding.",
                    next_start, expected_next_start,
                )
                url, params = url_with_params(start=expected_next_start)
            else:
                url, params = candidate_next, None  # follow server next
        else:
            # No next link; if page looks full, compute manual next page
            if len(results) >= limit:
                url, params = url_with_params(start=expected_next_start)
            else:
                break


# -----------------------------
# Crawl
# -----------------------------

def to_doc(base_url: str, item: dict, *, include_storage: bool) -> dict:
    body_storage = (((item.get("body") or {}).get("storage") or {}).get("value")) or ""
    doc = {
        "id": item.get("id"),
        "title": item.get("title", ""),
        "url": base_url.rstrip("/") + ((item.get("_links") or {}).get("webui") or ""),
        "version": ((item.get("version") or {}).get("number")),
        "ancestors": [a.get("id") for a in (item.get("ancestors") or []) if isinstance(a, dict)],
        "text": clean_html_to_text(body_storage),
    }
    if include_storage:
        doc["body_storage"] = body_storage
    return doc


def crawl_confluence(
    *,
    base_url: str,
    root_page_id: str,
    session: requests.Session,
    max_workers: int = 8,
    page_size: int = 250,
    include_storage: bool = True,
) -> List[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    visited: Set[str] = set()
    docs: List[dict] = []

    # Fetch root page first
    root = fetch_page(session, base_url, root_page_id)
    docs.append(to_doc(base_url, root, include_storage=include_storage))
    visited.add(str(root_page_id))
    LOG.info("📄 Root: %s (%s)", root.get("title"), root_page_id)

    frontier: List[str] = [str(root_page_id)]
    level = 0
    while frontier:
        level += 1
        LOG.info("🔎 Discovering level %d children for %d page(s)...", level, len(frontier))
        next_frontier: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(list, iter_children(session, base_url, pid, page_size=page_size)): pid
                for pid in frontier
            }
            for fut in as_completed(fut_map):
                pid = fut_map[fut]
                try:
                    children = fut.result()
                except Exception as e:
                    LOG.error("Failed to fetch children for %s: %s", pid, e)
                    continue
                LOG.info("↪️ %d child(ren) for page %s", len(children), pid)
                for ch in children:
                    ch_id = str(ch.get("id"))
                    if not ch_id or ch_id in visited:
                        continue
                    visited.add(ch_id)
                    docs.append(to_doc(base_url, ch, include_storage=include_storage))
                    next_frontier.append(ch_id)
        frontier = next_frontier
    LOG.info("✅ Completed crawl. Total docs: %d", len(docs))
    return docs


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl a Confluence page and its child pages.")
    p.add_argument("--base-url", required=True, help="Confluence base URL, e.g. https://confluence.example.com")
    p.add_argument("--page-id", required=True, help="Root page ID to start from")
    p.add_argument("--token", default=os.getenv("CONFLUENCE_PAT"), help="Personal Access Token (or set CONFLUENCE_PAT)")
    p.add_argument("--out", default="confluence_export.json", help="Output JSON filepath")
    p.add_argument("--max-workers", type=int, default=8, help="Max parallel requests when listing children")
    p.add_argument("--page-size", type=int, default=250, help="API page size for pagination")
    p.add_argument("--no-storage", action="store_true", help="Exclude raw body.storage from the export")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=5, help="Max HTTP retries for transient failures")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if not args.token:
        LOG.error("No token provided. Use --token or set CONFLUENCE_PAT env var.")
        return 2

    session = build_session(args.token, timeout=args.timeout, max_retries=args.retries)
    try:
        docs = crawl_confluence(
            base_url=args.base_url,
            root_page_id=str(args.page_id),
            session=session,
            max_workers=args.max_workers,
            page_size=args.page_size,
            include_storage=(not args.no_storage),
        )
    except Exception as e:
        LOG.exception("Crawl failed: %s", e)
        return 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    LOG.info("📦 Saved %d docs to %s", len(docs), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())