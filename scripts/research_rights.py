#!/usr/bin/env python3
"""Collect public rights evidence for the five books into private local reports.

This script records evidence, not legal permission. Source URLs and book IDs
are read from a local Git-ignored JSON config.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
USER_AGENT = "HI-WW2-rights-research/1.0 (local academic rights inventory)"
TERMS = re.compile(r"copyright|permission|license|licence|rights reserved|artificial intelligence|machine learning|training|reprodu", re.I)

def load_sources(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Source config must map local book IDs to URL lists")
    for key, urls in data.items():
        if not isinstance(key, str) or not isinstance(urls, list) or not urls:
            raise ValueError("Invalid source config entry")
        if any(not isinstance(url, str) or urlparse(url).scheme != "https" for url in urls):
            raise ValueError("Only HTTPS source URLs are allowed")
    return data


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []
        self.title = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.hidden:
            self.hidden -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if not self.hidden:
            self.parts.append(data)


def robots_allowed(url):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        request = Request(robots_url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=12) as response:
            if response.status != 200:
                return False, f"robots HTTP {response.status}"
            rules = response.read(512_000).decode("utf-8", "replace").splitlines()
        parser = RobotFileParser()
        parser.parse(rules)
        return parser.can_fetch(USER_AGENT, url), "robots.txt checked"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return False, f"robots unavailable: {type(exc).__name__}"


def fetch(url):
    allowed, reason = robots_allowed(url)
    if not allowed:
        return {"url": url, "status": "skipped", "reason": reason}
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        with urlopen(req, timeout=20) as response:
            content_type = response.headers.get_content_type()
            body = response.read(2_000_001)
            final_url = response.url
            status = response.status
        if len(body) > 2_000_000:
            return {"url": url, "status": "skipped", "reason": "page exceeds 2 MB"}
        if content_type != "text/html":
            return {"url": url, "status": "skipped", "reason": f"unsupported {content_type}"}
        parser = PageText()
        parser.feed(body.decode("utf-8", "replace"))
        text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
        matches = []
        for match in TERMS.finditer(text):
            excerpt = text[max(0, match.start() - 100):match.end() + 180]
            if excerpt not in matches:
                matches.append(excerpt)
            if len(matches) == 15:
                break
        return {
            "url": url, "final_url": final_url, "status": "fetched",
            "http_status": status, "page_title": " ".join(parser.title).strip(),
            "content_sha256": hashlib.sha256(body).hexdigest(),
            "evidence_excerpts": matches,
            "note": "Search hits are evidence leads, not permission for this book or use.",
        }
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {"url": url, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", help="limit to one book ID from private config")
    ap.add_argument("--sources", type=Path, default=PRIVATE / "rights_sources.json")
    args = ap.parse_args()
    if PRIVATE.is_symlink():
        raise SystemExit("Private directory must not be a symlink")
    if subprocess.run(["git", "check-ignore", "-q", "data/private/rights-check.json"], cwd=ROOT).returncode != 0:
        raise SystemExit("data/private must be Git ignored before research")
    if subprocess.check_output(["git", "ls-files", "--", "data/private"], cwd=ROOT):
        raise SystemExit("Private output contains tracked files")
    os.umask(0o077)
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    PRIVATE.chmod(0o700)
    sources = load_sources(args.sources)
    if args.book and args.book not in sources:
        raise SystemExit("Unknown book ID")
    books = {args.book: sources[args.book]} if args.book else sources
    results = {key: [fetch(url) for url in urls] for key, urls in books.items()}
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "rights_status": "unverified",
        "training_permission": False,
        "redistribution_permission": False,
        "books": results,
    }
    target = PRIVATE / "rights_research.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved private rights leads for {len(results)} books to {target.relative_to(ROOT)}")
    for key, pages in results.items():
        print(f"{key}: " + ", ".join(page["status"] for page in pages))


if __name__ == "__main__":
    sys.exit(main())
