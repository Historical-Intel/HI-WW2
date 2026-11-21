#!/usr/bin/env python3
"""Extract local PDFs into Git-ignored, private, page-addressable text.

No network services or model APIs are used. Page numbers are PDF page numbers
(one-based), not the page numbers printed inside the book.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

try:
    import pymupdf
except ImportError:
    raise SystemExit("Install dependencies with: python3 -m pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
NUMBER = re.compile(r"^(?:[-–—]\s*)?(?:\d+|[ivxlcdm]+)(?:\s*[-–—])?$", re.I)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_private_directory():
    """Refuse redirected, tracked, or non-ignored output before writing text."""
    if PRIVATE.resolve() != PRIVATE or PRIVATE.parent.resolve() != PRIVATE.parent:
        raise ValueError("Private output must not pass through a symlink.")
    tracked = subprocess.run(
        ["git", "ls-files", "--", "data/private"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    )
    if tracked.stdout.strip():
        raise ValueError("Private output contains Git-tracked files; untrack them first.")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", "data/private/privacy-check.txt"], cwd=ROOT,
    )
    if ignored.returncode != 0:
        raise ValueError("data/private/ must be excluded by .gitignore before extraction.")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    return re.sub(r"[ \t]+", " ", text).strip()


def margin_key(text):
    return re.sub(r"\d+", "#", normalize(text).casefold())


def page_lines(page):
    """Retain coordinates to distinguish running margins from body text."""
    records = []
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    for block in page.get_text("dict", sort=True, flags=flags)["blocks"]:
        for line in block.get("lines", []):
            text = normalize("".join(span["text"] for span in line["spans"]))
            if not text:
                continue
            _, top, _, bottom = line["bbox"]
            region = "top" if bottom < page.rect.height * .10 else (
                "bottom" if top > page.rect.height * .90 else "body"
            )
            records.append((text, region))
    return records


def clean_page(lines, recurring):
    retained = []
    removed = 0
    for text, region in lines:
        if region != "body" and (
            NUMBER.fullmatch(text) or (region, margin_key(text)) in recurring
        ):
            removed += 1
        else:
            retained.append(text)
    # Keep line breaks and paragraph boundaries conservative. Hyphenated terms,
    # quotations, headings, footnotes, and historical spellings remain intact.
    return "\n".join(retained).strip(), removed


def excluded_pages(config, filename, count):
    pages = set()
    for interval in config.get(filename, {}).get("exclude_pages", []):
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("exclude_pages entries must be [first, last] pairs.")
        first, last = interval
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= count:
            raise ValueError("Invalid one-based PDF page exclusion range.")
        pages.update(range(first, last + 1))
    return pages


def process_pdf(path, target, config):
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("Encrypted PDF requires a password.")
        if not document.page_count:
            raise ValueError("PDF contains no pages.")
        exclusions = excluded_pages(config, path.name, document.page_count)
        pages = [page_lines(page) for page in document]
        occurrences = Counter()
        for lines in pages:
            occurrences.update({(region, margin_key(text)) for text, region in lines
                                if region != "body" and len(text) <= 160})
        threshold = max(3, math.ceil(len(pages) * .15))
        recurring = {key for key, count in occurrences.items() if count >= threshold}
        target.mkdir(mode=0o700)
        sparse = []
        removed_total = 0
        kept = 0
        with (target / "raw_pages.jsonl").open("w", encoding="utf-8") as raw, (
            target / "cleaned_pages.jsonl"
        ).open("w", encoding="utf-8") as cleaned, (
            target / "cleaned.txt"
        ).open("w", encoding="utf-8") as plain:
            for number, lines in enumerate(pages, 1):
                raw_text = "\n".join(text for text, _ in lines)
                raw.write(json.dumps({"page": number, "text": raw_text}, ensure_ascii=False) + "\n")
                text, removed = clean_page(lines, recurring)
                removed_total += removed
                if len(re.sub(r"\s", "", raw_text)) < 40:
                    sparse.append(number)
                if number in exclusions or not text:
                    continue
                cleaned.write(json.dumps({"source": path.name, "page": number, "text": text},
                                         ensure_ascii=False) + "\n")
                plain.write(text + "\n\n")
                kept += 1
        report = {
            "source": path.name, "pdf_metadata": document.metadata,
            "pages": len(pages), "pages_kept": kept,
            "excluded_pages": sorted(exclusions), "removed_margin_lines": removed_total,
            "recurring_margin_candidates": sorted([list(key) for key in recurring]),
            "sparse_pages_needing_visual_or_ocr_review": sparse,
            "pdf_repaired_by_reader": document.is_repaired,
            "rights_status": "unverified; local extraction does not establish training or redistribution rights",
            "author_metadata_status": "unverified; confirm against the title and copyright pages",
            "front_back_matter_status": "manual page review required; only explicit exclusions are removed",
            "ready_for_training": False,
        }
        write_json(target / "review.json", report)
        return {k: report[k] for k in ("pages", "pages_kept", "removed_margin_lines")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / ".env")
    parser.add_argument("--review-config", type=Path,
                        help="Local JSON mapping exact PDF filenames to exclude_pages ranges")
    args = parser.parse_args()
    check_private_directory()
    sources = sorted(args.input_dir.glob("*.pdf"))
    if not sources:
        raise ValueError("No PDFs found in the input directory.")
    config = json.loads(args.review_config.read_text(encoding="utf-8")) if args.review_config else {}
    if not isinstance(config, dict) or set(config) - {p.name for p in sources}:
        raise ValueError("Review config must map known PDF filenames to review settings.")
    # New files are owner-readable only; no source text is printed to the terminal.
    os.umask(0o077)
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    PRIVATE.chmod(0o700)
    run = PRIVATE / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(mode=0o700)
    entries, seen = [], {}
    for path in sources:
        entry = {"source": path.name}
        try:
            checksum = digest(path)
            entry["sha256"] = checksum
            if checksum in seen:
                entry.update(status="duplicate", duplicate_of=seen[checksum])
            else:
                seen[checksum] = path.name
                directory = run / checksum[:16]
                entry.update(process_pdf(path, directory, config))
                entry.update(status="extracted_pending_review", output_directory=directory.name)
        except Exception as exc:
            # Exceptions are saved locally so damaged books do not hide other results.
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
        entries.append(entry)
    write_json(run / "manifest.json", {"resources": entries, "training_approved": False})
    failures = sum(e["status"] == "error" for e in entries)
    print(f"Processed {len(entries)} PDFs; {failures} errors. Private output: {run.relative_to(ROOT)}")
    print("Review local manifests, sparse pages, metadata, and exclusion ranges before training.")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
