#!/usr/bin/env python3
"""Make private text windows and book-disjoint dataset splits from cleaned PDFs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
WORD = re.compile(r"\S+")


def private_check():
    if PRIVATE.is_symlink() or PRIVATE.resolve() != PRIVATE:
        raise ValueError("Private directory must not be a symlink")
    if subprocess.run(["git", "check-ignore", "-q", "data/private/check"], cwd=ROOT).returncode:
        raise ValueError("Private output is not Git ignored")
    if subprocess.check_output(["git", "ls-files", "--", "data/private"], cwd=ROOT):
        raise ValueError("Private output contains tracked files")


def latest_run():
    candidates = []
    for path in PRIVATE.iterdir():
        manifest = path / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            if data.get("resources") and all(r.get("status") == "extracted_pending_review" for r in data["resources"]):
                candidates.append(path)
    if not candidates:
        raise ValueError("No complete private extraction run found")
    return max(candidates)


def load_config(path, sources):
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != set(sources):
        raise ValueError("Private review config must contain exactly every source filename")
    for source, item in config.items():
        ranges = item.get("core_pages")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"Missing reviewed core page ranges for {source}")
        previous = 0
        for pair in ranges:
            if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair):
                raise ValueError("Ranges must contain integer [start, end] pairs")
            first, last = pair
            if first <= previous or last < first or last > sources[source]["pages"]:
                raise ValueError("Core page ranges overlap or exceed the PDF")
            previous = last
        for chapter in item.get("chapters", []):
            if (not isinstance(chapter.get("name"), str) or
                type(chapter.get("start")) is not int or
                type(chapter.get("end")) is not int or
                chapter["start"] > chapter["end"]):
                raise ValueError("Invalid chapter metadata")
    return config


def selected(page, ranges):
    return any(start <= page <= end for start, end in ranges)


def make_windows(rows, size, overlap, source, config):
    # Keep the source page for every token. Windows never cross book boundaries.
    tokens = []
    for row in rows:
        if selected(row["page"], config["core_pages"]):
            for match in WORD.finditer(row["text"]):
                tokens.append((match.group(), row["page"]))
    stride = size - overlap
    windows = []
    for start in range(0, len(tokens), stride):
        part = tokens[start:start + size]
        if not part:
            break
        text = " ".join(word for word, _ in part)
        page_start, page_end = part[0][1], part[-1][1]
        chapters = config.get("chapters", [])
        chapter = next((ch["name"] for ch in chapters
                        if ch["start"] <= page_start <= ch["end"]), None)
        windows.append({"source": source, "title": config.get("title", source),
                        "chapter": chapter, "chunk_index": len(windows),
                        "page_start": page_start, "page_end": page_end,
                        "token_count": len(part), "text": text})
        if start + size >= len(tokens):
            break
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="private extraction run; default: latest complete")
    parser.add_argument("--config", type=Path, default=PRIVATE / "segmentation_review.json")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()
    private_check()
    if not 1 <= args.window <= 8192 or not 0 <= args.overlap < args.window:
        raise ValueError("Window must be positive and overlap smaller than window")
    run = args.run or latest_run()
    if not run.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("Input run must be inside the private directory")
    manifest = json.loads((run / "manifest.json").read_text())
    resources = {r["source"]: r for r in manifest["resources"]}
    if any(r.get("status") != "extracted_pending_review" for r in resources.values()):
        raise ValueError("All source extractions must be complete")
    config = load_config(args.config, resources)
    if len(resources) < 3:
        raise ValueError("At least three books are needed for book-level train/validation/test splits")
    # Stable whole-book assignment. Never shuffle or split overlapping windows.
    ordered = sorted(resources, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    assignments = {source: ("test" if i == 0 else "validation" if i == 1 else "train")
                   for i, source in enumerate(ordered)}
    os.umask(0o077)
    output = PRIVATE / ("segments-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(mode=0o700)
    counts = Counter()
    seen = set()
    handles = {split: (output / f"{split}.jsonl").open("w", encoding="utf-8")
               for split in ("train", "validation", "test")}
    try:
        for source in ordered:
            entry = resources[source]
            rows = [json.loads(line) for line in (run / entry["output_directory"] / "cleaned_pages.jsonl").open()]
            for window in make_windows(rows, args.window, args.overlap, source, config[source]):
                # Exact duplicate windows are dropped only within the assigned split.
                key = hashlib.sha256(window["text"].encode()).hexdigest()
                if key in seen:
                    counts["duplicates_removed"] += 1
                    continue
                seen.add(key)
                split = assignments[source]
                handles[split].write(json.dumps(window, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (output / "manifest.json").write_text(json.dumps({"source_run": run.name,
        "assignments": assignments, "window_whitespace_tokens": args.window,
        "overlap_whitespace_tokens": args.overlap, "counts": dict(counts),
        "training_permission_verified": False}, indent=2) + "\n")
    print(f"Private segments: {output.relative_to(ROOT)}; counts: {dict(counts)}")
    print("Whitespace tokens are a reproducible approximation; final model tokenizer may differ.")


if __name__ == "__main__":
    main()
