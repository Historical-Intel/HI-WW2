#!/usr/bin/env python3
"""Reproducibly shuffle private text chunks within each book-disjoint split."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
SPLITS = ("train", "validation", "test")


def verify_private():
    if PRIVATE.is_symlink() or PRIVATE.resolve() != PRIVATE:
        raise ValueError("Private directory must not be a symlink")
    if subprocess.run(["git", "check-ignore", "-q", "data/private/shuffle-check"], cwd=ROOT).returncode:
        raise ValueError("Private output is not Git ignored")
    if subprocess.check_output(["git", "ls-files", "--", "data/private"], cwd=ROOT):
        raise ValueError("Private files are tracked by Git")


def latest_segments():
    runs = sorted(p for p in PRIVATE.glob("segments-*") if (p / "manifest.json").is_file())
    if not runs:
        raise ValueError("No segmentation run found")
    return runs[-1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="private segments directory; default: latest")
    parser.add_argument("--seed", type=int, default=42, help="reproducible seed (default: 42)")
    args = parser.parse_args()
    verify_private()
    source = args.input or latest_segments()
    if not source.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("Input must be inside data/private")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    assignments = manifest["assignments"]
    os.umask(0o077)
    output = PRIVATE / ("shuffled-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(mode=0o700)
    checks = {}
    for split in SPLITS:
        infile = source / f"{split}.jsonl"
        rows = [json.loads(line) for line in infile.open(encoding="utf-8")]
        expected = manifest["counts"].get(split, 0)
        if len(rows) != expected or any(assignments.get(row.get("source")) != split for row in rows):
            raise ValueError(f"Invalid source assignment or count in {split}")
        # Derive a separate deterministic stream for each split.
        split_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{split}".encode()).digest()[:8], "big")
        random.Random(split_seed).shuffle(rows)
        outfile = output / f"{split}.jsonl"
        with outfile.open("w", encoding="utf-8") as stream:
            for position, row in enumerate(rows):
                row["shuffle_index"] = position
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        checks[split] = {"rows": len(rows), "input_sha256": sha256(infile),
                         "output_sha256": sha256(outfile)}
    result = {"input_run": source.name, "seed": args.seed,
              "algorithm": "Python random.Random seeded per split with SHA-256(seed:split)",
              "assignments": assignments, "counts": {s: checks[s]["rows"] for s in SPLITS},
              "checksums": checks, "training_permission_verified": False}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Private shuffled dataset: {output.relative_to(ROOT)}")
    print("Counts: " + ", ".join(f"{s}={checks[s]['rows']}" for s in SPLITS))


if __name__ == "__main__":
    main()
