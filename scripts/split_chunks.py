#!/usr/bin/env python3
"""Reassign private chunks by whole book to train, validation, and test."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
SPLITS = ("train", "validation", "test")


def check_private():
    if PRIVATE.is_symlink() or PRIVATE.resolve() != PRIVATE:
        raise ValueError("Private directory must not be a symlink")
    if subprocess.run(["git", "check-ignore", "-q", "data/private/split-check"], cwd=ROOT).returncode:
        raise ValueError("Private output is not Git ignored")
    if subprocess.check_output(["git", "ls-files", "--", "data/private"], cwd=ROOT):
        raise ValueError("Private output contains tracked files")


def latest_segments():
    runs = sorted(p for p in PRIVATE.glob("segments-*") if (p / "manifest.json").is_file())
    if not runs:
        raise ValueError("No private segmentation run found")
    return runs[-1]


def choose(groups, ratios, seed):
    names = sorted(groups)
    if len(names) < 3:
        raise ValueError("At least three source books are needed")
    total = sum(map(len, groups.values()))
    candidates = []
    for assignment in product(SPLITS, repeat=len(names)):
        if set(assignment) != set(SPLITS):
            continue
        counts = Counter()
        for name, split in zip(names, assignment):
            counts[split] += len(groups[name])
        error = sum(abs(counts[s] / total - target) for s, target in zip(SPLITS, ratios))
        tie = hashlib.sha256(f"{seed}:{assignment}".encode()).hexdigest()
        candidates.append((round(error, 12), tie, assignment))
    return dict(zip(names, min(candidates)[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="private segments run; default: latest")
    parser.add_argument("--train", type=float, default=.7)
    parser.add_argument("--validation", type=float, default=.2)
    parser.add_argument("--test", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    check_private()
    source = args.input or latest_segments()
    if not source.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("Input must be under data/private")
    ratios = (args.train, args.validation, args.test)
    if any(not 0 < ratio < 1 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Ratios must be positive and sum to 1")
    input_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for split in SPLITS:
        rows = [json.loads(line) for line in (source / f"{split}.jsonl").open(encoding="utf-8")]
        if len(rows) != input_manifest["counts"][split]:
            raise ValueError("Input count mismatch")
        for row in rows:
            if input_manifest["assignments"][row["source"]] != split:
                raise ValueError("Input book crosses splits")
            row.pop("shuffle_index", None)
            groups[row["source"]].append(row)
    assignments = choose(groups, ratios, args.seed)
    os.umask(0o077)
    output = PRIVATE / ("split-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(mode=0o700)
    counts = Counter()
    for split in SPLITS:
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for name in sorted(groups):
                if assignments[name] == split:
                    for row in sorted(groups[name], key=lambda r: r["chunk_index"]):
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[split] += 1
    total = sum(counts.values())
    report = {"input_run": source.name, "seed": args.seed,
              "target_ratios": dict(zip(SPLITS, ratios)),
              "actual_ratios": {s: counts[s] / total for s in SPLITS},
              "assignments": assignments, "counts": dict(counts),
              "method": "minimum absolute ratio error, whole-book groups",
              "training_permission_verified": False}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Private split: {output.relative_to(ROOT)}")
    print("Counts: " + ", ".join(f"{s}={counts[s]}" for s in SPLITS))
    print("Ratios: " + ", ".join(f"{s}={report['actual_ratios'][s]:.1%}" for s in SPLITS))


if __name__ == "__main__":
    main()
