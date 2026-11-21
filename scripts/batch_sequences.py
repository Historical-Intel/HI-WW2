#!/usr/bin/env python3
"""Make private fixed-length causal language model batches from numeric IDs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
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
    if subprocess.run(["git", "check-ignore", "-q", "data/private/batch-check"], cwd=ROOT).returncode:
        raise ValueError("Private output is not Git ignored")
    if subprocess.check_output(["git", "ls-files", "--", "data/private"], cwd=ROOT):
        raise ValueError("Private output contains tracked files")


def latest_input():
    runs = sorted(p for p in PRIVATE.glob("tokenized-*") if (p / "manifest.json").is_file())
    if not runs:
        raise ValueError("No private tokenized run found")
    return runs[-1]


def sequences(row, length, pad_id):
    ids = row["input_ids"]
    if not ids or len(ids) != row["id_count"]:
        raise ValueError("Invalid tokenized row")
    for index, start in enumerate(range(0, len(ids), length)):
        active = ids[start:start + length]
        padding = length - len(active)
        yield {
            "input_ids": active + [pad_id] * padding,
            "attention_mask": [1] * len(active) + [0] * padding,
            "labels": active + [-100] * padding,
            "metadata": {key: row.get(key) for key in
                         ("source", "title", "chapter", "chunk_index", "shuffle_index",
                          "page_start", "page_end")}
            | {"sequence_index": index, "start_id": start, "active_ids": len(active)},
        }


def write_report(output, manifest):
    """Write a private, content-free report for inspecting the batching step."""
    lines = ["# Private tokenization 2 report", "",
             f"Input run: `{manifest['input_run']}`  ",
             f"Tokenizer: `{manifest['tokenizer_kind']}`  ",
             f"Sequence length: {manifest['sequence_length']} IDs  ",
             f"Batch size: up to {manifest['batch_size']} sequences  ",
             f"Pad ID: {manifest['pad_id']}  ",
             "", "| Split | Source chunks | Sequences | Batches | Real IDs | Padding IDs | Padding share |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in SPLITS:
        item = manifest["splits"][split]
        total = item["real_ids"] + item["padding_ids"]
        lines.append(f"| {split} | {item['source_chunks']:,} | {item['sequences']:,} | "
                     f"{item['batches']:,} | {item['real_ids']:,} | {item['padding_ids']:,} | "
                     f"{item['padding_ids'] / total:.1%} |")
    lines += ["", "Each JSONL row is one batch. It contains equally sized `input_ids`, "
              "`attention_mask`, and `labels` arrays plus per-sequence source metadata.",
              "Padding positions have mask 0 and label -100. Source chunks are split only "
              "within their original book and dataset split.",
              "", "The original text chunks already overlap; this step does not add overlap. "
              "No sequence crosses a chunk boundary, so the transition between two sequences "
              "is not used as a next-token target.",
              "", "The current byte tokenizer is an inspection baseline. Rerun both tokenization "
              "steps with the exact target model tokenizer and valid pad ID before model use.",
              "Rights status remains unverified.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="private tokenized run; default: latest")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pad-id", type=int, help="required for model tokenizers without a pad ID")
    args = parser.parse_args()
    check_private()
    source = args.input or latest_input()
    if not source.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("Input must be under data/private")
    if not 2 <= args.sequence_length <= 16384 or not 1 <= args.batch_size <= 1024:
        raise ValueError("Invalid sequence length or batch size")
    input_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = input_manifest["tokenizer"]
    pad_id = args.pad_id if args.pad_id is not None else (
        0 if tokenizer["kind"] == "utf8_bytes" else tokenizer.get("pad_token_id"))
    if pad_id is None or type(pad_id) is not int or pad_id < 0:
        raise ValueError("A non-negative pad ID is required for this tokenizer")
    os.umask(0o077)
    output = PRIVATE / ("batches-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(mode=0o700)
    report = {"input_run": source.name, "tokenizer_kind": tokenizer["kind"],
              "sequence_length": args.sequence_length, "batch_size": args.batch_size,
              "pad_id": pad_id, "assignments": input_manifest["assignments"],
              "training_permission_verified": False, "splits": {}}
    for split in SPLITS:
        stats = Counter()
        pending = []
        with (source / f"{split}.jsonl").open(encoding="utf-8") as infile, (
            output / f"{split}.jsonl"
        ).open("w", encoding="utf-8") as outfile:
            def flush():
                if pending:
                    outfile.write(json.dumps({"batch_index": stats["batches"],
                        "input_ids": [s["input_ids"] for s in pending],
                        "attention_mask": [s["attention_mask"] for s in pending],
                        "labels": [s["labels"] for s in pending],
                        "metadata": [s["metadata"] for s in pending]}, ensure_ascii=False) + "\n")
                    stats["batches"] += 1
                    pending.clear()

            for line in infile:
                row = json.loads(line)
                if input_manifest["assignments"].get(row["source"]) != split:
                    raise ValueError("Input book crosses split boundary")
                stats["source_chunks"] += 1
                for sequence in sequences(row, args.sequence_length, pad_id):
                    pending.append(sequence)
                    stats["sequences"] += 1
                    stats["real_ids"] += sequence["metadata"]["active_ids"]
                    stats["padding_ids"] += args.sequence_length - sequence["metadata"]["active_ids"]
                    if len(pending) == args.batch_size:
                        flush()
            flush()
        if stats["source_chunks"] != input_manifest["counts"][split]:
            raise ValueError("Source chunk count mismatch")
        if stats["real_ids"] != input_manifest["total_ids"][split]:
            raise ValueError("Token IDs lost during batching")
        report["splits"][split] = dict(stats)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output, report)
    print(f"Private batches and report: {output.relative_to(ROOT)}")
    print("Batches: " + ", ".join(f"{s}={report['splits'][s]['batches']}" for s in SPLITS))


if __name__ == "__main__":
    main()
