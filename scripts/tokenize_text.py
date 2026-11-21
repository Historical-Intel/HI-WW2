#!/usr/bin/env python3
"""Convert private text chunks to numeric IDs and write a private inspection report.

Default byte IDs are a reversible learning baseline. For fine-tuning, pass a
local directory containing the exact target model's Hugging Face tokenizer.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
SPLITS = ("train", "validation", "test")


def check_private():
    if PRIVATE.is_symlink() or PRIVATE.resolve() != PRIVATE:
        raise ValueError("Private directory must not be a symlink")
    if subprocess.run(["git", "check-ignore", "-q", "data/private/token-check"], cwd=ROOT).returncode:
        raise ValueError("Private output is not Git ignored")
    if subprocess.check_output(["git", "ls-files", "--", "data/private"], cwd=ROOT):
        raise ValueError("Private output contains tracked files")


def latest_input():
    runs = sorted(p for p in PRIVATE.glob("shuffled-*") if (p / "manifest.json").is_file())
    if not runs:
        raise ValueError("No private shuffled run found")
    return runs[-1]


def tokenizer_for(path):
    if path is None:
        return (lambda text: list(text.encode("utf-8"))), (lambda ids: bytes(ids).decode("utf-8")), {
            "kind": "utf8_bytes", "vocab_size": 256,
            "model_compatible": False,
            "note": "Educational baseline; IDs must be regenerated with the target model tokenizer.",
        }
    if not path.is_dir():
        raise ValueError("Tokenizer must be a local directory")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install transformers to use --tokenizer-path") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    vocab = tokenizer.get_vocab()
    identity = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()
    return (
        lambda text: tokenizer.encode(text, add_special_tokens=False, truncation=False),
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        {"kind": "huggingface_local", "name": path.name, "vocab_size": len(vocab),
         "vocab_sha256": identity, "model_compatible": "verify tokenizer matches chosen model"},
    )


def write_report(output, manifest, lengths, roundtrip_failures, examples):
    """Write an inspectable private Markdown report without copying source prose."""
    lines = ["# Private tokenization report", "",
             f"Input dataset: `{manifest['input_run']}`  ",
             f"Tokenizer: `{manifest['tokenizer']['kind']}`  ",
             f"Vocabulary size: {manifest['tokenizer']['vocab_size']:,}  ",
             "No truncation or padding was applied. Token IDs remain in the private dataset.", "",
             "| Split | Chunks | Total IDs | Min | Median | Max | Exact decode mismatches |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in SPLITS:
        values = lengths[split]
        lines.append(f"| {split} | {len(values):,} | {sum(values):,} | "
                     f"{min(values) if values else 0:,} | "
                     f"{statistics.median(values) if values else 0:,.0f} | "
                     f"{max(values) if values else 0:,} | {roundtrip_failures[split]:,} |")
    lines += ["", "## Sample numeric IDs", "",
              "Only the first 16 IDs from one chunk per split are shown. No source text is shown.", ""]
    for split in SPLITS:
        lines.append(f"- {split}: `{examples.get(split, [])}`")
    lines += ["", "## Inspection notes", "",
              "- The source, page span, chapter field, chunk index, and shuffle index are preserved in each JSONL row.",
              "- Split assignments and chunk counts are copied from the input manifest and checked during conversion.",
              "- UTF-8 byte IDs are reversible but are not the vocabulary of a pretrained language model.",
              "- Model tokenizers may normalize text, so exact decoding can differ; mismatches are counted above.",
              "- Padding, attention masks, and batching belong to the next tokenization step.",
              "- The rights status remains unverified; this report does not authorize training or distribution.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="private shuffled run; default: latest")
    parser.add_argument("--tokenizer-path", type=Path, help="local target-model tokenizer directory")
    args = parser.parse_args()
    check_private()
    source = args.input or latest_input()
    if not source.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("Input must be under data/private")
    encode, decode, tokenizer_info = tokenizer_for(args.tokenizer_path)
    input_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    os.umask(0o077)
    output = PRIVATE / ("tokenized-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(mode=0o700)
    lengths = {split: [] for split in SPLITS}
    failures = Counter()
    examples = {}
    for split in SPLITS:
        with (source / f"{split}.jsonl").open(encoding="utf-8") as infile, (
            output / f"{split}.jsonl"
        ).open("w", encoding="utf-8") as outfile:
            for line in infile:
                row = json.loads(line)
                if input_manifest["assignments"].get(row["source"]) != split:
                    raise ValueError("Input book crosses split boundary")
                original = row.pop("text")
                ids = encode(original)
                if not ids or any(type(value) is not int or value < 0 for value in ids):
                    raise ValueError("Tokenizer returned invalid IDs")
                if decode(ids) != original:
                    failures[split] += 1
                lengths[split].append(len(ids))
                examples.setdefault(split, ids[:16])
                row["input_ids"] = ids
                row["id_count"] = len(ids)
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
        if len(lengths[split]) != input_manifest["counts"][split]:
            raise ValueError("Input chunk count mismatch")
    manifest = {"input_run": source.name, "tokenizer": tokenizer_info,
                "assignments": input_manifest["assignments"],
                "counts": {split: len(lengths[split]) for split in SPLITS},
                "total_ids": {split: sum(lengths[split]) for split in SPLITS},
                "exact_decode_mismatches": dict(failures),
                "training_permission_verified": False}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output, manifest, lengths, failures, examples)
    print(f"Private token IDs and report: {output.relative_to(ROOT)}")
    print("Chunks: " + ", ".join(f"{split}={len(lengths[split])}" for split in SPLITS))


if __name__ == "__main__":
    main()
