# HI-WW2
Retrieving and Comparing WWII Historical Evidence and Propaganda

## Local resource cleaning

Install dependencies and run the local PDF extraction and cleaning pipeline:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/clean_resources.py
```

Inputs are the PDFs in `.env/`. Each run writes a new directory under
`data/private/`, which is excluded from Git. The script checks that exclusion and
refuses to run if output files are already tracked. It uses no network services.
Private files use owner-only permissions; Git ignore rules do not prevent manual
force-adds, backups, or cloud synchronization.

Each book produces `raw_pages.jsonl`, `cleaned_pages.jsonl`, `cleaned.txt`, and
`review.json`. A run-level `manifest.json` records SHA-256 checksums, exact
duplicates, and errors. Cleaned JSONL preserves the source filename and one-based
PDF page number for citation and later dataset splitting.

Cleaning normalizes Unicode and spacing, removes control characters and standalone
margin page numbers, and removes recurring lines in the outer 10% of pages.
Body headings, quotations, footnotes, and line breaks are retained conservatively.
Review removed margin candidates and sparse pages locally. Sparse pages may need
OCR; the script does not perform OCR or invent missing text.

Front/back matter needs manual review to avoid losing historical content. Create
`data/private/review_config.json`, mapping exact input filenames to inclusive
PDF page ranges, for example:

```json
{
  "example.pdf": {"exclude_pages": [[1, 8], [290, 300]]}
}
```

Then run `python3 scripts/clean_resources.py --review-config data/private/review_config.json`.
Only use ranges verified for each book. Raw extraction is retained privately for
comparison. PDF metadata is a starting point for author and copyright research;
it does not establish permission. Rights and author attribution require review
before training or redistribution. Outputs remain marked as pending review.

## Private research materials

Research source files and extracted text are stored locally under `.env/` and
`data/private/`. These directories are excluded from Git. Do not commit source
files, extracted text, source identities, rights notes, or model artifacts that
reproduce source passages. The collection includes World War I background for
studying the lead-up to World War II.

Rights research uses `python3 scripts/research_rights.py` and a private JSON
mapping of local book IDs to publisher or rights-holder URLs at
`data/private/rights_sources.json`. Results stay under `data/private/`. Web pages
may provide leads but do not themselves establish permission for model training.
Keep any permission records local and review the applicable terms before use.

## Private text segmentation

After extraction, put reviewed one-based PDF page ranges in the ignored
`data/private/segmentation_review.json`. Each key is an exact local PDF filename;
each entry has a `title`, `core_pages` as inclusive `[start, end]` pairs, and
optional `chapters` with `name`, `start`, and `end` page fields. Inspect the page
ranges before relying on them: a page with both endnotes and narrative cannot be
separated accurately using page ranges alone.

Run `python3 scripts/segment_text.py` to create fixed windows of 512 whitespace
tokens with 64 tokens of overlap. `--window` and `--overlap` change those values.
Whitespace tokens are only a local approximation; a model's tokenizer will count
tokens differently. The private output contains one JSONL file for each of train,
validation, and test, plus a manifest. Each chunk records its local source,
title, optional chapter, index, and source page span. Splits are assigned by
whole book to keep adjacent windows together. Identical windows are removed.
The output remains under `data/private/` and is marked as pending rights review.

The current pipeline handles text. Page images and layout regions would need a
separate vision workflow if the project later trains on visual PDF content.

Run `python3 scripts/shuffle_chunks.py` after segmentation to shuffle chunks
independently within train, validation, and test. The default seed is 42; pass
`--seed` to reproduce another ordering. The new private `shuffled-*` directory
contains the three JSONL files and a manifest with the seed, counts, and input
and output checksums. Original `chunk_index` and page references are preserved;
`shuffle_index` records each row's new position. Book-level split assignments
do not change.

Run `python3 scripts/split_chunks.py` to make the split an explicit step. It
groups chunks by whole book and chooses assignments closest to 70% train,
20% validation, and 10% test by chunk count. The private manifest records
both requested and actual ratios. With five unevenly sized books, exact ratios
may be impossible. Use `--train`, `--validation`, and `--test` for other ratios.
To shuffle the resulting dataset, pass its private directory to
`scripts/shuffle_chunks.py --input`.

## Private tokenization, step 1

Run `python3 scripts/tokenize_text.py` to convert the latest shuffled chunks
into numeric UTF-8 byte IDs. This reversible baseline is useful for inspecting
the conversion, but it is not compatible with a pretrained model's vocabulary.
For fine-tuning, rerun with `--tokenizer-path` pointing to a **local** directory
containing the exact target model's Hugging Face tokenizer. That mode requires
the optional `transformers` package and never downloads tokenizer files.
Each private `tokenized-*` run contains numeric JSONL files, a manifest, and
`report.md` with counts, ID lengths, and a small numeric sample. The script
does not pad or truncate chunks; batching is a separate step.
