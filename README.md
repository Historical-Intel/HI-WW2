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

The source PDFs were previously committed to Git. Ignoring cleaned output does
not make those existing source files private or remove them from Git history.
