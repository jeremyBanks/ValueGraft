# AGENTS.md — docs archive

This directory is an archive for research notes, drafts, reviews, and small reports.
Archive notes should be Markdown files.

Markdown archive files must use UTC, sortable, kebab-case names:

```text
YYYYMMDDHHMMSS-short-kebab-case-title.md
```

The timestamp should be the document's original creation time, traced through git
renames when possible, not the time it was moved into `docs/`.

```bash
python3 scripts/normalize_docs_archive_names.py
```

Usual workflow: drop a note into `docs/`, run the normalizer, then commit the
resulting rename. The normalizer accepts `.md` and markdown-like `.txt` notes, and
outputs `.md`. `AGENTS.md` is the intentional unprefixed instruction file in this
directory.
