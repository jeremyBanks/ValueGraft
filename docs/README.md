# Documentation Archive Naming

This directory is an archive for research notes, drafts, reviews, and small
reports that are no longer meant to sit at the repository root.

Archive files should use:

```text
YYYYMMDDHHMMSS-short-kebab-case-title.md
```

The timestamp is UTC. It should come from the file's original creation time in
git, not from the time it was moved into `docs/`. When possible, trace renames
back to the first commit that added the document:

```bash
git log --follow --date=iso-strict -- path/to/file.md
```

The title portion should be lowercase kebab case: words separated with hyphens,
no spaces, no underscores, and a `.md` extension.

The usual workflow is to drop a markdown file into `docs/`, then run:

```bash
python3 scripts/normalize_docs_archive_names.py
```

The script normalizes every markdown file in this directory except `README.md`.
For tracked files, it uses the first git commit timestamp in UTC, following
renames. For brand-new untracked files, it falls back to filesystem creation time
when available, then modification time.

`README.md` is the only intentional unprefixed markdown file in this directory.
