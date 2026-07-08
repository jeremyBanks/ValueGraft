# Documentation Archive Naming

This directory is an archive for research notes, drafts, reviews, and small
reports that are no longer meant to sit at the repository root.

Archive files should use:

```text
YYYY-MM-DD-HH-short-kebab-case-title.md
```

The prefix should come from the file's original creation time in git, not from
the time it was moved into `docs/`. When possible, trace renames back to the
first commit that added the document:

```bash
git log --follow --reverse --date=iso-strict -- path/to/file.md
```

Use the local date and hour from that first git commit for the prefix. The title
portion should be lowercase kebab case: words separated with hyphens, no spaces,
no underscores, and a `.md` extension.

`README.md` is the only intentional unprefixed markdown file in this directory.
