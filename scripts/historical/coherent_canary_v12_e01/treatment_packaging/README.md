# Lossless JSON artifact packaging (scratch implementation)

`artifact_packager.py` preserves the source file's bytes exactly while splitting
its deterministic `gzip(mtime=0)` stream across canonical JSON files containing
RFC 4648 base64.  Each chunk file is capped at 3,900,000 bytes and is asserted
to be strictly smaller than 4,000,000 bytes.

```sh
python3 artifact_packager.py pack source.json package-directory
python3 artifact_packager.py verify package-directory --output reconstructed.json
```

The verifier checks the ordered inventory, every chunk file hash and size, every
decoded compressed-segment hash and size, the reconstructed gzip hash and size,
the canonical zero-mtime gzip header, gzip CRC/format, and the original byte hash
and size before publishing an output.
It refuses to overwrite a package or reconstruction.

The manifest records the packager source hash.  A later version of the tool may
verify an older package: a tool-hash mismatch is reported, not treated as data
corruption, because all byte-level integrity fields are independently checked.
Input paths and filenames are intentionally omitted, so identical source bytes
under different names produce identical package files on the same runtime.
The Python and compile/runtime zlib versions are also recorded.  The gzip header
is fully canonical, but stdlib does not promise identical DEFLATE choices across
different zlib releases; byte-for-byte package determinism is therefore claimed
for the same tool and recorded runtime, while lossless verification is portable.

Input JSON validation uses Python's standard-library parser and therefore can
need memory proportional to the parsed document.  `--skip-json-validation`
exists for exceptionally large, already-validated artifacts and is explicitly
recorded as `json_validated: false`; compression and verification remain
streaming and byte-exact.
