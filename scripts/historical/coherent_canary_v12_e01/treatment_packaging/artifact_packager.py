#!/usr/bin/env python3
"""Deterministically package a JSON artifact as bounded gzip/base64 JSON chunks.

The source bytes are preserved exactly.  JSON parsing is only an input gate; the
parsed representation is never serialized back to disk.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOOL_NAME = "artifact_packager.py"
TOOL_VERSION = "1.0.0"
MANIFEST_SCHEMA = "lossless-json-gzip-base64-package/v1"
CHUNK_SCHEMA = "lossless-json-gzip-base64-chunk/v1"
MANIFEST_NAME = "manifest.json"

# A 2,850,000-byte compressed segment base64-encodes to exactly 3,800,000
# characters.  The canonical JSON wrapper remains well below both the declared
# 3.9 MB package cap and the repository's 4,000,000-byte hard limit.
COMPRESSED_BYTES_PER_CHUNK = 2_850_000
MAX_CHUNK_FILE_BYTES = 3_900_000
ABSOLUTE_CHUNK_FILE_LIMIT = 4_000_000
IO_BLOCK_BYTES = 1024 * 1024
CHUNK_NAME_RE = re.compile(r"^chunk-[0-9]{6}\.json$")
CANONICAL_GZIP_HEADER = bytes.fromhex("1f8b08000000000002ff")


class PackageError(RuntimeError):
    """A package failed a format, integrity, or safety check."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(IO_BLOCK_BYTES), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def current_tool_sha256() -> str:
    return sha256_bytes(Path(__file__).resolve().read_bytes())


def validate_json_source(path: Path) -> None:
    # json.load accepts the JSON encodings defined by RFC 8259.  This gate can
    # require memory proportional to the parsed document, but never changes the
    # source bytes subsequently compressed.
    try:
        with path.open("rb") as handle:
            json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"source is not readable valid JSON: {exc}") from exc


def gzip_source_deterministically(source: Path, destination: Path) -> Tuple[str, int]:
    original_digest = hashlib.sha256()
    original_size = 0
    with source.open("rb") as src, destination.open("wb") as raw_out:
        # filename="" suppresses input-path metadata; mtime=0 freezes the only
        # time field.  GzipFile emits OS=255, unlike some gzip.compress versions.
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_out,
            mtime=0,
        ) as gz_out:
            for block in iter(lambda: src.read(IO_BLOCK_BYTES), b""):
                original_digest.update(block)
                original_size += len(block)
                gz_out.write(block)
    return original_digest.hexdigest(), original_size


def validate_canonical_gzip_header(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            header = handle.read(len(CANONICAL_GZIP_HEADER))
    except OSError as exc:
        raise PackageError(f"cannot inspect gzip header: {exc}") from exc
    if header != CANONICAL_GZIP_HEADER:
        raise PackageError(
            "gzip header is not canonical (required: no optional fields, "
            "mtime=0, level-9 XFL, OS=unknown)"
        )


def _chunk_record(
    *, index: int, name: str, offset: int, segment: bytes, package_dir: Path
) -> Dict[str, Any]:
    segment_sha = sha256_bytes(segment)
    chunk_object = {
        "compressed_offset_bytes": offset,
        "compressed_segment_sha256": segment_sha,
        "compressed_segment_size_bytes": len(segment),
        "index": index,
        "name": name,
        "payload": base64.b64encode(segment).decode("ascii"),
        "payload_encoding": "base64",
        "schema": CHUNK_SCHEMA,
    }
    encoded = canonical_json_bytes(chunk_object)
    if len(encoded) > MAX_CHUNK_FILE_BYTES:
        raise PackageError(
            f"internal error: {name} is {len(encoded)} bytes, above declared cap "
            f"{MAX_CHUNK_FILE_BYTES}"
        )
    if len(encoded) >= ABSOLUTE_CHUNK_FILE_LIMIT:
        raise PackageError(
            f"internal error: {name} is not below {ABSOLUTE_CHUNK_FILE_LIMIT} bytes"
        )
    (package_dir / name).write_bytes(encoded)
    return {
        "compressed_offset_bytes": offset,
        "compressed_segment_sha256": segment_sha,
        "compressed_segment_size_bytes": len(segment),
        "file_sha256": sha256_bytes(encoded),
        "file_size_bytes": len(encoded),
        "index": index,
        "name": name,
    }


def pack(source: Path, output_dir: Path, *, validate_json: bool = True) -> Dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise PackageError(f"source is not a regular file: {source}")
    if output_dir.exists():
        raise PackageError(f"refusing to overwrite existing output path: {output_dir}")
    if not output_dir.parent.is_dir():
        raise PackageError(f"output parent does not exist: {output_dir.parent}")
    if validate_json:
        validate_json_source(source)

    with tempfile.TemporaryDirectory(
        prefix=".artifact-pack-build-", dir=str(output_dir.parent)
    ) as build_name:
        build_dir = Path(build_name)
        compressed_path = build_dir / "source.json.gz"
        package_dir = build_dir / "package"
        package_dir.mkdir()

        original_sha, original_size = gzip_source_deterministically(
            source, compressed_path
        )
        validate_canonical_gzip_header(compressed_path)
        compressed_sha, compressed_size = sha256_file(compressed_path)

        chunks: List[Dict[str, Any]] = []
        offset = 0
        index = 1
        with compressed_path.open("rb") as compressed:
            while True:
                segment = compressed.read(COMPRESSED_BYTES_PER_CHUNK)
                if not segment:
                    break
                name = f"chunk-{index:06d}.json"
                chunks.append(
                    _chunk_record(
                        index=index,
                        name=name,
                        offset=offset,
                        segment=segment,
                        package_dir=package_dir,
                    )
                )
                offset += len(segment)
                index += 1

        if not chunks:
            raise PackageError("internal error: gzip stream unexpectedly produced no chunks")
        if offset != compressed_size:
            raise PackageError(
                f"internal error: chunked {offset} of {compressed_size} compressed bytes"
            )

        manifest = {
            "chunk_count": len(chunks),
            "chunks": chunks,
            "compression": {
                "compressed_sha256": compressed_sha,
                "compressed_size_bytes": compressed_size,
                "format": "gzip",
                "level": 9,
                "mtime_unix": 0,
                "stored_filename": "",
            },
            "encoding": {
                "absolute_chunk_file_limit_bytes_exclusive": ABSOLUTE_CHUNK_FILE_LIMIT,
                "compressed_bytes_per_chunk": COMPRESSED_BYTES_PER_CHUNK,
                "maximum_chunk_file_bytes": MAX_CHUNK_FILE_BYTES,
                "outer": "canonical-json-utf8",
                "payload": "base64-rfc4648",
            },
            "original": {
                "format": "json",
                "json_validated": validate_json,
                "sha256": original_sha,
                "size_bytes": original_size,
            },
            "schema": MANIFEST_SCHEMA,
            "runtime": {
                "python": platform.python_version(),
                "zlib_compile": zlib.ZLIB_VERSION,
                "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            },
            "tool": {
                "name": TOOL_NAME,
                "sha256": current_tool_sha256(),
                "version": TOOL_VERSION,
            },
        }
        (package_dir / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest))

        # The destination does not exist by contract.  A same-filesystem rename
        # makes a completed package appear as a unit.
        package_dir.rename(output_dir)
        return manifest


def _expect_dict(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise PackageError(f"{where} must be an object")
    return value


def _expect_list(value: Any, where: str) -> List[Any]:
    if not isinstance(value, list):
        raise PackageError(f"{where} must be an array")
    return value


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise PackageError(f"{where} must be a string")
    return value


def _expect_int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PackageError(f"{where} must be an integer >= {minimum}")
    return value


def _expect_sha(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise PackageError(f"{where} must be a lowercase SHA-256 hex digest")
    return text


def _load_manifest(package_dir: Path) -> Dict[str, Any]:
    path = package_dir / MANIFEST_NAME
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read manifest {path}: {exc}") from exc
    manifest = _expect_dict(manifest, "manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PackageError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    return manifest


def _validate_manifest_header(
    manifest: Dict[str, Any]
) -> Tuple[List[Any], int, str, int, str, int]:
    chunks = _expect_list(manifest.get("chunks"), "manifest.chunks")
    count = _expect_int(manifest.get("chunk_count"), "manifest.chunk_count", minimum=1)
    if count != len(chunks):
        raise PackageError(f"chunk_count={count}, but manifest lists {len(chunks)} chunks")

    encoding = _expect_dict(manifest.get("encoding"), "manifest.encoding")
    declared_cap = _expect_int(
        encoding.get("maximum_chunk_file_bytes"),
        "manifest.encoding.maximum_chunk_file_bytes",
        minimum=1,
    )
    absolute_limit = _expect_int(
        encoding.get("absolute_chunk_file_limit_bytes_exclusive"),
        "manifest.encoding.absolute_chunk_file_limit_bytes_exclusive",
        minimum=2,
    )
    if declared_cap >= absolute_limit or absolute_limit > ABSOLUTE_CHUNK_FILE_LIMIT:
        raise PackageError("manifest chunk limits do not preserve the <4,000,000-byte invariant")
    if encoding.get("payload") != "base64-rfc4648":
        raise PackageError("unsupported payload encoding")

    compression = _expect_dict(manifest.get("compression"), "manifest.compression")
    if compression.get("format") != "gzip" or compression.get("mtime_unix") != 0:
        raise PackageError("manifest does not declare gzip with mtime=0")
    compressed_size = _expect_int(
        compression.get("compressed_size_bytes"),
        "manifest.compression.compressed_size_bytes",
        minimum=1,
    )
    compressed_sha = _expect_sha(
        compression.get("compressed_sha256"),
        "manifest.compression.compressed_sha256",
    )

    original = _expect_dict(manifest.get("original"), "manifest.original")
    if original.get("format") != "json":
        raise PackageError("manifest original format is not json")
    original_size = _expect_int(
        original.get("size_bytes"), "manifest.original.size_bytes", minimum=1
    )
    original_sha = _expect_sha(original.get("sha256"), "manifest.original.sha256")
    return chunks, declared_cap, compressed_sha, compressed_size, original_sha, original_size


def _validate_chunk(
    package_dir: Path,
    listed: Any,
    *,
    expected_index: int,
    expected_offset: int,
    declared_cap: int,
) -> Tuple[bytes, int]:
    record = _expect_dict(listed, f"manifest.chunks[{expected_index - 1}]")
    index = _expect_int(record.get("index"), f"chunk {expected_index} index", minimum=1)
    if index != expected_index:
        raise PackageError(f"expected chunk index {expected_index}, found {index}")
    name = _expect_str(record.get("name"), f"chunk {index} name")
    if not CHUNK_NAME_RE.fullmatch(name) or Path(name).name != name:
        raise PackageError(f"unsafe or noncanonical chunk name: {name!r}")
    offset = _expect_int(record.get("compressed_offset_bytes"), f"{name} offset")
    if offset != expected_offset:
        raise PackageError(f"{name} offset {offset} != expected {expected_offset}")
    expected_file_size = _expect_int(record.get("file_size_bytes"), f"{name} file size")
    expected_file_sha = _expect_sha(record.get("file_sha256"), f"{name} file sha256")
    expected_segment_size = _expect_int(
        record.get("compressed_segment_size_bytes"), f"{name} segment size", minimum=1
    )
    expected_segment_sha = _expect_sha(
        record.get("compressed_segment_sha256"), f"{name} segment sha256"
    )

    path = package_dir / name
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PackageError(f"cannot read {path}: {exc}") from exc
    if len(raw) != expected_file_size:
        raise PackageError(f"{name} size mismatch: {len(raw)} != {expected_file_size}")
    if len(raw) > declared_cap or len(raw) >= ABSOLUTE_CHUNK_FILE_LIMIT:
        raise PackageError(f"{name} violates the package chunk-file size limit")
    if sha256_bytes(raw) != expected_file_sha:
        raise PackageError(f"{name} file SHA-256 mismatch")

    try:
        chunk = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"{name} is not valid JSON: {exc}") from exc
    chunk = _expect_dict(chunk, name)
    mirrored = {
        "schema": CHUNK_SCHEMA,
        "index": index,
        "name": name,
        "compressed_offset_bytes": offset,
        "compressed_segment_size_bytes": expected_segment_size,
        "compressed_segment_sha256": expected_segment_sha,
        "payload_encoding": "base64",
    }
    for key, expected in mirrored.items():
        if chunk.get(key) != expected:
            raise PackageError(f"{name} field {key!r} does not match its manifest record")
    payload = _expect_str(chunk.get("payload"), f"{name}.payload")
    try:
        segment = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PackageError(f"{name} has invalid base64 payload: {exc}") from exc
    if len(segment) != expected_segment_size:
        raise PackageError(f"{name} decoded segment size mismatch")
    if sha256_bytes(segment) != expected_segment_sha:
        raise PackageError(f"{name} decoded segment SHA-256 mismatch")
    return segment, offset + len(segment)


def verify_and_reconstruct(
    package_dir: Path, output: Optional[Path] = None
) -> Dict[str, Any]:
    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise PackageError(f"package is not a directory: {package_dir}")
    if output is not None:
        output = output.resolve()
        if output.exists():
            raise PackageError(f"refusing to overwrite reconstruction output: {output}")
        if not output.parent.is_dir():
            raise PackageError(f"reconstruction parent does not exist: {output.parent}")

    manifest = _load_manifest(package_dir)
    (
        chunks,
        declared_cap,
        expected_compressed_sha,
        expected_compressed_size,
        expected_original_sha,
        expected_original_size,
    ) = _validate_manifest_header(manifest)

    expected_names = {
        _expect_str(_expect_dict(item, "chunk record").get("name"), "chunk name")
        for item in chunks
    }
    actual_names = {path.name for path in package_dir.glob("chunk-*.json")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise PackageError(f"chunk inventory mismatch; missing={missing}, extra={extra}")

    tool = _expect_dict(manifest.get("tool"), "manifest.tool")
    packaged_tool_sha = _expect_sha(tool.get("sha256"), "manifest.tool.sha256")

    with tempfile.TemporaryDirectory(
        prefix=".artifact-verify-", dir=str(package_dir.parent)
    ) as verify_name:
        verify_dir = Path(verify_name)
        compressed_path = verify_dir / "reconstructed.json.gz"
        compressed_digest = hashlib.sha256()
        compressed_size = 0
        offset = 0
        with compressed_path.open("wb") as compressed_out:
            for index, listed in enumerate(chunks, start=1):
                segment, offset = _validate_chunk(
                    package_dir,
                    listed,
                    expected_index=index,
                    expected_offset=offset,
                    declared_cap=declared_cap,
                )
                compressed_out.write(segment)
                compressed_digest.update(segment)
                compressed_size += len(segment)

        if compressed_size != expected_compressed_size:
            raise PackageError(
                f"compressed size mismatch: {compressed_size} != {expected_compressed_size}"
            )
        if compressed_digest.hexdigest() != expected_compressed_sha:
            raise PackageError("compressed stream SHA-256 mismatch")
        validate_canonical_gzip_header(compressed_path)

        reconstructed_temp = verify_dir / "reconstructed.json"
        original_digest = hashlib.sha256()
        original_size = 0
        try:
            with gzip.open(compressed_path, "rb") as gz_in, reconstructed_temp.open(
                "wb"
            ) as reconstructed:
                while True:
                    block = gz_in.read(IO_BLOCK_BYTES)
                    if not block:
                        break
                    original_size += len(block)
                    if original_size > expected_original_size:
                        raise PackageError(
                            "decompressed stream exceeds manifest original size"
                        )
                    original_digest.update(block)
                    reconstructed.write(block)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise PackageError(f"gzip reconstruction failed: {exc}") from exc

        if original_size != expected_original_size:
            raise PackageError(
                f"original size mismatch: {original_size} != {expected_original_size}"
            )
        if original_digest.hexdigest() != expected_original_sha:
            raise PackageError("original artifact SHA-256 mismatch")

        if output is not None:
            reconstructed_temp.rename(output)

    current_sha = current_tool_sha256()
    return {
        "chunk_count": len(chunks),
        "compressed_sha256": expected_compressed_sha,
        "compressed_size_bytes": expected_compressed_size,
        "current_tool_sha256": current_sha,
        "original_sha256": expected_original_sha,
        "original_size_bytes": expected_original_size,
        "output": str(output) if output is not None else None,
        "package": str(package_dir),
        "packaging_tool_matches_current": packaged_tool_sha == current_sha,
        "packaging_tool_sha256": packaged_tool_sha,
        "status": "VERIFIED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="create a new package")
    pack_parser.add_argument("source", type=Path)
    pack_parser.add_argument("output_dir", type=Path)
    pack_parser.add_argument(
        "--skip-json-validation",
        action="store_true",
        help="package bytes without first parsing JSON (recorded in manifest)",
    )

    verify_parser = subparsers.add_parser(
        "verify", help="verify a package and optionally reconstruct its source"
    )
    verify_parser.add_argument("package_dir", type=Path)
    verify_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "pack":
            manifest = pack(
                args.source,
                args.output_dir,
                validate_json=not args.skip_json_validation,
            )
            report = {
                "chunk_count": manifest["chunk_count"],
                "compressed_sha256": manifest["compression"]["compressed_sha256"],
                "compressed_size_bytes": manifest["compression"]["compressed_size_bytes"],
                "manifest": str(args.output_dir.resolve() / MANIFEST_NAME),
                "original_sha256": manifest["original"]["sha256"],
                "original_size_bytes": manifest["original"]["size_bytes"],
                "status": "PACKAGED",
                "tool_sha256": manifest["tool"]["sha256"],
            }
        else:
            report = verify_and_reconstruct(args.package_dir, args.output)
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0
    except PackageError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
