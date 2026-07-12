#!/usr/bin/env python3
"""Unit tests for artifact_packager.py (stdlib only)."""

import json
import gzip
import tempfile
import unittest
from pathlib import Path

import artifact_packager as ap


class ArtifactPackagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="unit-", dir=str(Path(__file__).resolve().parent)
        )
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_source(self, name="source.json"):
        source = self.root / name
        source.write_bytes(
            b'{\n  "unicode": "\\u2603", "ordered": [3, 2, 1], "padding": "xyz"\n}\n'
        )
        return source

    def test_round_trip_is_byte_exact(self):
        source = self.write_source()
        package = self.root / "package"
        output = self.root / "output.json"
        manifest = ap.pack(source, package)
        report = ap.verify_and_reconstruct(package, output)
        self.assertEqual(source.read_bytes(), output.read_bytes())
        self.assertEqual(manifest["original"]["sha256"], report["original_sha256"])
        self.assertEqual("VERIFIED", report["status"])

    def test_two_packages_are_byte_for_byte_deterministic(self):
        source = self.write_source()
        renamed_source = self.write_source("same-bytes-different-name.json")
        first = self.root / "first"
        second = self.root / "second"
        ap.pack(source, first)
        ap.pack(renamed_source, second)
        first_files = sorted(path.name for path in first.iterdir())
        second_files = sorted(path.name for path in second.iterdir())
        self.assertEqual(first_files, second_files)
        for name in first_files:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_invalid_json_is_rejected_without_output(self):
        source = self.root / "bad.json"
        source.write_bytes(b'{"not": valid}')
        package = self.root / "package"
        with self.assertRaises(ap.PackageError):
            ap.pack(source, package)
        self.assertFalse(package.exists())

    def test_chunk_corruption_is_rejected(self):
        source = self.write_source()
        package = self.root / "package"
        ap.pack(source, package)
        chunk_path = package / "chunk-000001.json"
        chunk = json.loads(chunk_path.read_bytes())
        payload = chunk["payload"]
        chunk["payload"] = ("A" if payload[0] != "A" else "B") + payload[1:]
        chunk_path.write_bytes(ap.canonical_json_bytes(chunk))
        with self.assertRaisesRegex(ap.PackageError, "size mismatch|SHA-256 mismatch"):
            ap.verify_and_reconstruct(package)

    def test_refuses_overwrite(self):
        source = self.write_source()
        package = self.root / "package"
        ap.pack(source, package)
        with self.assertRaisesRegex(ap.PackageError, "overwrite"):
            ap.pack(source, package)
        output = self.root / "output.json"
        output.write_text("already here")
        with self.assertRaisesRegex(ap.PackageError, "overwrite"):
            ap.verify_and_reconstruct(package, output)

    def test_nonzero_gzip_mtime_is_rejected(self):
        path = self.root / "noncanonical.gz"
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=1) as stream:
                stream.write(b"{}")
        with self.assertRaisesRegex(ap.PackageError, "header is not canonical"):
            ap.validate_canonical_gzip_header(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
