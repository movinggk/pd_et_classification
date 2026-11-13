"""Offline acquisition tests: synthetic HTTP responses, never real downloads."""

from __future__ import annotations

from contextlib import ExitStack
import csv
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError
from urllib.parse import unquote, urlsplit

from sota_pd_et import colab_download as download


def digest(content):
    return hashlib.sha256(content).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, content, url):
        super().__init__(content)
        self.url = url

    def geturl(self):
        return self.url


class FakeHTTP:
    def __init__(self, files, outcomes=None):
        self.files = files
        self.outcomes = outcomes or {}
        self.calls = []

    def __call__(self, request, timeout):
        assert timeout == download.TIMEOUT_SECONDS
        name = unquote(urlsplit(request.full_url).path.removeprefix(urlsplit(download.BASE_URL).path))
        self.calls.append(name)
        queued = self.outcomes.get(name)
        outcome = queued.pop(0) if queued else self.files[name]
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResponse(outcome, request.full_url)


def fixture_files():
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=["id", "condition", "label"])
    writer.writeheader()
    rows = [
        {"id": "004", "condition": "Parkinson's", "label": "1"},
        {"id": "066", "condition": "Essential Tremor", "label": "2"},
        {"id": "002", "condition": "Other Movement Disorders", "label": "2"},
    ]
    writer.writerows(rows)
    files = {
        "preprocessed/file_list.csv": text.getvalue().encode(),
        "scripts/run_preprocessing.py": b"# provenance only\n",
        "scripts/utils/data_handling.py": b"# provenance only\n",
    }
    for row in rows[:2]:
        files[f"patients/patient_{row['id']}.json"] = json.dumps(row).encode()
        files[f"preprocessed/movement/{row['id']}_ml.bin"] = b"\x00" * 16
    files["SHA256SUMS.txt"] = "".join(
        f"{digest(content)}  {name}\n" for name, content in sorted(files.items())
    ).encode()
    return files


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = fixture_files()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(download, "CHECKSUM_LIST_SHA256",
                                                  digest(self.files["SHA256SUMS.txt"])))
        self.stack.enter_context(mock.patch.object(download, "EXPECTED_CONDITION_COUNTS",
                                                  {"Parkinson's": 1, "Essential Tremor": 1}))
        self.stack.enter_context(mock.patch.object(download, "EXPECTED_TABLE_SIZE", 3))
        self.stack.enter_context(mock.patch.object(download, "MOVEMENT_BYTES", 16))
        self.stack.enter_context(mock.patch.object(download.time, "sleep"))
        self.stack.enter_context(mock.patch("builtins.print"))

    def test_only_selected_files_acquired_and_verified_cache_reused(self):
        http = FakeHTTP(self.files)
        with mock.patch.object(download, "urlopen", side_effect=http):
            first = download.download_dataset(self.root, workers=2)
            second = download.download_dataset(self.root, workers=2)
        self.assertEqual(set(http.calls), set(self.files))
        self.assertEqual(len(http.calls), 8)
        self.assertEqual(first["participant_ids"], ["004", "066"])
        self.assertEqual(first["downloaded_files"], 8)
        self.assertEqual(first["reused_files"], 0)
        self.assertEqual(second["downloaded_files"], 0)
        self.assertEqual(second["reused_files"], 8)
        self.assertFalse(list(self.root.rglob("*.part")))
        self.assertFalse((self.root / "download_manifest.json").exists())
        for name, content in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), content)

    def test_network_errors_truncated_reads_and_checksum_corruption_are_retried(self):
        name = "scripts/run_preprocessing.py"
        http = FakeHTTP(self.files, outcomes={name: [
            URLError("offline test"), IncompleteRead(b"partial", 100), b"corruption",
            self.files[name],
        ]})
        with mock.patch.object(download, "urlopen", side_effect=http):
            result = download._download_verified(self.root, name, digest(self.files[name]))
        self.assertFalse(result["reused"])
        self.assertEqual(http.calls, [name] * 4)
        self.assertEqual((self.root / name).read_bytes(), self.files[name])
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_failed_download_preserves_existing_file_and_cleans_temporary_files(self):
        name = "preprocessed/file_list.csv"
        target = self.root / name
        target.parent.mkdir()
        target.write_bytes(b"old invalid cache")
        http = FakeHTTP({name: b"bad hash"})
        with mock.patch.object(download, "urlopen", side_effect=http):
            with self.assertRaisesRegex(RuntimeError, "after 4 attempts"):
                download._download_verified(self.root, name, digest(self.files[name]))
        self.assertEqual(target.read_bytes(), b"old invalid cache")
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_corrupted_cache_is_replaced(self):
        name = "scripts/run_preprocessing.py"
        target = self.root / name
        target.parent.mkdir()
        target.write_bytes(b"corrupted local copy")
        http = FakeHTTP(self.files)
        with mock.patch.object(download, "urlopen", side_effect=http):
            result = download._download_verified(self.root, name, digest(self.files[name]))
        self.assertFalse(result["reused"])
        self.assertEqual(http.calls, [name])
        self.assertEqual(target.read_bytes(), self.files[name])

    def test_checksum_pin_rejects_wrong_list_before_table_download(self):
        http = FakeHTTP({"SHA256SUMS.txt": b"untrusted list"})
        with mock.patch.object(download, "urlopen", side_effect=http):
            with self.assertRaisesRegex(RuntimeError, "SHA256SUMS"):
                download.download_dataset(self.root)
        self.assertEqual(http.calls, ["SHA256SUMS.txt"] * 4)
        self.assertFalse((self.root / "SHA256SUMS.txt").exists())

    def test_all_table_ids_validated_including_excluded_condition(self):
        name = self.root / "table.csv"
        name.write_text("id,condition\n004,Parkinson's\n066,Essential Tremor\n../x,Other\n")
        with self.assertRaisesRegex(ValueError, "exactly three"):
            download._select_participants(name)
        name.write_text("id,condition\n004,Parkinson's\n066,Essential Tremor\n066,Other\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            download._select_participants(name)

    def test_condition_strings_not_numeric_labels_define_selection(self):
        name = self.root / "table.csv"
        name.write_bytes(self.files["preprocessed/file_list.csv"])
        self.assertEqual([r["id"] for r in download._select_participants(name)], ["004", "066"])
        name.write_text("id,condition\n004,Parkinson's\n066,Other\n002,Other\n")
        with self.assertRaisesRegex(ValueError, "condition counts"):
            download._select_participants(name)

    def test_traversals_urls_and_external_symlinks_are_rejected(self):
        for name in ["../x", "/absolute", "x/../y", "x//y", "x\\y", "https://host/x", "./x"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                download._destination(self.root, name)
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                download._destination(self.root, "escape/x")

    def test_malformed_and_duplicate_checksum_records_rejected(self):
        checksum = self.root / "checksum.txt"
        for content in ["not-a-checksum x", f"{'0' * 64} ../escape\n",
                        f"{'0' * 64} x\n{'1' * 64} x\n"]:
            checksum.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                download._read_checksums(checksum)

    def test_worker_limit_is_enforced(self):
        for workers in [0, 7, True, 1.5, "4"]:
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                download.download_dataset(self.root, workers=workers)


if __name__ == "__main__":
    unittest.main()
