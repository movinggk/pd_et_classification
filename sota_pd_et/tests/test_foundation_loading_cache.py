"""Cache only immutable metadata, while keeping each fit's weights independent."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from sota_pd_et.foundations import encoders
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class FoundationLoadingCacheTests(unittest.TestCase):
    def setUp(self):
        for function in (encoders._hash_file_version, encoders._hub_snapshot,
                         encoders._hub_file, encoders._package_versions):
            function.cache_clear()

    tearDown = setUp

    def test_unchanged_checkpoint_is_hashed_once_but_rewrites_invalidate(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.bin"
            path.write_bytes(b"first checkpoint")
            expected_first = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(encoders, "sha256_file", wraps=encoders.sha256_file) as hashing:
                first = encoders.checkpoint_sha256(path)
                second = encoders.checkpoint_sha256(path)
                self.assertEqual(first, expected_first)
                self.assertEqual(second, expected_first)
                self.assertEqual(hashing.call_count, 1)
                previous = path.stat()
                path.write_bytes(b"other checkpoint")  # Same size; contents changed.
                os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
                changed = encoders.checkpoint_sha256(path)
                self.assertEqual(changed, hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertNotEqual(changed, first)
                self.assertEqual(hashing.call_count, 2)
                self.assertEqual(encoders.checkpoint_sha256(path), changed)
                self.assertEqual(hashing.call_count, 2)

    def test_replacing_same_size_and_mtime_checkpoint_invalidates_inode_identity(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.bin"
            path.write_bytes(b"old bytes")
            first = encoders.checkpoint_sha256(path)
            previous = path.stat()
            replacement = path.with_suffix(".tmp")
            replacement.write_bytes(b"new bytes")
            os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            replacement.replace(path)
            self.assertEqual(path.stat().st_size, previous.st_size)
            self.assertEqual(path.stat().st_mtime_ns, previous.st_mtime_ns)
            second = encoders.checkpoint_sha256(path)
            self.assertNotEqual(first, second)
            self.assertEqual(second, hashlib.sha256(b"new bytes").hexdigest())

    def test_hub_snapshot_and_file_metadata_reuse_pinned_resolution(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            checkpoint = snapshot / "model.safetensors"
            checkpoint.write_bytes(b"fixture")
            cache = str(root / "cache")
            revision, alternate_revision = "a" * 40, "b" * 40
            with mock.patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
                first = encoders._hub_snapshot("publisher/model", revision, cache)
                second = encoders._hub_snapshot("publisher/model", revision, cache)
                self.assertEqual(first, snapshot)
                self.assertEqual(second, snapshot)
                self.assertTrue(second.is_dir())
                self.assertEqual(download.call_count, 1)
                self.assertEqual(download.call_args.kwargs["revision"], revision)
                self.assertEqual(download.call_args.kwargs["cache_dir"], cache)
                encoders._hub_snapshot("publisher/model", alternate_revision, cache)
                self.assertEqual(download.call_count, 2)
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)) as download:
                first = encoders._hub_file("publisher/model", "model.safetensors", revision, cache)
                second = encoders._hub_file("publisher/model", "model.safetensors", revision, cache)
                self.assertEqual(first, checkpoint)
                self.assertEqual(second, checkpoint)
                self.assertTrue(second.is_file())
                self.assertEqual(download.call_count, 1)
                self.assertEqual(download.call_args.kwargs["revision"], revision)
                self.assertEqual(download.call_args.kwargs["cache_dir"], cache)
                encoders._hub_file("publisher/model", "model.safetensors", alternate_revision, cache)
                self.assertEqual(download.call_count, 2)

    def test_package_version_results_are_independent_copies(self):
        with mock.patch.object(encoders, "version", side_effect=lambda name: "fixture-" + name) as metadata:
            first = encoders.package_versions()
            count = metadata.call_count
            first["torch"] = "mutated by caller"
            first["caller_only"] = True
            second = encoders.package_versions()
            self.assertIsNot(first, second)
            self.assertEqual(second["torch"], "fixture-torch")
            self.assertNotIn("caller_only", second)
            self.assertEqual(metadata.call_count, count)

    def test_deleted_snapshot_or_checkpoint_is_resolved_again(self):
        for deleted in ("directory", "weights"):
            with self.subTest(deleted=deleted), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                snapshot = root / "snapshot"
                cache = str(root / "cache")
                revision = "c" * 40

                def download_snapshot(**kwargs):
                    snapshot.mkdir(exist_ok=True)
                    (snapshot / "config.json").write_text("{}")
                    (snapshot / "model.safetensors").write_bytes(b"restored checkpoint")
                    return str(snapshot)

                encoders._hub_snapshot.cache_clear()
                with mock.patch("huggingface_hub.snapshot_download", side_effect=download_snapshot) as download:
                    first = encoders._hub_snapshot("publisher/model", revision, cache)
                    self.assertTrue((first / "model.safetensors").is_file())
                    if deleted == "directory":
                        shutil.rmtree(snapshot)
                    else:
                        (snapshot / "model.safetensors").unlink()
                    restored = encoders._hub_snapshot("publisher/model", revision, cache)
                    self.assertEqual(restored, snapshot)
                    self.assertTrue((restored / "config.json").is_file())
                    self.assertTrue((restored / "model.safetensors").is_file())
                    self.assertEqual(download.call_count, 2)

    def test_deleted_hub_file_is_resolved_again(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "model.ckpt"
            cache = str(root / "cache")
            revision = "d" * 40

            def download_file(*args, **kwargs):
                checkpoint.write_bytes(b"restored checkpoint")
                return str(checkpoint)

            with mock.patch("huggingface_hub.hf_hub_download", side_effect=download_file) as download:
                first = encoders._hub_file("publisher/model", "model.ckpt", revision, cache)
                self.assertTrue(first.is_file())
                first.unlink()
                restored = encoders._hub_file("publisher/model", "model.ckpt", revision, cache)
                self.assertEqual(restored, checkpoint)
                self.assertTrue(restored.is_file())
                self.assertEqual(download.call_count, 2)

    def test_model_specific_memory_and_checkpointing_defaults_respect_overrides(self):
        for name in encoders.SPECS:
            with self.subTest(model=name):
                expected_chunk = {"tspulse": 128, "chronos2": 24}.get(name, 8)
                self.assertEqual(encoders.resolve_channel_batch_size(name), expected_chunk)
                self.assertEqual(encoders.resolve_channel_batch_size(name, 3), 3)
                self.assertEqual(encoders.resolve_gradient_checkpointing(name, None), name != "tspulse")
                self.assertIs(encoders.resolve_gradient_checkpointing(name, True), True)
                self.assertIs(encoders.resolve_gradient_checkpointing(name, False), False)
                options = EncoderOptions(name)
                self.assertEqual(options.channel_batch_size, expected_chunk)
                self.assertEqual(options.gradient_checkpointing, name != "tspulse")
                explicit = EncoderOptions(name, channel_batch_size=3, gradient_checkpointing=True)
                self.assertEqual(explicit.channel_batch_size, 3)
                self.assertIs(explicit.gradient_checkpointing, True)

    def test_resolved_checkpoint_cache_never_reuses_trained_parameter_objects(self):
        torch.set_num_threads(2)
        options = EncoderOptions("chronos2", chronos_group_size=2)
        original = tiny_encoder(options).eval()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot = root / "snapshot"
            original.backbone.save_pretrained(snapshot)
            with mock.patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
                first = encoders.load_encoder(options, root / "cache")
                first_parameter = next(first.backbone.parameters())
                initial = first_parameter.detach().clone()
                with torch.no_grad():
                    first_parameter.add_(10.)
                second = encoders.load_encoder(options, root / "cache")
                second_parameter = next(second.backbone.parameters())
                self.assertEqual(download.call_count, 1)
                self.assertIsNot(first.backbone, second.backbone)
                self.assertNotEqual(first_parameter.data_ptr(), second_parameter.data_ptr())
                torch.testing.assert_close(second_parameter, initial, rtol=0, atol=0)
                self.assertFalse(torch.equal(first_parameter, second_parameter))


if __name__ == "__main__":
    unittest.main()
