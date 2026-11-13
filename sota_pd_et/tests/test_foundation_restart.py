"""Restart a whole incomplete model and preserve completed model bytes."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sota_pd_et import compare_foundations as comparison
from sota_pd_et.foundations.runtime import model_runtime_settings
from sota_pd_et.tests.test_foundation_protocol import fixture


class FoundationRestartTests(unittest.TestCase):
    def test_restart_reuses_settings_and_skips_completed_models_without_rewriting_them(self):
        cohort = fixture()
        score = np.where(cohort.y, 1., -1.)
        summary = comparison._summary(cohort.y, score[None], [comparison.metrics(cohort.y, score)])
        calls = []

        def failed_evaluation(data, plan, args, destination, options):
            calls.append(options.model)
            if options.model == "tirex":
                (destination / "partial_training.txt").write_text("failed first model attempt")
                raise RuntimeError("Nonfinite backbone gradients.")
            comparison.write_json(destination / "summary.json", summary)
            (destination / "trained_weights.txt").write_text("completed " + options.model)
            return summary

        def successful_evaluation(data, plan, args, destination, options):
            calls.append(options.model)
            self.assertEqual(args.max_epochs, 7)
            self.assertEqual(args.learning_rates, [.00002])
            self.assertFalse((destination / "partial_training.txt").exists())
            self.assertEqual(options.channel_batch_size, 32 if options.model == "tirex" else 16)
            self.assertFalse(options.gradient_checkpointing)
            comparison.write_json(destination / "summary.json", summary)
            return summary

        with tempfile.TemporaryDirectory() as temporary:
            args = comparison.parse_args([
                "--models", "tspulse", "utica", "tirex", "moment", "--mode", "finetune", "--device", "cpu",
                "--outer-folds", "2", "--inner-folds", "2", "--max-epochs", "7", "--learning-rates", ".00002",
                "--output", str(Path(temporary) / "results"), "--cache-dir", str(Path(temporary) / "cache"), "--no-progress",
            ])
            with mock.patch.object(comparison, "load_cohort", return_value=cohort), \
                 mock.patch.object(comparison, "evaluate_finetuned", side_effect=failed_evaluation), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                comparison.run(args)
            output = next((Path(temporary) / "results").iterdir()).resolve()
            before = json.loads((output / "manifest.json").read_text())
            protected = {p: p.read_bytes() for name in ("tspulse", "utica") for p in (output / name).rglob("*") if p.is_file()}
            split_files = {p: p.read_bytes() for p in (output / "splits.json", output / "finetune_splits.json", output / "cohort.csv")}
            table = (output / "comparison.csv").read_text().splitlines()
            restart = comparison.parse_args(["--restart-incomplete", str(output), "--no-progress",
                                             "--tirex-channel-batch-size", "32",
                                             "--tirex-gradient-checkpointing", "off",
                                             "--model-channel-batch-size", "moment=16",
                                             "--model-gradient-checkpointing", "moment=off"])
            with mock.patch.object(comparison, "load_cohort", return_value=cohort), \
                 mock.patch.object(comparison, "evaluate_finetuned", side_effect=successful_evaluation), \
                 contextlib.redirect_stdout(io.StringIO()):
                actual = comparison.run(restart)
            self.assertEqual(actual, output)
            self.assertEqual(calls, ["tspulse", "utica", "tirex", "tirex", "moment"])
            for path, content in {**protected, **split_files}.items():
                self.assertEqual(path.read_bytes(), content)
            after = json.loads((output / "manifest.json").read_text())
            self.assertEqual(after["status"], "complete")
            self.assertEqual(after["run_id"], before["run_id"])
            self.assertEqual(after["runs"][:2], before["runs"][:2])
            self.assertEqual(after["restart_history"][0]["tirex_runtime_overrides"],
                             {"channel_batch_size": 32, "gradient_checkpointing": "off"})
            self.assertEqual(after["restart_history"][0]["model_runtime_overrides"],
                             {"channel_batch_size": {"tirex": 32, "moment": 16},
                              "gradient_checkpointing": {"tirex": "off", "moment": "off"}})
            previous = after["runs"][2]["previous_attempts"][0]
            self.assertEqual((output / previous["archived_directory"] / "partial_training.txt").read_text(), "failed first model attempt")
            backup = json.loads((output / after["restart_history"][0]["previous_manifest"]).read_text())
            self.assertEqual(backup, before)
            self.assertEqual((output / "comparison.csv").read_text().splitlines()[:3], table)
            with self.assertRaisesRegex(ValueError, "running or completed|active or completed"):
                comparison.parse_args(["--restart-incomplete", str(output)])

    def test_restart_disallows_silent_training_setting_overrides(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            comparison.parse_args(["--restart-incomplete", "unused", "--models", "tirex"])

    def test_each_model_runtime_override_is_independent_and_global_defaults_remain_available(self):
        names = list(comparison.SPECS)
        flags = ["--channel-batch-size", "7", "--gradient-checkpointing"]
        for index, name in enumerate(names):
            flags.extend(["--model-channel-batch-size", f"{name}={index + 8}",
                          "--model-gradient-checkpointing", f"{name}={'on' if index % 2 else 'off'}"])
        args = comparison.parse_args(flags)
        for index, name in enumerate(names):
            self.assertEqual(model_runtime_settings(args, name), (index + 8, bool(index % 2)))
        args = comparison.parse_args(["--channel-batch-size", "7", "--gradient-checkpointing",
                                      "--model-channel-batch-size", "moment=16",
                                      "--model-gradient-checkpointing", "moment=off"])
        self.assertEqual(model_runtime_settings(args, "moment"), (16, False))
        self.assertEqual(model_runtime_settings(args, "tivit"), (7, True))

    def test_invalid_duplicate_and_conflicting_runtime_settings_are_rejected(self):
        invalid = (["--model-channel-batch-size", "moment=0"],
                   ["--model-channel-batch-size", "unknown=16"],
                   ["--model-channel-batch-size", "moment=2.5"],
                   ["--model-gradient-checkpointing", "tivit=yes"],
                   ["--model-channel-batch-size", "moment=16", "--model-channel-batch-size", "moment=32"],
                   ["--tirex-channel-batch-size", "32", "--model-channel-batch-size", "tirex=16"])
        for flags in invalid:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                comparison.parse_args(flags)

    def test_restart_merges_per_model_settings_and_accepts_prior_tirex_only_manifests(self):
        saved = comparison.parse_args(["--models", "tirex", "moment", "tivit", "--mode", "finetune",
                                        "--tirex-channel-batch-size", "32", "--tirex-gradient-checkpointing", "on"])
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(saved).items()}
        config.pop("model_channel_batch_size")
        config.pop("model_gradient_checkpointing")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            manifest = {"status": "interrupted", "config": config,
                        "protocol": {"adapter_version": 3, "finetune_protocol": comparison.FINETUNE_PROTOCOL}}
            comparison.write_json(path / "manifest.json", manifest)
            restored = comparison.parse_args(["--restart-incomplete", str(path)])
            self.assertEqual(model_runtime_settings(restored, "tirex"), (32, True))
            changed = comparison.parse_args(["--restart-incomplete", str(path),
                                             "--model-channel-batch-size", "tirex=64",
                                             "--model-gradient-checkpointing", "tirex=off"])
            self.assertEqual(model_runtime_settings(changed, "tirex"), (64, False))
            config["model_channel_batch_size"] = {"moment": 16, "tivit": 4}
            comparison.write_json(path / "manifest.json", manifest)
            changed = comparison.parse_args(["--restart-incomplete", str(path),
                                             "--model-channel-batch-size", "moment=32"])
            self.assertEqual(model_runtime_settings(changed, "moment"), (32, None))
            self.assertEqual(model_runtime_settings(changed, "tivit"), (4, None))


if __name__ == "__main__":
    unittest.main()
