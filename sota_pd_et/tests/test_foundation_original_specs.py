"""Compare adapter outputs with the authors' public library entry points.

The models are small actual upstream architectures, not mock encoders. These
checks verify feature contracts, not the authors' downstream training recipes.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
import torch
from torch.nn import functional as F

from sota_pd_et import predict_foundation
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class FoundationOriginalSpecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.values = torch.randn(2, 4, 64, generator=torch.Generator().manual_seed(628))

    def encoder(self, name):
        torch.manual_seed(159)
        return tiny_encoder(EncoderOptions(name, channel_batch_size=8, chronos_group_size=2)).eval()

    def test_mantis_and_utica_match_author_trainer_channel_concatenation(self):
        from mantis.trainer import MantisTrainer

        for name in ("utica", "mantis_v2"):
            with self.subTest(model=name):
                encoder = self.encoder(name)
                self.assertEqual(encoder.backbone.return_transf_layer, 2)
                self.assertEqual(encoder.backbone.output_token, "combined")
                trainer = MantisTrainer(device="cpu", network=encoder.backbone)
                resized = F.interpolate(self.values, size=512, mode="linear", align_corners=False)
                expected = trainer.transform(resized, batch_size=2)
                with torch.no_grad():
                    actual = encoder(self.values).numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_moment_keeps_native_revin_and_masked_embedding_path(self):
        encoder = self.encoder("moment")
        batch, channels, _ = self.values.shape
        # Native short-series sizing keeps observations and masks left padding.
        padding = 512 - self.values.shape[-1]
        resized = F.pad(self.values, (padding, 0))
        mask = torch.ones(batch * channels, 512)
        mask[:, :padding] = 0
        with torch.no_grad():
            expected = encoder.backbone(
                x_enc=resized.reshape(batch * channels, 1, 512),
                input_mask=mask, reduction="mean",
            ).embeddings.reshape(batch, -1)
            actual = encoder(self.values)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_tspulse_matches_native_unmasked_backbone_before_custom_pooling(self):
        encoder = self.encoder("tspulse")
        batch, channels, _ = self.values.shape
        resized = F.interpolate(self.values, 512, mode="linear", align_corners=False)
        with torch.no_grad():
            hidden = encoder.backbone(
                past_values=resized.reshape(batch * channels, 512, 1),
                enable_masking=False, return_dict=True,
            ).last_hidden_state
            registers = encoder.backbone.config.patch_register_tokens
            patches = (hidden.shape[2] - registers) // 2
            expected = torch.cat((hidden[:, 0, :patches].mean(1),
                                  hidden[:, 0, patches:2 * patches].mean(1),
                                  hidden[:, 0, -registers:].mean(1)), -1).reshape(batch, -1)
            actual = encoder(self.values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_chronos_group_features_match_native_pipeline_context_tokens(self):
        from chronos.chronos2 import Chronos2Pipeline

        encoder = self.encoder("chronos2")
        groups = self.values.reshape(-1, 2, self.values.shape[-1])
        pipeline = Chronos2Pipeline(encoder.backbone)
        embeddings, _ = pipeline.embed(groups, batch_size=8)
        # This checkpoint uses one REG and one masked future token. The
        # comparison deliberately pools context tokens only.
        expected = torch.stack([item[:, :-2].mean(1) for item in embeddings]).reshape(2, -1)
        with torch.no_grad():
            actual = encoder(self.values)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_tirex_audited_features_match_complete_native_embedding_wrapper(self):
        from tirex.models.embedding import TiRexEmbedding

        encoder = self.encoder("tirex")
        # Use the original public embedding wrapper with the identical tiny
        # model; only its checkpoint download is replaced.
        with mock.patch("tirex.models.embedding.load_model", return_value=encoder.backbone):
            original = TiRexEmbedding(device="cpu", data_augmentation=False, batch_size=2)
        for length in (51, 64, 80):
            with self.subTest(length=length):
                values = torch.randn(2, 4, length, generator=torch.Generator().manual_seed(length))
                expected = original(values)
                with torch.no_grad():
                    actual = encoder(values)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)

    def test_tivit_audited_features_match_author_wrapper_and_participant_l2(self):
        encoder = self.encoder("tivit")
        with torch.no_grad():
            images = encoder.backbone.ts2image_transformation(
                self.values.reshape(-1, 64, 1), patch_size=math.isqrt(64), stride=encoder.backbone.stride,
            )
            hidden = encoder.backbone.forward_vit(images)
            raw = encoder.backbone.aggregate_hidden_representations(
                hidden, encoder.backbone.aggregation,
            ).reshape(2, -1).numpy()
            expected = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
            actual = encoder(self.values).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), np.ones(2), rtol=1e-6)

    def test_vqshape_tokens_match_native_tokenize_representation(self):
        encoder = self.encoder("vqshape")
        resized = F.interpolate(self.values, 512, mode="linear", align_corners=False)
        with torch.no_grad():
            tokens, _ = encoder.backbone(resized.reshape(-1, 512), mode="tokenize")
            expected = tokens["token"].reshape(2, -1)
            actual = encoder(self.values)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_unsupported_frozen_artifacts_are_rejected_before_loading_weights(self):
        X = self.values.numpy()
        for name in ("tirex", "tivit", "moment"):
            for version in (None, 1, 2):
                with self.subTest(model=name, version=version):
                    options = asdict(EncoderOptions(name))
                    if version is None:
                        options.pop("adapter_version")
                    else:
                        options["adapter_version"] = version
                    bundle = {
                        "format_version": 1, "model_type": "foundation_comparison", "mode": "frozen",
                        "encoder_options": options, "input_shape": [4, 64],
                    }
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / "unsupported_model.joblib"
                        joblib.dump(bundle, path)
                        with mock.patch.object(predict_foundation, "load_encoder") as loader, \
                             self.assertRaisesRegex(ValueError, "adapter version 3"):
                            predict_foundation.predict(path, X)
                        loader.assert_not_called()

    def test_adapter_policy_is_fixed_and_cli_has_no_version_selector(self):
        from sota_pd_et.compare_foundations import parse_args

        self.assertFalse(hasattr(parse_args([]), "adapter_version"))
        self.assertEqual(EncoderOptions("tspulse").adapter_version, 3)
        for version in (1, 2, 3):
            with self.subTest(version=version), self.assertRaises(TypeError):
                EncoderOptions("tspulse", adapter_version=version)
            with self.subTest(version=version), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                parse_args(["--adapter-version", str(version)])


if __name__ == "__main__":
    unittest.main()
