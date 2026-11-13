"""Exercise real library checkpoint loaders using small local test checkpoints."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from sota_pd_et.foundations import encoders
from sota_pd_et.foundations.encoders import EncoderOptions, load_utica_state
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class FoundationCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_chronos_checkpoint_round_trip_through_production_loader(self):
        options = EncoderOptions("chronos2", chronos_group_size=2)
        original = tiny_encoder(options).eval()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            original.backbone.save_pretrained(path)
            with mock.patch.object(encoders, "_snapshot", return_value=(path, {"fixture": "chronos"})):
                loaded = encoders.load_encoder(options, path / "cache")
            X = torch.randn(2, 2, 64)
            with torch.no_grad():
                torch.testing.assert_close(original(X), loaded(X))

    def test_moment_reconstruction_checkpoint_to_embedding_adapter(self):
        from momentfm import MOMENTPipeline
        options = EncoderOptions("moment")
        original = tiny_encoder(options).eval()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            # Released MOMENT weights include the reconstruction head. Save a
            # real tiny reconstruction checkpoint to test strict loading, then
            # switching to embedding mode, including all encoder weights.
            config = dict(original.backbone.config.__dict__)
            config["task_name"] = "reconstruction"
            reconstruction = MOMENTPipeline(config, model_kwargs={"task_name": "reconstruction"})
            reconstruction.save_pretrained(path)
            reconstruction.init()
            with mock.patch.object(encoders, "_snapshot", return_value=(path, {"fixture": "moment"})):
                loaded = encoders.load_encoder(options, path / "cache")
            for key, value in reconstruction.encoder.state_dict().items():
                torch.testing.assert_close(value, loaded.backbone.encoder.state_dict()[key])
            self.assertEqual(loaded(torch.randn(1, 2, 64)).shape, (1, 64))

    def test_tivit_full_clip_checkpoint_loads_only_vision_backbone(self):
        from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor
        config = CLIPConfig.from_text_vision_configs(
            CLIPTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4),
            CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                             image_size=224, patch_size=32),
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            original = CLIPModel(config)
            original.save_pretrained(path)
            CLIPImageProcessor().save_pretrained(path)
            with mock.patch.object(encoders, "_snapshot", return_value=(path, {"fixture": "tivit"})):
                loaded = encoders.load_encoder(EncoderOptions("tivit"), path / "cache")
            self.assertEqual(loaded(torch.randn(1, 2, 64)).shape, (1, 64))
            torch.testing.assert_close(original.vision_model.embeddings.patch_embedding.weight,
                                        loaded.backbone.vit.embeddings.patch_embedding.weight)

    def test_utica_allows_only_documented_pretraining_differences(self):
        original = tiny_encoder(EncoderOptions("utica")).backbone
        state = {k: v for k, v in original.state_dict().items() if not k.startswith("prj.")}
        state.update({"transf_unit.mask_token": torch.zeros(1), "transf_unit.norm.weight": torch.ones(64),
                      "transf_unit.norm.bias": torch.zeros(64)})
        target = tiny_encoder(EncoderOptions("utica")).backbone
        ignored = load_utica_state(target, state)
        self.assertEqual(len(ignored), 3)
        broken = dict(state)
        broken.pop(next(k for k in broken if k.startswith("tokgen_unit.")))
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_utica_state(target, broken)

    def test_tirex_checkpoint_round_trip_through_official_loader(self):
        original = tiny_encoder(EncoderOptions("tirex")).eval()
        import tirex.base
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.ckpt"
            config = original.backbone.config
            torch.save({
                "state_dict": original.backbone.state_dict(),
                "hyper_parameters": {
                    "model_config": {"input_patch_size": config.input_patch_size,
                                     "output_patch_size": config.output_patch_size,
                                     "quantiles": config.quantiles, "block_kwargs": config.block_kwargs,
                                     "input_ff_dim": config.input_ff_dim},
                    "train_ctx_len": config.train_ctx_len,
                },
            }, path)
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(path)), \
                 mock.patch.object(tirex.base, "hf_hub_download", return_value=str(path)):
                loaded = encoders.load_encoder(EncoderOptions("tirex"), Path(root) / "cache")
            X = torch.randn(2, 2, 64)
            with torch.no_grad():
                torch.testing.assert_close(original(X), loaded(X))


if __name__ == "__main__":
    unittest.main()
