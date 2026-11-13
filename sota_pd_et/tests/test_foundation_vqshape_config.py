"""Protect non-state VQShape architecture metadata from silent fallback."""

from argparse import Namespace
import copy
import unittest

import torch

from sota_pd_et.foundations._vendor.vqshape.model import VQShape
from sota_pd_et.foundations.fast_paths import vqshape_token_features
from sota_pd_et.foundations.vqshape_config import vqshape_kwargs_from_hyperparameters


def hyperparameters():
    return {
        "normalize_length": 512, "dim_embedding": 24, "patch_size": 8, "num_patch": 64,
        "num_token": 4, "num_transformer_enc_heads": 3, "num_transformer_enc_layers": 1,
        "num_tokenizer_heads": 6, "num_tokenizer_layers": 1, "num_transformer_dec_heads": 3,
        "num_transformer_dec_layers": 1, "len_s": 16, "s_smooth_factor": 3,
        "num_code": 8, "dim_code": 4, "codebook_type": "standard",
        "lambda_vq_commit": .25, "lambda_vq_entropy": .1, "entropy_gamma": 1., "mask_ratio": .25,
    }


class FoundationVQShapeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_direct_and_nested_namespace_layouts_preserve_saved_settings(self):
        parameters = hyperparameters()
        unchanged = copy.deepcopy(parameters)
        expected = vqshape_kwargs_from_hyperparameters(parameters)
        self.assertEqual(expected["num_enc_head"], 3)
        self.assertEqual(expected["num_tokenizer_head"], 6)
        self.assertEqual(expected["lambda_commit"], .25)
        for value in (Namespace(**parameters), {"args": Namespace(**parameters)},
                      Namespace(args={"args": dict(parameters)})):
            self.assertEqual(vqshape_kwargs_from_hyperparameters(value), expected)
        self.assertEqual(parameters, unchanged)

    def test_shape_compatible_wrong_head_count_changes_features_and_missing_metadata_is_rejected(self):
        parameters = hyperparameters()
        kwargs = vqshape_kwargs_from_hyperparameters(parameters)
        torch.manual_seed(517)
        original = VQShape(**kwargs).eval()
        wrong = VQShape(**{**kwargs, "num_tokenizer_head": 3}).eval()
        # A strict state load succeeds despite different attention semantics.
        wrong.load_state_dict(original.state_dict(), strict=True)
        values = torch.randn(2, 512, generator=torch.Generator().manual_seed(181))
        with torch.no_grad():
            expected = vqshape_token_features(original, values)
            incorrect = vqshape_token_features(wrong, values)
        self.assertGreater(float((expected - incorrect).abs().max()), 1e-5)
        parameters.pop("num_tokenizer_heads")
        with self.assertRaisesRegex(ValueError, "num_tokenizer_heads"):
            vqshape_kwargs_from_hyperparameters(parameters)

    def test_missing_architecture_fields_are_rejected_instead_of_guessed(self):
        parameters = hyperparameters()
        pretraining = {"lambda_vq_commit", "lambda_vq_entropy", "entropy_gamma", "mask_ratio"}
        for field in parameters.keys() - pretraining:
            with self.subTest(field=field):
                altered = dict(parameters)
                altered.pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    vqshape_kwargs_from_hyperparameters(altered)

    def test_pretraining_only_defaults_do_not_change_tokens(self):
        full = vqshape_kwargs_from_hyperparameters(hyperparameters())
        parameters = hyperparameters()
        for field in ("lambda_vq_commit", "lambda_vq_entropy", "entropy_gamma", "mask_ratio"):
            parameters.pop(field)
        reduced = vqshape_kwargs_from_hyperparameters(parameters)
        torch.manual_seed(717)
        reference = VQShape(**full).eval()
        actual = VQShape(**reduced).eval()
        actual.load_state_dict(reference.state_dict(), strict=True)
        values = torch.randn(2, 512, generator=torch.Generator().manual_seed(19))
        with torch.no_grad():
            torch.testing.assert_close(vqshape_token_features(actual, values),
                                       vqshape_token_features(reference, values), rtol=0, atol=0)

    def test_invalid_architecture_metadata_fails_with_explicit_errors(self):
        cases = (("num_tokenizer_heads", 5), ("num_transformer_enc_heads", True),
                 ("num_token", 0), ("normalize_length", 256), ("codebook_type", "vqtorch"),
                 ("patch_size", 4), ("mask_ratio", float("nan")))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                altered = hyperparameters()
                altered[field] = value
                with self.assertRaises(ValueError):
                    vqshape_kwargs_from_hyperparameters(altered)
        for value in (None, [1, 2], {"args": "invalid"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vqshape_kwargs_from_hyperparameters(value)


if __name__ == "__main__":
    unittest.main()
