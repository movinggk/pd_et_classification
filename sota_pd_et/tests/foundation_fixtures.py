"""Small random upstream architectures for offline interface/gradient tests only.

These are never used by the experiment runner or represented as pretrained
results. They exercise each actual installed architecture without model downloads.
"""

import torch

from sota_pd_et.foundations.encoders import Encoder


def tiny_encoder(options, cache_dir=None):
    name = options.model
    if name in ("utica", "mantis_v2"):
        from mantis.architecture import MantisV1, MantisV2
        cls = MantisV1 if name == "utica" else MantisV2
        backbone = cls(hidden_dim=64, hidden_dim_scalar_enc=8, transf_depth=3,
                       transf_num_heads=2, transf_mlp_dim=64, transf_dim_head=32,
                       transf_dropout=0., return_transf_layer=2, output_token="combined", device="cpu")
    elif name == "moment":
        from momentfm import MOMENTPipeline
        config = {
            "task_name": "embedding", "d_model": 32, "seq_len": 512, "patch_len": 8, "patch_stride_len": 8,
            "transformer_type": "encoder_only", "transformer_backbone": "google/flan-t5-small",
            "t5_config": {"d_model": 32, "d_ff": 64, "d_kv": 8, "num_heads": 4, "num_layers": 2,
                          "vocab_size": 16, "dropout_rate": 0., "feed_forward_proj": "relu"},
            "freeze_embedder": False, "freeze_encoder": False, "enable_gradient_checkpointing": False,
        }
        backbone = MOMENTPipeline(config, model_kwargs={"task_name": "embedding"})
        backbone.init()
    elif name == "tspulse":
        from tsfm_public.models.tspulse import TSPulseConfig, TSPulseForReconstruction
        config = TSPulseConfig(
            context_length=512, patch_length=16, patch_stride=16, d_model=8,
            decoder_d_model=8, num_layers=2, decoder_num_layers=1,
            num_input_channels=1, mode="common_channel", decoder_mode="common_channel",
            patch_register_tokens=2, fuse_fft=True, dropout=0., head_dropout=0.,
            scaling="std", enable_fft_prob_loss=True, register_mixer_layers=1,
        )
        backbone = TSPulseForReconstruction(config).backbone
    elif name == "tirex":
        from tirex.models.tirex import TiRexZero
        backbone = TiRexZero("torch", model_config={
            "input_patch_size": 8, "output_patch_size": 8, "quantiles": [.1, .5, .9],
            "block_kwargs": {"embedding_dim": 16, "num_heads": 2, "num_blocks": 2},
            "input_ff_dim": 32,
        }, train_ctx_len=64)
        # TiRex constructors allocate these tensors with torch.empty; the
        # production loader fills them from the checkpoint. Initialize them
        # explicitly for a valid random, checkpoint-free test architecture.
        with torch.no_grad():
            for key, parameter in backbone.named_parameters():
                if key.endswith("_bias_"):
                    parameter.zero_()
                elif key.endswith("_recurrent_kernel_") or any(
                    key.endswith(f".{gate}.weight") for gate in ("fgate", "igate", "zgate", "ogate")
                ):
                    parameter.normal_(mean=0., std=.02)
    elif name == "chronos2":
        from chronos.chronos2 import Chronos2Model
        from chronos.chronos2.config import Chronos2CoreConfig
        config = Chronos2CoreConfig(
            d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=2, dropout_rate=0.,
            chronos_config={"context_length": 512, "input_patch_size": 8, "output_patch_size": 8,
                            "input_patch_stride": 8, "quantiles": [.1, .5, .9], "use_reg_token": True},
        )
        backbone = Chronos2Model(config)
    elif name == "tivit":
        from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
        from sota_pd_et.foundations._vendor.tivit.tivit import TiViT_HF
        config = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, image_size=224, patch_size=32)
        backbone = TiViT_HF(CLIPImageProcessor(), CLIPVisionModel(config).vision_model,
                            layer_idx=2, aggregation="mean", patch_size=31, stride=.1)
    elif name == "vqshape":
        from sota_pd_et.foundations._vendor.vqshape.model import VQShape
        backbone = VQShape(dim_embedding=24, num_enc_head=3, num_enc_layer=1,
                           num_tokenizer_head=3, num_tokenizer_layer=1, num_dec_head=3,
                           num_dec_layer=1, num_token=4, len_s=16, s_smooth_factor=3,
                           num_code=8, dim_code=4, lambda_entropy=0.)
    else:
        raise ValueError(name)
    return Encoder(backbone, options, {"test_only": True, "checkpoint_sha256": {"tiny": name}}).to(options.device)
