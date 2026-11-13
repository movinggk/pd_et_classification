# Original model specification audit

Audit date: **2026-11-12**. Scope: the seven alternative encoders and optional
MantisV2 in this comparison runner. Original code, pinned checkpoint configs,
author examples, installed package implementations, and our adapters were
checked separately. Tests use generated signals, never a new PADS experiment.

**The runner compares pretrained backbones under shared classifiers. It does
not reproduce eight original classification systems.** Only the corrected
adapter-version-3 processing is implemented: it includes TiRex's final
normalization/ordering, TiViT's final participant normalization and dynamic
signal patch size, MOMENT's sizing/masks, and strict VQShape architecture
metadata validation. Earlier processing branches and the version selector
have been removed. Other departures below remain explicit comparison choices.

## What “consistent” means here

Checkpoint consistency means loading the chosen published weights into the
corresponding architecture. Feature consistency additionally covers input
axes, resizing, normalization, masks, token selection, pooling, and ordering.
Recipe consistency would also require the author's classifier, parameter
freezing, optimizer, training schedule, augmentation, data split, and selection
metric. Passing the first two does not establish the third.

| Model | Checkpoint/core assessment | Feature-processing assessment | Original classification recipe reproduced? |
| --- | --- | --- | --- |
| TSPulse | Correct classification-specialized r1 checkpoint and original backbone | Resizing, RevIN/FFT path and axes agree; three-way mean pooling is our readout | No: TSLens/decoder, training masks, parameter freezing, and input scaling differ |
| UTICA | Correct published checkpoint through the supported MantisV1 architecture | Layer 2 plus combined tokens is explicitly supported by Mantis maintainers; resize agrees | No: shared Ridge/neural heads and training protocol |
| TiRex | Official loader restores checkpoint configuration/weights | Matches the complete non-augmented `TiRexEmbedding` wrapper | No: published classification uses a frozen encoder; our backbone fine-tuning is an extension |
| MOMENT-1-base | Correct embedding architecture/configuration; original RevIN retained | Masked left-padding and endpoint-preserving long-series resizing | No: per-channel concatenation and shared heads/protocol; dataset-specific external scalers not copied |
| Chronos-2 | Correct model/configuration and native encode path | Native normalization/patching/group attention retained; context mean pooling is our choice | No native published classification recipe to reproduce |
| TiViT-H | Correct CLIP vision backbone, layer and image processor | Signal image/quantization agree; final participant L2 normalization included | No: original frozen-feature classifier differs from our shared heads; supervised fine-tuning is an extension |
| VQShape | Correct author-released 256/512 checkpoint; strict state restore | Interpolation, token attributes and quantization agree for the same input; UEA dataset preprocessing is not copied | No: token-only shared classifier and backbone fine-tuning differ from the published benchmark |
| MantisV2 | Correct checkpoint/configuration | Recommended intermediate layer/combined tokens and resize agree | No: author fine-tuning head/trainer differs from the common comparison head |

## Data common to every adapter

The existing PADS author-preprocessed movement files are read without further
gravity filtering or task cropping. Their generation script orders task,
wrist, sensor, and axis; removes time channels and three tasks; gravity-corrects
accelerometers; and removes 48 leading samples. The resulting participant
array is `(132,976)`: 11 separate task segments, each with 12 wrist/sensor/axis
series. These are not 132 simultaneously sampled physical axes.

The cohort uses condition strings to select PD and ET, then maps PD to 0 and ET
to 1. The dataset's original numeric label 2 also covers other conditions and
is not used as an ET indicator. Participants remain intact in every split.
Chronos groups only the 12 series belonging to the same task. Source checksums
and the cohort/split rules are unchanged.

## TSPulse

The pinned `66cfb5464ed5e5a5cd7f5df1339af4e51461d955` revision corresponds to
`tspulse-block-dualhead-512-p16-r1`, the classification-specialized variant.
Its config specifies 512 context points, patch/stride 16, eight backbone
layers, width 48, five register tokens, FFT fusion, and affine RevIN scaling.
The adapter supplies `(channel_batch,512,1)`, and uses the original scaler,
time/FFT encoders, tokens, and encoder blocks. Linear interpolation with
`align_corners=False` agrees with the classification dataset implementation.
The optimized path preserves the FFT of affine RevIN output and removes only
the reconstruction-target FFT/scaler.

Our readout concatenates time-patch, FFT-patch, and register means: 144 values
per series, or 19,008 per participant. This is **not TSLens**. The author
notebook adds input scaling, channel expansion, a channel-mixing classification
decoder and training masking; it freezes the backbone except time/FFT patch
encodings. Our unmasked pooled features and adaptation of all active backbone
parameters deliberately differ. Checkpoint `num_targets=3` belongs to the
discarded original head; our binary classifier produces one PD/ET logit.

Sources: [pinned model card](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/blob/66cfb5464ed5e5a5cd7f5df1339af4e51461d955/README.md),
[author classification notebook, v0.3.12](https://github.com/ibm-granite/granite-tsfm/blob/v0.3.12/notebooks/hfdemo/tspulse_classification.ipynb),
[backbone/classification implementation](https://github.com/ibm-granite/granite-tsfm/blob/v0.3.12/tsfm_public/models/tspulse/modeling_tspulse.py).

## UTICA and MantisV2

Both use the supported 512-point linear resize and independent per-channel
processing. Our zero-based layer index 2 and concatenated CLS/mean patch tokens
produce 512 values per series, or 67,584 per participant. The maintained Mantis
documentation explicitly recommends this readout for **both** checkpoints.
UTICA's basic quickstart instead uses the final CLS token. That is an alternate
documented readout, not evidence that the selected intermediate readout is a bug.

UTICA's `vit_unit` names are mapped to current `transf_unit` names. Only its
pretraining mask token/final normalization and absent, unused projection head
are excluded, as the maintainer tutorial explains. All selected-path tensors
load strictly. MantisV2 architecture settings come from its pinned config;
readout overrides are the documented feature-extraction settings.

The common fine-tuning head is LayerNorm/dropout/linear. Mantis 1.1.0's default
trainer instead uses BatchNorm/linear. Our class weighting, width-scaled AdamW
head rate, nested selection, and training schedule are comparison settings.
Only layers on the selected intermediate representation receive classifier
gradients; “backbone fine-tuning” does not mean that unused later layers learn.

Sources: [pinned Mantis maintainer documentation](https://github.com/vfeofanov/mantis/blob/23ecfe493830db177546ef159947de7d79a573bd/README.md),
[UTICA maintainer tutorial](https://github.com/vfeofanov/mantis/blob/23ecfe493830db177546ef159947de7d79a573bd/getting_started/utica.ipynb),
[UTICA quickstart](https://github.com/fegounna/Utica/blob/e8cd3e85f7c0e7ad7c02fc45efb6ee28d0f40323/README.md),
[MantisV2 pinned config](https://huggingface.co/paris-noah/MantisV2/blob/99fe0f548960e272fbfa4b82fd9b5b5956779dfd/config.json).

## TiRex

The official local checkpoint loader restores its saved hyperparameters and
state; our loader does not guess its layer count, hidden width, patch size, or
training context length. Original context preparation handles scaling, masks,
left padding, and truncation to the checkpoint context limit. The original
PyTorch backend, including its recurrent-state precision, is retained.

The recurrent product retains the native forward calculation but uses an
analytic headwise backward, avoiding per-series copies of the recurrent
matrix in autograd's saved tensors. The exponential-gate stability correction
also remains active. This changes no checkpoint tensor or feature definition;
BF16 gradient reductions may round differently. See the
[released-checkpoint check](../tmp/foundation_validation/tirex_headwise_gradient_validation.json)
and [storage/derivative tests](tests/test_foundation_tirex_stability.py).

The native non-augmented wrapper pools context patches separately by layer and
variate, L2-normalizes each hidden vector, flattens in **layer/channel/hidden**
order, and applies participant-local LayerNorm. The current adapter follows
this processing; regression checks compare the complete author wrapper for shorter,
equal-length, and longer-than-context synthetic inputs.

Optional differences/statistics augmentation remains disabled, matching the
library's `data_augmentation=False` default. Enabling its cohort-normalized
statistics would require fold-local fitting. The published classification
encoder is frozen; our supervised backbone adaptation is an experimental
extension. Feature width is `132 * num_blocks * embedding_dim`, resolved from
the actual checkpoint rather than asserted from an unavailable config file.

Sources: [embedding implementation, v1.4.2](https://github.com/NX-AI/tirex/blob/v1.4.2/src/tirex/models/embedding.py),
[author classification description](https://github.com/NX-AI/tirex/blob/v1.4.2/docs/content/how-to/classification/theory.md),
[linear classifier](https://github.com/NX-AI/tirex/blob/v1.4.2/src/tirex/models/classification/linear_classifier.py).

## MOMENT-1-base

The pinned checkpoint config selects the T5-base encoder: 12 layers, width
768, 512-point input, and patch/stride 8. Reconstruction weights load before
switching to the supported embedding task. Original RevIN, positional/patch
embedding, and attention-mask processing are retained. On complete resized
signals, our per-channel patch means match native `embed(reduction="mean")`
with one channel. Concatenation produces 101,376 participant features.

The performance adapter retains native T5 attention and caches a small table
of deterministic integer relative-position bucket IDs. The table is computed
using native arithmetic on its destination device. Each call still reads the
current learned bias embedding, so bias values and gradients are never frozen
by the cache. It supports arbitrary cache positions and device changes; the
shortcut is restricted to the audited Transformers version. Input sizing and
mask construction run before activation checkpoint replay, while native RevIN,
patch embedding, and all trainable encoder operations retain their gradients.

Input sizing follows author examples: short series are left-padded with
an observed-value mask; long series use endpoint-preserving interpolation;
exactly 512 samples pass through unchanged. Native `reduction="mean"` applies
the observed-patch mask in pooling, while original observed-only RevIN handles
normalization. Inputs shorter than a complete patch receive a clear error.
The PTB-XL example's external per-series z-score and dataset-specific scaler
fitting are not added. This is author-supported sizing, not reproduction of
every dataset-specific preprocessing pipeline. Earlier heads that depend on
different sizing are rejected by prediction and require a new fit.

Sources: [pinned checkpoint config](https://huggingface.co/AutonLab/MOMENT-1-base/blob/5e44b0ea26376a176360f87831124e018f876d96/config.json),
[embedding implementation](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/models/moment.py),
[short-series example](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/data/classification_dataset.py),
[PTB-XL preprocessing](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/data/ptbxl_classification_dataset.py).

## Chronos-2

The pinned model has width 768, 12 layers, context capacity 8,192, and 16-point
input/output patches. The 976 observations are passed without resampling.
Native instance scaling, arcsinh transform, patch/time encoding and attention
are retained. Explicit group IDs join only a participant's simultaneous
12-series task; different tasks/participants never share an attention group.

The original embedding API returns context tokens plus REG and a masked future
token. Our representation averages only context tokens and concatenates series,
giving 101,376 participant features. Synthetic tests compare these tokens with
`Chronos2Pipeline.embed` using the same task grouping. Pooling and classification
are our extensions to a forecasting model; there is no original Chronos-2
classification head or classification training recipe being reproduced.

For complete finite contexts, the adapter batches independent equal-size tasks
directly along the group-attention batch axis. This avoids constructing and
computing masked cross-task blocks. Native context/future preparation, token
insertion, time attention, normalization, residuals and feedforward layers are
retained. Incomplete/nonfinite contexts and unverified package versions use
native `encode`, preserving entirely masked-patch behavior. The default chunk
is now 24 channels (two 12-variate tasks); there is still no cross-task or
cross-participant attention. CPU/MPS feature, gradient and group-boundary tests
are in [the throughput tests](tests/test_foundation_large_model_speed.py).

Sources: [pinned checkpoint config](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/config.json),
[native embedding API](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos2/pipeline.py),
[native encoder](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos2/model.py).

## TiViT with CLIP ViT-H

The selected CLIP vision config has width 1,280, 32 available blocks, 14-pixel
vision patches, and 224-pixel images. The TiViT example uses the first 14 blocks,
mean token aggregation, signal patch size `floor(sqrt(T))`, and stride fraction
0.1. For our fixed `T=976`, the signal patch size is 31; this is distinct from
CLIP's 14-pixel patch size. The adapter computes `floor(sqrt(T))` per call so
shorter generic inputs no longer inherit fixed patch size 31 and fail.
The local override does not mutate the model and is retained by fallback
processing and activation-checkpoint recomputation. PADS-length features
are unchanged by this sizing correction.

The signal-to-image transform is performed once per channel chunk, outside
activation-checkpoint replay. The standard processor path retains one uint8
grayscale image and rebuilds its exact three-channel normalized pixels by a
small lookup during replay. Its checkpoint input occupies 1/12 of the storage
of prepared FP32 RGB pixels. This adds approximately 12.6 MiB of retained
images at `(2,132,976)`, instead of repeating image creation from the raw
signals. Nonstandard processors continue through the original vision path.

The official robust scaling, overlapping signal patches, replicate padding,
contrast/gamma operation, nearest image resize, RGB repetition, 8-bit PIL
quantization, and checkpoint image normalization are retained. Tensor lookup
processing replaces only equivalent host conversion operations. The whole
concatenated participant vector is L2-normalized. The feature width is 168,960.

The author's frozen linear classification workflow and our shared Ridge head
differ. Our supervised backbone fine-tuning is also distinct from the frozen
workflow and the paper's separate continual-pretraining experiments.
The original UCR loader additionally supports missing-value interpolation,
unequal-length padding and dataset-level scaling. Our finite, fixed-length
PADS binaries do not go through that dataset-specific loader.

Sources: [model/image implementation](https://github.com/ExplainableML/TiViT/blob/c74a76ffc4197c6d2d5d766c8c90d438efb88cfd/src/tivit.py),
[author embedding wrapper](https://github.com/ExplainableML/TiViT/blob/c74a76ffc4197c6d2d5d766c8c90d438efb88cfd/src/embedding.py),
[signal patch-size rule](https://github.com/ExplainableML/TiViT/blob/c74a76ffc4197c6d2d5d766c8c90d438efb88cfd/src/utils.py),
[checkpoint vision config](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/1c2b8495b28150b8a4922ee1c8edee224c284c0c/config.json).

## VQShape

The selected author-released 256-dimensional/512-code checkpoint is explicitly
listed as an additional release, not one of the paper's reported checkpoints.
Its saved hyperparameters specify 64 tokens, code width 8, 512 input points,
8-point patches, 64 patches, four encoder/tokenizer layers, and two decoder
layers. The loader reads these values and restores all 196 state tensors
strictly. It does not use constructor defaults in place of saved dimensions.
The deeper audit removed the earlier fallback kwargs: all 16 architecture
settings must now be explicit. Attention-head counts cannot be inferred from
weight shapes, so strict tensor loading alone is insufficient. A regression
test demonstrates that different head counts can accept the same state dict
yet produce different tokens. The authentic release is accepted and retains
bitwise-identical token features under the new loader.

The original linear interpolation, per-series sample-variance normalization,
nearest-code assignment, straight-through estimator, and token attribute order
are preserved. Each token has 12 coordinates: code plus position, length,
location, and scale. Flattening 64 tokens gives 768 per series and 101,376 per
participant. Histograms are excluded deliberately. The author's feature scaler
and trained classifier differ from the shared comparison head; our supervised
encoder adaptation is an extension. Skipping unused decoder dropout changes
later RNG consumption, as already documented.

The native UEA benchmark loader also standardizes its dataset dataframe,
handles missing/unequal-length cases, and has a dataset-specific numerical
special case. The standalone tokenization example does not require that
loader. We retain the standalone tokenization contract for finite PADS arrays;
we do not claim to reproduce the UEA loader or fit input statistics jointly
across outer-training and outer-test participants.

Sources: [release/checkpoint description](https://github.com/YunshiWen/VQShape/blob/24fc507461e17b4b8485d5f097aa91eaef98ae67/README.md),
[native tokenizer/model](https://github.com/YunshiWen/VQShape/blob/24fc507461e17b4b8485d5f097aa91eaef98ae67/vqshape/model.py),
[classification benchmark](https://github.com/YunshiWen/VQShape/blob/24fc507461e17b4b8485d5f097aa91eaef98ae67/benchmarks/mtsc.py).

## Validation and compatibility

The [new specification tests](tests/test_foundation_original_specs.py) compare
all eight adapters with native library operations or the author wrapper using
small actual architectures. They also check rejection of unsupported saved
artifact versions. The current suite covers native gradient comparisons,
variable-length Chronos boundaries, current-artifact prediction, and
train/validate/train checks for all eight architectures; see the
[current regression record](../tmp/foundation_validation/large_model_speed_regression.json). Source URLs,
retrieved hashes, config snapshots and scope findings are recorded in the
[machine-readable audit](../tmp/foundation_spec_audit/audit.json).

An additional [released-checkpoint check](../tmp/foundation_spec_audit/cached_checkpoint_validation.json)
used already cached TSPulse, UTICA and VQShape weights on generated
`(2,2,976)` signals. All three matched their stated reference calculations
bit for bit. The [check script](../tmp/foundation_spec_audit/verify_cached_checkpoints.py)
is included for reproduction. For TSPulse the reference is the original
backbone followed by our explicit pooling, not the TSLens classification head.

The common classifier now defaults to inner-validation threshold selection
(`--decision-rule inner-balanced`) and a broader frozen Ridge alpha grid
(`1e-3` through `1e9`). These are shared comparison-protocol changes, not
changes to an encoder or reproductions of an author's classifier recipe.
Each outer fold saves its inner-selected threshold, adjusted margins, raw
scores, and raw-zero metrics. Fine-tuning now reserves checkpoint validation
before LR tuning, confines inner tuning to fitting participants, restores the
outer/final model's own best checkpoint, and retains that checkpoint's own
validation threshold. The median-epoch refit and raw threshold transfer have
been removed. The same rule applies to all eight adapters and final-model fits.
The [decision tests](tests/test_foundation_decision.py) include negative scores,
constant scores, tied scores, and invariance of a fold's threshold/alpha to
changes in that fold's outer-test labels. The subsequent
[all-model decision tests](tests/test_foundation_all_model_decisions.py) exercise
both nested modes and saved-threshold replay through every small actual
architecture. The [checkpoint tests](tests/test_foundation_checkpoint_validation.py)
also force later training to worsen the selection metric, then verify exact
weight restoration and replay of selected validation scores. They verify
separation of fitting, checkpoint validation, LR tuning, and outer testing.
Direct training/tuning calls share the CLI's inner-balanced
default. Tuned artifacts with missing raw thresholds fail before model loading;
they cannot silently revert to raw zero. See the current
[all-model validation record](../tmp/foundation_validation/validation_checkpoint_validation.json).

These checks do not establish numerical identity on every full-size checkpoint
or device. Large uncached weights were not downloaded solely for this audit.
No real-cohort training, tuning against outer-test labels, or accuracy comparison
was executed. The TiRex/TiViT corrections do not establish the cause of earlier
TSPulse/UTICA single-class predictions.

All runs use adapter version 3, recorded as a fixed metadata marker. The
implementation has no version selector or older processing branches.
Prediction requires version-3 metadata and rejects older versions or a missing
marker before loading a model: equal feature widths do not imply equal feature
values or ordering. Existing runs and artifacts remain intact; older heads
require a new fit with the current processing. A running process retains its
already imported code, so the current implementation requires a new launch.

The deeper audit also tested inference-first and train/validate/train cycles.
No inference-mode cache poisoning was reproduced in these pinned libraries;
prediction remains in inference mode. Further constant, low-variance and shifted
TSPulse/VQShape checks matched native outputs and active gradients exactly.
See the [MOMENT/Chronos record](../tmp/foundation_spec_audit/deeper_moment_chronos.json)
and [VQShape metadata record](../tmp/foundation_spec_audit/vqshape_config_verification.json).
