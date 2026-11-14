# PADS comparison of pretrained feature extractors and fine-tuned encoders

This runner implements **TSPulse, UTICA, TiRex, MOMENT-1-base, Chronos-2,
TiViT with CLIP ViT-H, and VQShape**. MantisV2 is available as an optional eighth
model. It supports two modes for every model:

- **`frozen`**: fixed pretrained features, a fold-local StandardScaler, and a
  class-balanced ridge classifier selected by inner cross-validation.
- **`finetune`**: supervised adaptation of the pretrained feature-extraction
  backbone and a common LayerNorm → Dropout → Linear classification head.
- **`both`** runs both modes using the same saved outer-test assignments.
  Fine-tuning additionally reserves checkpoint validation within outer training.

Both modes default to `--decision-rule inner-balanced`: select a raw-score
threshold using training-pool validation predictions only. Every backbone uses the
same rule. Scores returned for classification subtract this saved threshold;
positive adjusted margins mean ET. `--decision-rule zero` retains the fixed
raw-zero decision rule. Class weighting remains enabled under either rule.

The implementation has been tested without running a PADS training benchmark.
The test suite uses small random configurations of the actual upstream
architectures, synthetic participant arrays, and local checkpoint round trips.
Additional smoke checks use the released TSPulse, UTICA, and VQShape checkpoints.
See `../tmp/foundation_validation/` for the local validation records. No synthetic
test accuracy is represented as a model comparison result.

## Install

Use **Python 3.12** and a separate environment. From the project root:

```sh
python3.12 -m venv .venv-foundations
.venv-foundations/bin/python -m pip install -r requirements.txt
.venv-foundations/bin/python -m pip check
```

The repository-root `requirements.txt` installs every pipeline, plotting,
report builders, and notebook tools. For only this model comparison, the narrower
`sota_pd_et/requirements-foundations.txt` remains available. Git must be installed
for the MOMENT source dependency. Use a fresh environment rather than combining
the historical platform-specific lock files. The portable requirements pin the
model packages, PyTorch 2.11.0, torchvision 0.26.0, Transformers 4.57.6, and
scikit-learn 1.7.2. MOMENT is installed from an immutable upstream commit because
its PyPI release pins dependencies that conflict with the other models.
SciPy is limited to below 1.18 for compatibility with aeon in the complete
environment. The original resolved macOS environment is recorded in
`requirements-foundations-lock-macos.txt`;
prefer the portable requirements file for Linux/CUDA rather than installing
the macOS lock unchanged.

For NVIDIA hardware, install the matching CUDA-enabled PyTorch/torchvision
builds for your system. `--device auto` chooses CUDA when available and CPU
otherwise. `--device cuda` requires a working CUDA installation. On this Mac,
request the GPU explicitly with `--device mps`. All eight adapters have small
architecture checks on CPU. Synthetic MPS checks cover the released TSPulse
checkpoint and small TiViT/VQShape architectures; CUDA has not been tested.

The released TSPulse checkpoint has additionally passed a synthetic MPS
forward/backward check on this Mac. PyTorch 2.11.0's MPS FFT can emit an
internal buffer-resize warning with expected output shape `[batch,257,1]`.
An eight-signal CPU/MPS comparison found relative L2 differences below `1e-6`
for the FFT, extracted features, and backbone gradients. The adapter filters
only that precise warning from TSPulse's module, only on MPS with PyTorch
2.11.0, and only during that backbone call. The FFT computation is unchanged.
This does not validate full cohort training or the other backbones on MPS.
See the [upstream issue](https://github.com/pytorch/pytorch/issues/174003) and
`../tmp/foundation_validation/mps_fft_warning_validation.json` for the check.

Model weights download on the first real run, then remain in
`models/foundation_cache/`. Hub model revisions are immutable commits. VQShape
uses the authors' versioned release archive and records its checkpoint SHA-256.
The tested small released checkpoints are already cached in this workspace.

## Verify the code and data without training

```sh
.venv-foundations/bin/python -m unittest \
  sota_pd_et.tests.test_foundation_adapters \
  sota_pd_et.tests.test_foundation_checkpoint_io \
  sota_pd_et.tests.test_foundation_protocol \
  sota_pd_et.tests.test_transformer_protocol \
  sota_pd_et.tests.test_foundation_fast_paths \
  sota_pd_et.tests.test_foundation_tspulse_fast \
  sota_pd_et.tests.test_foundation_ridge \
  sota_pd_et.tests.test_foundation_decision \
  sota_pd_et.tests.test_foundation_all_model_decisions \
  sota_pd_et.tests.test_foundation_checkpoint_validation \
  sota_pd_et.tests.test_foundation_tirex_stability \
  sota_pd_et.tests.test_foundation_restart \
  sota_pd_et.tests.test_foundation_large_model_speed \
  sota_pd_et.tests.test_foundation_training_performance \
  sota_pd_et.tests.test_foundation_head_stability \
  sota_pd_et.tests.test_foundation_original_specs \
  sota_pd_et.tests.test_foundation_moment_input \
  sota_pd_et.tests.test_foundation_tivit_sizing \
  sota_pd_et.tests.test_foundation_vqshape_config \
  sota_pd_et.tests.test_foundation_training_lifecycle \
  sota_pd_et.tests.test_foundation_loading_cache -v

.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --check-data \
  --alphas 0.001 0.01 0.1 1 10 100 1000 \
  --baseline artifacts/pads_mantis_v2
```

The first command uses actual small upstream model architectures, performs
forward/backward checks, verifies backbone weight changes, and tests nested
evaluation and prediction on synthetic data. It does not download pretrained
weights. The second verifies the official source-file checksums, cohort, and
outer/inner assignments against the existing frozen MantisV2 run. It exits
before downloading weights or fitting any classifier.

## Run the comparison yourself

Frozen models only:

```sh
.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --models all --mode frozen \
  --device auto \
  --output artifacts/foundations_frozen
```

Fine-tune every model:

```sh
.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --models all --mode finetune \
  --device auto \
  --learning-rates 1e-5 3e-5 \
  --max-epochs 10 --patience 3 \
  --batch-size 2 --accumulation-steps 4 \
  --output artifacts/foundations_finetuned
```

Both modes, including final models for later prediction:

```sh
.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --models all --mode both --fit-final \
  --device auto \
  --output artifacts/foundations_both
```

`all` means the seven alternatives. To include MantisV2 in the same comparison:

```sh
.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --models tspulse utica tirex moment chronos2 tivit vqshape mantis_v2 \
  --mode both --fit-final --device auto \
  --output artifacts/foundations_with_mantis
```

Run one model by supplying just its name, for example `--models tspulse`.
Progress bars are enabled by default for model/mode runs, outer and inner
cross-validation folds, feature extraction, fine-tuning epochs and participant
microbatches, and validation/test prediction. They show completed work,
elapsed time, speed, and an estimated remaining time as measurements become
available. Fine-tuning also shows the running loss, inner validation balanced
accuracy, and the best epoch. Epoch totals are upper limits when early
stopping is enabled.

Add `--no-progress` to either the comparison or prediction command to disable
these bars. The comparison then retains its plain-text status and epoch logs.
Checkpoint-download bars are controlled by the upstream download libraries.

`--output` names a **parent results directory**, which can already exist.
Every ordinary invocation creates a new subdirectory with an automatic run ID, for
example `run_20261111T040000000000Z_a1b2c3d4e5f6`. The ID combines a UTC
timestamp and a random suffix; directory creation is atomic and retries a
collision. You can rerun the same command with the same `--output` value.
Existing results, including runs saved directly in the parent by earlier
versions, remain in place. The run ID and complete results directory are
printed before model loading and recorded in that run's `manifest.json`.

Omitting `--output` uses `artifacts/foundation_comparison/` as the parent.
All model, mode, fold, comparison, and final-model files are saved beneath
the new run directory. Failed runs retain their completed files and status. `--continue-on-error`
allows the other requested model/mode runs to proceed, records each failure,
and still returns a nonzero exit status when any requested run failed.

To restart whole incomplete models in the original run directory:

```sh
.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --restart-incomplete artifacts/foundations_comparison/REPLACE_WITH_RUN_ID
```

This restores the original configuration and verifies dataset provenance and
both saved split plans. Completed model/mode tasks are skipped and their files
are untouched. Each incomplete model/mode restarts at its first outer fold;
its previous directory is renamed to `MODE_interrupted_TIMESTAMP_SUFFIX`.
The previous manifest and new source hashes are retained in `restart_history/`.
Pending models then continue in the original order. The existing run ID is
retained and the comparison table keeps completed rows.

`--no-progress`, `--check-data`, `--model-channel-batch-size MODEL=N`, and
`--model-gradient-checkpointing MODEL=on|off` can accompany this flag. The two
model options are repeatable and work for every listed encoder. The existing
`--tirex-channel-batch-size` and `--tirex-gradient-checkpointing on|off` options
remain accepted. Training settings come from the saved manifest; runtime
overrides apply only when an incomplete model is restarted and are recorded
in restart history. Completed records and model files stay untouched, even
if an override names that completed model. A running or completed run cannot be restarted.
This does not resume an optimizer or partially trained inner/outer fold.

Fine-tuning has substantial cost: the defaults require **35 separate backbone
fits per model**: 5 outer folds × (2 learning rates × 3 inner folds + 1 outer
checkpoint fit). `--fit-final` adds 7 more fits per model. Start with one model if you
want to estimate runtime. Frozen feature extraction runs once per model.

TiViT-H is the largest backbone here. Reducing `--batch-size 1` and
`--channel-batch-size 1` reduces activation memory. Channel-chunk gradient
checkpointing recomputes activations during backward to reduce memory.
It is enabled by default for every model except the small TSPulse backbone.
Use `--gradient-checkpointing` to enable it explicitly or
`--no-gradient-checkpointing` to disable it explicitly for all requested models.
The default channel chunk is **128 for TSPulse, 24 for Chronos-2, and 8 for the other models**;
`--channel-batch-size N` overrides it for all requested models. The resolved
settings are printed and saved per model in `manifest.json`.
AdamW still needs memory for the trainable
weights, their gradients, and optimizer states. The runner does not enable AMP
or substitute LoRA. TiRex's official PyTorch backend internally uses bfloat16
for its recurrent state computations; that upstream behavior is retained.

### Runtime settings for the other large models

Use model-specific settings to avoid applying a large TiRex/TSPulse chunk to
TiViT-H. For example, the following keeps activation recomputation enabled,
uses two 12-variate tasks per Chronos chunk, and keeps TiViT at chunk 8:

```sh
.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --restart-incomplete artifacts/foundations_comparison/REPLACE_WITH_RUN_ID \
  --model-channel-batch-size moment=24 \
  --model-channel-batch-size chronos2=24 \
  --model-channel-batch-size tivit=8 \
  --model-channel-batch-size vqshape=24
```

For new runs the same flags accompany the usual `--models`, `--mode`, and
`--device` options. Per-model settings take precedence over global settings.
Duplicate model entries and conflicting TiRex aliases are rejected. On
restart, supplied entries replace that model's saved runtime entry; other
saved entries are retained. The model/mode training order remains the original one.

When memory permits, append `--model-gradient-checkpointing moment=off` or
`--model-gradient-checkpointing vqshape=off` to remove recomputation for those
models. TiViT-H's released backbone is much larger than the test architecture;
keep its default chunk/checkpointing initially. Larger chunks and disabling
recomputation increase activation memory; neither setting is an accuracy fix.

The automatic changes preserve the existing representations:

- MOMENT keeps native T5 attention and caches only deterministic integer
  relative-position buckets. Learned bias values are read from current weights
  on every call, with their original gradient path.
- Chronos-2 packs independent task groups into a batch, avoiding masked
  cross-task attention matrices. Its input preparation, REG/future tokens,
  time attention, and context-only pooling remain native. Nonfinite/incomplete
  contexts and unsupported library versions use the original encode path.
- TiViT creates its official image once per channel chunk, saving one uint8
  grayscale image for checkpoint replay. Only the inexpensive pixel-table
  lookup and vision backbone are recomputed. At two participants × 132 channels
  the retained images total approximately 12.6 MiB, versus 151.6 MiB if all
  prepared inputs were retained as three-channel FP32 pixels. The former
  raw-signal-only checkpoint input was approximately 1 MiB, so this optimization
  adds a small fixed image cache to avoid repeating image creation.
- MOMENT sizing/masking, VQShape resizing/per-series normalization, and
  UTICA/Mantis resizing also run outside activation recomputation. Trainable
  transforms remain inside the differentiable model path.

The [MPS component benchmark](../tmp/foundation_validation/large_model_speed_benchmark.json)
uses generated `(1,24,976)` signals, one warmup, and three timed backward calls
per setting, without optimizer updates. MOMENT, Chronos-2, and TiViT use
explicit smaller random upstream architectures; VQShape uses its cached
released checkpoint. Relative to the previous chunk-8/checkpointed path:

| Model | Previous milliseconds | Optimized chunk 24, checkpointing on | Optimized chunk 24, checkpointing off |
| --- | ---: | ---: | ---: |
| MOMENT, smaller architecture | 51.60 | 49.01 (1.05x) | 36.32 (1.42x) |
| Chronos-2, smaller architecture | 60.02 | 54.10 (1.11x) | 40.17 (1.49x) |
| TiViT, smaller architecture | 156.16 | 147.50 (1.06x) | 111.23 (1.40x) |
| VQShape, released weights | 161.58 | 148.79 (1.09x) | 104.07 (1.55x) |

The automatic shortcuts alone were modest at chunk 8: approximately 1.00x,
1.00x, 1.05x, and 1.03x respectively. The main additional gain in this workload
comes from batching and avoiding recomputation. These are component timings,
not full-run speedups, memory-fit guarantees, or clinical accuracy results.
All tested gradients were finite; native evaluation features at chunk 8
matched bit for bit. The [tests](tests/test_foundation_large_model_speed.py)
also compare native parameter gradients and independent task boundaries.

### TiRex throughput on MPS

The portable sLSTM path has many small recurrent operations. With two
participants and 132 channels, chunk 8 makes 33 channel chunks per forward.
The recurrent matmul also originally caused autograd to retain a broadcast
copy of its weights per series at every timestep. The adapter now preserves
that native forward calculation but computes its analytic backward using
headwise matrix products, avoiding those saved copies. It retains the
overflow-safe exponential gates and finite-gradient checks.

Try chunk 32 with recomputation retained:

```sh
.venv-foundations/bin/python -m sota_pd_et.compare_foundations \
  --restart-incomplete artifacts/foundations_comparison/REPLACE_WITH_RUN_ID \
  --tirex-channel-batch-size 32 \
  --tirex-gradient-checkpointing on
```

Stop an active run yourself before restarting. This starts the incomplete
model at fold 1 and preserves completed models. To reduce recomputation when
memory permits, use `--tirex-gradient-checkpointing off`. A full participant
batch uses more memory than the short benchmark, including optimizer state;
turning off checkpointing is not guaranteed to fit every Mac.

The [released-weight MPS benchmark](../tmp/foundation_validation/tirex_batching_benchmark.json)
used one generated 32-channel, 976-point participant, one warmup, and two timed
feature-forward/backward calls per setting, without optimizer updates:

| TiRex channel chunk | Checkpointing | Median time | Relative throughput |
| --- | --- | ---: | ---: |
| 8 | on | 6.864 s | 1.00x |
| 32 | on | 4.967 s | 1.38x |
| 32 | off | 2.958 s | 2.32x |

These are component timings, not full-run speedups. Gradients stayed finite;
the feature difference between chunk settings was at most `2.87e-6` in this
check. The [native-forward check](../tmp/foundation_validation/tirex_headwise_gradient_validation.json)
matched the released native features bit for bit at its original chunk size.
The [tests](tests/test_foundation_tirex_stability.py) include analytic derivative
checking, reduced saved-tensor storage, and feature/gradient checks for batching
and recomputation. Native BF16 reductions can introduce small rounding changes.

## Performance fixes and validation

The runner avoids unused reconstruction/forecasting computations in VQShape,
TiRex, and TSPulse. TSPulse computes the FFT needed by classifier features once,
without the second FFT used only for reconstruction targets. TiViT retains the
official signal-to-image transform and exact 8-bit image quantization while
replacing repeated PIL/NumPy host processing with a tensor lookup. The
TSPulse and TiViT shortcuts fall back to upstream forwards for unverified
runtime/configuration or processor settings.

Training batches avoid unnecessary NumPy copies. Gradient checks aggregate
their results on the device, validation predictions transfer together, and
progress loss displays synchronize only when they refresh. Frozen feature
extraction preallocates its output matrix, avoiding a second full feature copy.
Frozen Ridge searches reuse one weighted eigendecomposition per inner fold
across alpha candidates. Each fold still fits its own scaler and balanced
weights; numerical edge cases use the original scikit-learn solver.

Pinned checkpoint paths, file digests, and package versions are cached within
each process. File changes invalidate cached digests; deleted checkpoint paths
are resolved again. Every fit still loads a fresh backbone with independent
parameters. Trained weights are never reused between folds.

The [performance report](FOUNDATION_PERFORMANCE.md) records synthetic CPU and
MPS timings, correctness checks, and their limits. Those timings are component
measurements, not end-to-end cohort benchmark speedups. No real training
benchmark was run as part of these fixes.

Skipping VQShape's unused decoder also skips its dropout RNG consumption, so
the sequence of later dropout masks changes relative to the previous code.
Changing a channel chunk can likewise change dropout draws and floating-point
batch kernels. Fixed-seed training results therefore need not match an older
version bit for bit. The feature equations, class weighting, participant
splits, inner selection, and outer evaluation protocol remain the same.

## Exactly what is trained and tested

The cohort loader is unchanged: **304 participants, 276 PD and 28 ET**; labels
are **PD=0, ET=1**, selected by the published condition strings. Every
participant contributes one `(132,976)` movement array. The 132 channels are
11 task segments × 2 wrists × 2 sensor types × 3 axes. They are not 132
simultaneously recorded axes. No participant is split into independent cases.

The default outer/inner split counts are 5 and 3, with one repeat and seed
`20261017`, matching the existing evaluation. `splits.json` stores the exact
assignments. Optional `--baseline PATH` checks identities, original data
hashes, labels, alpha candidates, and every outer/inner split before training.
For a baseline saved with the earlier seven-alpha grid, supply the explicit
`--alphas` list shown above; use `--decision-rule zero` when reproducing its
fixed-zero classification procedure. Matching cohort/split metadata alone
does not make a threshold-tuned result the same procedure as that baseline.

### Frozen mode

1. Load the pinned pretrained checkpoint and disable gradients and dropout.
2. Extract independent participant representations once. No statistics from
   other participants are used by the encoder.
3. In each inner training fold, fit a fresh StandardScaler and train balanced
   ridge heads. The default alpha grid is all powers of ten from `1e-3`
   through `1e9` (13 values); the shared Gram decomposition serves every value.
   The earlier upper limit of `1000` was selected in all five saved TSPulse
   folds, so the expanded grid allows substantially stronger regularization.
4. With `inner-balanced`, select one common threshold for each alpha from its
   inner out-of-fold scores, maximizing mean inner-fold balanced accuracy.
   Choose the alpha with the highest resulting mean, favoring the larger
   alpha for ties within `1e-12`. With `zero`, evaluate every alpha at raw zero.
5. Fit a fresh scaler and selected head on the full outer training set and
   predict the outer test participants once, using the already selected threshold.

### Fine-tuned mode

The only supported training procedure is `reserved_validation_checkpoint_v1`.
The median-epoch full-training-pool refit and transfer of a raw threshold from
inner models have been removed. There is no legacy-procedure selector.

1. Reserve a stratified validation set from each outer training pool before
   learning-rate selection. `--validation-fraction 0.2` is the default; the
   remaining approximately 80% are fitting participants. The outer test stays
   separate. `finetune_splits.json` stores these same assignments for every model.
2. Make the inner CV folds using only fitting participants. Every inner fit
   starts from the pinned pretrained weights and a new common classification
   head. Train with AdamW and class-balanced binary cross entropy. Class weights
   `n/(2*n_class)` use that fit's gradient-training participants only. The head
   rate is `0.001 * min(1, 256 / D)` unless explicitly overridden. Backbone LR
   candidates remain `1e-5` and `3e-5`; weight decay is `0.01`, dropout `0.1`,
   gradient clipping `1.0`, and default patience `3`.
3. Each inner model selects its own earliest best validation epoch and its own
   decision threshold. Choose the LR by mean best inner-fold balanced accuracy;
   ties within `1e-12` favor the smaller LR. Inner thresholds and epoch counts
   are not transferred to the outer model. Discard inner models, retaining their
   predictions, history, and selection records without writing unused weights.
4. Initialize the outer model, train on fitting participants, and evaluate the
   reserved validation set each epoch. Stop after `patience` epochs without
   improvement, bounded by `--max-epochs`. Save each strictly better checkpoint
   and restore the earliest best one, including parameters and registered buffers.
5. Use that restored checkpoint's own validation scores to select its threshold.
   With `inner-balanced`, maximize validation balanced accuracy; `zero` retains
   the explicit raw-zero rule. Predict the outer test once. No additional
   training occurs after checkpoint/threshold selection, and outer-test labels
   never select any setting.
6. `--fit-final` follows the same rule using the entire cohort as the selection
   pool: reserve validation, tune LR within fitting participants, train with
   validation early stopping, restore the best checkpoint, and save its own
   threshold. Validation participants never enter weight updates.

Each validation set must contain both diagnoses, and the fitting set must have
at least `--inner-folds` participants per diagnosis. Infeasible combinations fail
before checkpoint downloads. `--check-data --mode both` checks these conditions.
The holdout reduces the number of participants used for gradient updates; for
5-fold outer CV and a validation fraction of 0.2, roughly 64% of the cohort
updates any one outer model's weights. Validation data selects the model and
threshold; only the outer test measures performance.

Restoration uses a temporary checkpoint file and memory-mapped CPU loading.
The retained model's weights, saved validation predictions, and decision cutoff
therefore refer to the same checkpoint. The temporary file is removed after
restoration. `--save-fold-models` additionally keeps the restored fold model.

For both modes, threshold candidates cover every distinct strict
`raw_score_et > raw_decision_threshold` partition: midpoints between unique
scores, finite outside boundaries, and zero. Tied scores stay together.
Threshold ties within `1e-12` favor the candidate closest to zero, then the
larger threshold. Frozen selection weights folds and classes equally. Neural
checkpoint selection balances the two classes in the reserved validation set. Constant scores keep threshold
zero rather than forcing either class. Selection statistics are optimistic
because they select settings; only outer predictions evaluate the procedure.
This follows the separation of model fitting and decision-threshold tuning
described in the [scikit-learn guide](https://scikit-learn.org/stable/modules/classification_threshold.html).

The rule is shared by TSPulse, UTICA, TiRex, MOMENT, Chronos-2, TiViT, VQShape,
and optional MantisV2. The CLI, `TrainOptions`, and foundation Ridge tuner all
default to `inner-balanced`; direct Python calls no longer default to raw zero.
The low-level tuner returns raw OOF scores and stores the selected threshold
in the selected alpha's search row. The comparison runner applies that
threshold to its output margins and persists it for inference.

The [all-model decision tests](tests/test_foundation_all_model_decisions.py)
exercise small actual architectures for all eight models: both nested modes,
fold/final saving, class imbalance, negative inner-validation scores, and
inference after shifting every raw score below zero along with the threshold.
Models declaring `inner-balanced` are rejected before encoder loading if their
raw threshold is missing. Earlier adapter-3 artifacts using the fixed-zero rule
retain their original decision rule; loading them does not select a threshold.
See the [current checkpoint validation record](../tmp/foundation_validation/validation_checkpoint_validation.json).
These tests use synthetic inputs and small random weights, not real-cohort
performance measurements or full-size pretrained-checkpoint validation.

TiRex's PyTorch recurrent path uses `exp(min(log_gate, 0))` in place of
`min(exp(log_gate), 1)`. The expressions have the same forward values, but the
latter can overflow before clipping and yield `0*inf` during backward. The
adapter preserves the native BF16 recurrent state/matmul and FP32 pointwise
equations, pretrained tensors, feature ordering, and nonfinite-gradient checks.
It changes the callback on loaded TiRex cells rather than editing site-packages.
The [released-checkpoint MPS check](../tmp/foundation_validation/tirex_released_gradient_validation.json)
reproduced 47 nonfinite gradient tensors with native arithmetic and zero with
the stable equivalent, with bit-identical forward features on generated signals.
The [current regression record](../tmp/foundation_validation/large_model_speed_regression.json)
also includes restart preservation and gradient-parity tests.

Fine-tuning adapts the backbone layers used by the selected representation.
Pretraining/forecasting heads and layers beyond the selected intermediate
readout do not contribute to its classifier loss. The code verifies that
nonzero gradients actually reach backbone parameters, and refuses detached
inference-only feature paths. It does not call head-only training fine-tuning.

`--accumulation-steps` combines microbatches before each optimizer step. Each
accumulation window is normalized by its actual participant count, including
the final incomplete window. It does not treat channels as labeled examples.

The frozen and fine-tuned modes intentionally have different classifier
training procedures. Compare backbones **within each mode**; a difference
between modes also includes the change from ridge to the neural head.

### Interpreting repeated outer-fold BA of 0.5000

Balanced accuracy is `(PD recall + ET recall) / 2`. Predicting every participant
as PD gives recalls `(1,0)` and BA `0.5`; predicting everyone as ET gives
recalls `(0,1)` and the same BA. This is an actual classification result even
when the scores vary between participants or their AUROC exceeds `0.5`.
Class-weighted fitting does not guarantee useful class decisions at raw zero.
In the saved TSPulse frozen run, folds 3 and 4 predicted zero ET participants
at raw zero despite AUROC of about `0.761` and `0.961`. The inner validation
scores also shifted toward negative values. The new default selects the
threshold on inner validation predictions, then fixes it before outer testing.
This changes the evaluation procedure; it does not establish improved PADS
accuracy until a new outer evaluation is run. A validation-selected threshold
can still yield a single predicted class on unseen participants.

The earlier fine-tuning default used head AdamW rate `0.001` for every feature
width. With many correlated feature coordinates, Adam can move a score by
order `learning_rate * D` in a single update. Global gradient clipping does
not bound those adaptive coordinate steps. Rare ET participants in small
accumulation windows add variability despite correct class weighting. The
automatic width scaling above limits this specific source of instability;
it is a stability default, not a guarantee of predictive performance.

Outer-fold console lines now include both class recalls, AUROC, predicted ET
count, and score range, with an explicit message for a single predicted class.
Every completed fold writes `score_diagnostics.json`, `threshold_selection.json`,
and `metrics_raw_zero.json` alongside the selected-rule `metrics.json`.
Prediction CSVs record `raw_score_et`, `raw_decision_threshold`, the adjusted
`decision_score_et`, and its `decision_threshold=0`. Thus
`predicted_et = (raw_score_et > raw_decision_threshold) = (decision_score_et > 0)`.
Saved fold/final models carry the same threshold, and prediction subtracts it
exactly once. Fine-tuned folds also save `selection_split.json`,
`validation_predictions.csv`, and `inner_fold_selections.json`. Their
`training_configuration.json` records fitting/validation IDs, best epoch, actual
epochs trained, and checkpoint-restoration status, together with actual optimizer rates,
feature width, training class counts, and class weights. Inner history files
record both selected-threshold and raw-zero BA/counts, raw score ranges, and
the per-epoch threshold. Epoch selection uses the earliest best inner BA
under the chosen rule. Outer/final training restores its own best validation
checkpoint and keeps that checkpoint's validation threshold.

To use the new automatic rate, omit `--head-learning-rate` from an older
command. An already running Python process retains its imported code and
parsed settings; start a new process for the new default. Existing checkpoints
must carry adapter-version-3 metadata to be used by the current prediction
code; older artifacts require a new fit with the current processing.
The [head stability tests](tests/test_foundation_head_stability.py) cover
coherent large-width Adam updates and learning a held-out synthetic signal
with imbalanced labels. They do not establish real-cohort accuracy.

## Model-specific representation choices

| Name | Representation and adaptation |
| --- | --- |
| `tspulse` | Classification checkpoint revision `tspulse-block-dualhead-512-p16-r1`, pinned to its commit. Concatenate mean time-patch, FFT-patch, and register representations. Disable pretraining masking in both modes. The pretrained backbone is adapted directly, rather than reproducing the paper's TSLens recipe. |
| `utica` | MantisV1 backbone, layer index 2, combined CLS and mean patch tokens. Explicitly exclude only the released pretraining mask token/final normalization, and remove its absent, unused projection head. All feature-path tensors load strictly. |
| `tirex` | Native context preparation/padding/truncation; per-layer mean context features, L2-normalized per layer. Features are ordered by layer, channel, hidden coordinate and receive final participant LayerNorm, matching `TiRexEmbedding`. Use the differentiable core behind `_embed_context`. Optional differenced/statistics augmentation is disabled. |
| `moment` | MOMENT-1-base embedding mode; concatenate channels. Short inputs are left-padded with an observed mask, long inputs use endpoint-preserving interpolation, and native masked patch-mean pooling excludes padding. |
| `chronos2` | Native-length context. Use the differentiable encoder, pool context patches, and exclude REG/future output tokens from that pooling. Group attention links the 12 axes within a task only. Different tasks and participants always have different groups. |
| `tivit` | Official TiViT transformation, stride 0.1, first 14 CLIP ViT-H blocks, mean image tokens. Signal patches use `floor(sqrt(input_length))` (31 at length 976), and the complete channel-concatenated participant vector is L2-normalized. |
| `vqshape` | The released 256-dimensional/512-code model. Flatten the published shape-token representation, including its straight-through code vectors and shape attributes. Histogram counts are not used because they cannot train the encoder through classification gradients. |
| `mantis_v2` | Optional reference: layer index 2 and combined CLS/mean patch tokens, matching the existing frozen experiment. |

Channel order is preserved. TiRex follows the author's layer-major layout,
with channels in their original order inside each layer; the other adapters
concatenate complete per-channel vectors.
Mantis/UTICA, TSPulse, and VQShape use linear interpolation from 976 to
512 points with `align_corners=False`, matching the earlier Mantis adapter.
MOMENT uses `align_corners=True` for long inputs, preserving endpoints;
short inputs are left-padded with an observed mask, and exactly 512 samples
pass through unchanged. Native RevIN is retained without extra dataset-fitted
input normalization. Fewer than eight observed samples are rejected because
native mean pooling needs at least one fully observed patch.
This interpolation has **no antialias filter** and can affect high-frequency
content. TiRex, Chronos-2, and TiViT take the original 976 points. Consequently,
the comparison evaluates each complete adapter/model pipeline, including its
time representation, rather than an identical sampling procedure for every
backbone. These choices are fixed in advance and recorded in the manifest.

The [original-specification audit](FOUNDATION_ORIGINAL_SPEC_AUDIT.md) documents
the checkpoint, preprocessing, readout, and training-recipe checks for all eight
models. The implementation contains only the corrected adapter-version-3
processing described above. There is no adapter-version selector. Model
metadata records the fixed version, and prediction rejects older versions or
missing version metadata before loading a model. Matching feature widths do
not make older heads compatible with current feature values or ordering.
Existing result folders are retained; using an older head requires a new fit
with the current code. VQShape loading also
requires all inference architecture metadata, including attention-head counts;
it does not guess missing settings from tensor shapes.

This is a shared-classifier backbone comparison. It does **not** reproduce
each model's original classification training recipe. In particular, the
TSPulse pooling/masking/head recipe differs from TSLens; Chronos-2 has no
native classification head; full-backbone fine-tuning of TiRex/TiViT/VQShape
is an extension of their frozen-feature recipes. MOMENT adopts the author
input-sizing mechanics but does not copy dataset-specific external z-scoring
or cohort input scalers. The fine-tuning head and
width-scaled learning-rate default are also comparison choices, not author
defaults. Treat results as performance of the adapters defined here.

Model packages, checkpoint identities, file hashes, extracted feature hashes,
and source hashes are recorded. The two small vendored upstream components
(TiViT and VQShape) retain their MIT licenses and exact source provenance in
`foundations/_vendor/PROVENANCE.json`. VQShape has only a relative-import change.
Consult the upstream model licenses when reusing weights; TiRex uses the NXAI
Community License.

## Results and final prediction

The root `comparison.csv` reports each model/mode's repeat-pooled and mean-fold
metrics side by side. Each model/mode directory contains outer predictions,
fold metrics, summaries, and search/training diagnostics. The primary metric
is balanced accuracy. AUROC, ET average precision, ET/PD recall, precision, F1,
macro F1, MCC, and ordinary accuracy are also saved. Positive scores mean ET;
scores exactly zero are PD. Scores are margins, not calibrated probabilities.

With `--fit-final`, the frozen final model is
`MODEL/frozen/final_model.joblib`. The fine-tuned final model is
`MODEL/finetune/final/model.json` plus `model_state.pt`. Final frozen fits use
all participants. Final neural fits reserve validation
participants from the cohort and retain their selected checkpoint. Neither has
an independent test set; their evaluation is the saved outer
cross-validation. Fold weights are saved only with `--save-fold-models`.

To predict already-preprocessed new participant arrays, set
`FOUNDATION_RUN_DIR` to the results directory printed by your run:

```sh
FOUNDATION_RUN_DIR="artifacts/foundations_both/REPLACE_WITH_PRINTED_RUN_ID"

.venv-foundations/bin/python -m sota_pd_et.predict_foundation \
  --model "$FOUNDATION_RUN_DIR/tspulse/frozen/final_model.joblib" \
  --inputs new_participants.npy --output predictions_frozen.csv --device auto

.venv-foundations/bin/python -m sota_pd_et.predict_foundation \
  --model "$FOUNDATION_RUN_DIR/tspulse/finetune/final/model.json" \
  --inputs new_participants.npy --output predictions_finetuned.csv --device auto
```

The input shape must be `(N,132,976)` with the same upstream preprocessing,
units, and channel order. Inference verifies recorded checkpoint hashes and
the fine-tuned state hash, and refuses to overwrite an output CSV. Restore the
recorded training package versions for consistent inference. Load only your
own joblib bundles, because joblib uses Python pickle serialization.

Official sources: [TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1),
[UTICA](https://github.com/fegounna/Utica),
[TiRex](https://github.com/NX-AI/tirex),
[MOMENT](https://github.com/moment-timeseries-foundation-model/moment),
[Chronos](https://github.com/amazon-science/chronos-forecasting),
[TiViT](https://github.com/ExplainableML/TiViT),
[VQShape](https://github.com/YunshiWen/VQShape).
