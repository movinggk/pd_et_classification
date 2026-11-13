# Foundation comparison performance and verification

The changes remove repeated Ridge factorizations, unused pretrained prediction
work, redundant host transfers, and repeated checkpoint metadata work. The
measurements below use synthetic signals or feature matrices. **No full PADS
benchmark or new comparison of clinical classification accuracy was run.**
The timings measure individual operations; they do not establish an overall
experiment speedup and must not be multiplied together.

The current neural protocol restores its own best validation checkpoint from
a temporary file; discarded inner models do not write checkpoint weights.
Historical fixed-epoch training timings below do not measure this added
checkpoint I/O.

The latest large-model changes cache only MOMENT's fixed integer T5 position
buckets, keep TiViT checkpoint inputs as single-channel uint8 images, move
weight-independent preparation outside activation recomputation, and pack
independent Chronos task groups. Native model attention, scaling, readout,
and trainable weights are retained. Chronos now defaults to 24 channels
(two 12-variate tasks); the other large models retain chunk 8. Per-model
batching and checkpointing flags work on both new runs and incomplete-model
restarts. See the [runtime guide](FOUNDATION_COMPARISON_README.md#runtime-settings-for-the-other-large-models)
for flags and the image-cache memory tradeoff.

The [latest MPS benchmark](../tmp/foundation_validation/large_model_speed_benchmark.json)
and its [reproduction script](../tmp/foundation_validation/benchmark_large_models.py)
use `(1,24,976)` generated signals, one warmup, and three timed feature
forward/backward calls per case. No optimizer updates are made. VQShape uses
its released cached weights; MOMENT, Chronos-2 and TiViT use explicitly recorded
smaller upstream architectures with random weights.

| Model and benchmark scope | Previous chunk 8, checkpointed | Optimized chunk 24, checkpointed | Optimized chunk 24, no checkpointing |
| --- | ---: | ---: | ---: |
| MOMENT, smaller architecture | 51.60 ms | 49.01 ms (1.05x) | 36.32 ms (1.42x) |
| Chronos-2, smaller architecture | 60.02 ms | 54.10 ms (1.11x) | 40.17 ms (1.49x) |
| TiViT, smaller architecture | 156.16 ms | 147.50 ms (1.06x) | 111.23 ms (1.40x) |
| VQShape, released weights | 161.58 ms | 148.79 ms (1.09x) | 104.07 ms (1.55x) |

At unchanged chunk 8 the automatic shortcuts measured approximately 1.00x,
1.00x, 1.05x, and 1.03x respectively; most additional gain comes from batching
and removing recomputation. These measurements do not establish speedups on
the released MOMENT/Chronos/TiViT backbones or full CV runs. Disabling
checkpointing and increasing chunks consume more activation memory.
All evaluated native features at chunk 8 matched bit for bit and all measured
gradients were finite. The [current regression](../tmp/foundation_validation/large_model_speed_regression.json)
reports 124 passing tests, no failures/errors/skips, including MPS gradient
comparisons and safe completed-model preservation. CPU MOMENT gradients matched
exactly; MPS embedding reductions and packed-group gradients are checked with
explicit floating-point tolerances in the [tests](tests/test_foundation_large_model_speed.py).
The timings later in this document are older component measurements.

TiRex additionally avoids retaining a broadcast recurrent-weight copy for
every series at every timestep. Its native forward and BF16 state precision
are retained; an analytic headwise backward reduces autograd storage. The
[current MPS batching measurements](../tmp/foundation_validation/tirex_batching_benchmark.json)
and [comparison guide](FOUNDATION_COMPARISON_README.md#tirex-throughput-on-mps)
describe this change separately from the older timings below.

The one-epoch fit timings and 48-test regression record below were recorded
before the subsequent head-stability fix. The current automatic head rate
is width-scaled; see the [comparison guide](FOUNDATION_COMPARISON_README.md#interpreting-repeated-outer-fold-ba-of-05000)
and [stability tests](tests/test_foundation_head_stability.py). Those earlier
timings remain historical component measurements at their original settings.
They also precede the participant normalization/order corrections for TiRex
and TiViT; see the [specification audit](FOUNDATION_ORIGINAL_SPEC_AUDIT.md).
The implementation now contains only adapter-version-3 processing. Historical
timings do not imply that an older adapter setting remains available.

The validation environment was macOS 26.6 on Apple arm64, Python 3.12.13,
PyTorch 2.11.0, and the pinned [foundation requirements](requirements-foundations.txt).
CPU microbenchmarks limited PyTorch and, where applicable, BLAS to two threads.
The experiment CLI independently defaults to `--threads 4`.

**CPU measurements.** The neural architecture tests use the actual upstream
implementations with small randomly initialized configurations from
[foundation_fixtures.py](tests/foundation_fixtures.py), rather than released
full-size weights. Ridge uses synthetic extracted features. A latency is the
median of warmed wall-clock measurements; model loading and data preparation
are excluded unless stated otherwise.

| Operation and measurement scope | Before | After | Measured ratio |
| --- | ---: | ---: | ---: |
| Balanced Ridge inner search: 150 participants × 4,096 features, PD=135/ET=15, three folds, seven alphas, two BLAS threads | 0.0512832 s | 0.0227751 s | 2.25× |
| VQShape frozen feature forward: eight series × 512 points, small actual architecture | 11.5290 ms | 0.7733 ms | 14.91× |
| TiViT frozen feature forward: eight series × 976 points, small actual architecture | 13.1305 ms | 3.4988 ms | 3.75× |
| TiRex frozen feature forward: eight series, configured context 64, small actual architecture | 1.3947 ms | 1.3332 ms | Approximately 1.05× |
| TSPulse hidden-state forward: small RevIN-affine architecture, three signals × 512 points × one channel, two threads | 0.0005520 s | 0.0004472 s | 1.23× |

The [Ridge implementation](foundations/ridge.py) computes one eigendecomposition
of the smaller weighted primal or dual Gram matrix per training fold and reuses
it across the alpha grid. StandardScaler remains fitted on each inner training
fold only. The balanced sample weights are \(n/(2n_c)\), where \(n_c\) is the
number of training participants in class \(c\). Weighted centering preserves
the intercept. Numerically delicate cases and near-zero decision margins use
the reference scikit-learn solver. The current default searches 13 powers of
ten from `1e-3` through `1e9` and jointly selects an inner-OOF decision threshold.
The existing profile below used the earlier seven-alpha grid and fixed raw zero;
it measures the linear-algebra optimization, not the new selection procedure.
In the 150 × 4,096 profile, the selected
alpha was 1,000 with identical search metrics and an OOF-margin maximum
absolute difference of \(3.22\times10^{-15}\). Four timed searches were run for
each implementation; the first was excluded from the reported median. The
[Ridge profile](../tmp/foundation_validation/performance/ridge_search.json)
retains the raw timings; the original synthetic matrix/split seeds were not
retained. The [CPU feature-path profile](../tmp/foundation_validation/performance/feature_paths_cpu.json)
records exact small configurations and uses three warm-ups followed by the
median of five groups of calls: 12 calls per group for VQShape/TiRex and ten
for TiViT. Its raw timing arrays and the exact TiRex raw input length were not
retained; the latter is explicitly unknown, rather than assumed to equal its
configured context length.

The [feature paths](foundations/fast_paths.py) retain VQShape's shape tokens and
straight-through gradients while omitting its unused reconstruction and
pretraining outputs; retain TiRex's intermediate layers while omitting its
forecast head; and retain TiViT's quantized image representation while replacing
repeated PIL/NumPy processing with a tensor lookup. TiViT falls back to the
upstream processor for unsupported settings. The
[TSPulse hidden-state path](foundations/tspulse_fast.py) removes a second FFT
and scaling pass used only for reconstruction targets. It retains the FFT of
the correctly scaled input, including the trainable RevIN affine transform,
and falls back for unverified versions/configurations. Its
[profile](../tmp/foundation_validation/tspulse_hidden_performance_validation.json)
used 25 warmed calls per path and confirmed that FFT calls fell from two to one.

**Released TSPulse checkpoint on MPS.** These measurements use the pinned
`ibm-granite/granite-timeseries-tspulse-r1` revision
`66cfb5464ed5e5a5cd7f5df1339af4e51461d955`, rather than a small test architecture.
The table's shapes are `(participants, channels, original time points)`;
the adapter resamples each channel to the checkpoint's 512-point context.

| Workload and setting | Measured latency |
| --- | ---: |
| Frozen extraction, synthetic 4 × 132 × 976, channel chunk 8 | 0.54035 s |
| Same frozen extraction, channel chunk 128 | 0.09145 s |
| One training epoch, synthetic 8 × 132 × 976, PD=6/ET=2, chunk 8, checkpointing enabled | 3.58074 s |
| Same epoch, chunk 128, checkpointing enabled | 1.02355 s |
| Same epoch, chunk 128, checkpointing disabled | 0.77352 s |

Frozen extraction timings are medians of three measurements in the
[MPS channel profile](../tmp/foundation_validation/performance/channel_batch_mps.json).
They enclose `extract_features` with participant batch size 2, including input
and output transfers but excluding encoder construction.
The [CPU channel profile](../tmp/foundation_validation/performance/channel_batch_cpu.json)
on the same released checkpoint and input measured 0.27820 s at chunk 8 and
0.15714 s at chunk 128. The
[MPS training profile](../tmp/foundation_validation/performance/tspulse_mps_training.json)
ran three fresh one-epoch fits per setting and reports the median of the last
two fits. Each timing encloses `fit_classifier`, including a fresh cached
backbone load, common-head setup, optimizer setup, and one training epoch;
prediction is excluded. **Those channel/checkpointing profiles precede the additional
TSPulse FFT shortcut.** Their time differences must not be attributed to that
shortcut. Larger chunks reduce repeated dispatches; disabling activation
checkpointing removes recomputation and increases activation memory.

A separate [TiViT MPS profile](../tmp/foundation_validation/performance/tivit_mps.json)
uses eight channel series × 976 points, the small actual architecture, and
`PYTORCH_ENABLE_MPS_FALLBACK=1`. It measured 15.4533 → 4.5576 ms for a frozen
forward and 21.9296 → 9.0839 ms for forward/backward. Its
[reproduction script](../tmp/foundation_validation/performance/verify_tivit_mps.py)
warms up three calls and takes five samples of ten calls, synchronizing MPS
before and after the timed sample. These are model-operation measurements,
not complete participant-fold training times.

**Training, loading, and memory.** The
[training changes](foundations/training.py) collect prediction margins on the
device before one host transfer, reuse safe NumPy batch storage, compute the
head dimension from model configuration, reduce scalar transfers, and retain
finite-gradient checks before every optimizer step. Epoch-history collection
transfers small values once per epoch and preserves the previous summation
order; live loss values refresh when the progress bar is due. On
the small actual TSPulse architecture with synthetic 16 × 4 × 64 input,
one epoch, batch size 2, accumulation 4, and two CPU threads, the final
[training profile](../tmp/foundation_validation/training_performance_validation.json)
reduced profiled scalar extractions from 184 to 89. Separately warmed fits
had medians 0.0511011 → 0.0497515 s over seven samples; the measured CPU gain
was modest. Profiler startup wall times are not performance comparisons.

The [encoder loader](foundations/encoders.py) caches pinned Hub resolution,
package-version metadata, and checkpoint hashes within a process. Hashes are
invalidated by file identity/stat changes, and missing cached files are
resolved again. Every fit still creates fresh model parameters and restores
the pretrained weights; trained parameters are never reused across folds.
Frozen feature extraction preallocates its destination rather than keeping
all batches plus a second concatenated feature array in memory.

**Correctness and reproducibility boundaries.**

The [historical regression record](../tmp/foundation_validation/performance/regression.json)
reports 48 tests passed for the earlier implementation and retains its
source-file SHA-256 values. The current 124-test record is linked above.

- [Fast-path tests](tests/test_foundation_fast_paths.py) compare actual upstream
  outputs, input/parameter gradients, and checkpoint state layouts. Small
  CPU VQShape/TiRex/TiViT tests match retained outputs and active gradients
  exactly. The TiViT MPS profile also records exact tested pixels, outputs,
  and active gradients.
- [TSPulse tests](tests/test_foundation_tspulse_fast.py) cover standard and
  affine RevIN scaling, raw/scaled FFT inputs, seeded dropout, every active
  parameter gradient, and unsupported-layout fallback. The
  [released-checkpoint MPS check](../tmp/foundation_validation/performance/tspulse_mps_integrated.json)
  matched upstream outputs and input gradients exactly, matched all 137
  active parameter-gradient tensors exactly, and passed a one-epoch
  imbalanced synthetic classifier smoke check.
- [Ridge tests](tests/test_foundation_ridge.py) cover imbalanced primal/dual
  systems, rank deficiency, zero margins, numerical fallback, selected
  alpha, metrics, and OOF scores.
- [Training tests](tests/test_foundation_training_performance.py) cover unequal
  class counts, accumulation with an incomplete final effective batch,
  nonfinite gradients on later updates, nonzero subnormal gradients, and
  prediction transfer count. The recorded small CPU training comparison
  retained bitwise-identical parameters and loss histories.
- [Loading tests](tests/test_foundation_loading_cache.py) verify metadata reuse,
  rewrite/replacement invalidation, deleted-cache recovery, independent
  metadata dictionaries, and fresh pretrained parameter objects after a
  previous loaded model was modified.
- VQShape's omitted decoder consumes dropout randomness in the upstream
  training path. Removing it changes subsequent RNG consumption. Changing
  channel chunk size can also change dropout assignment and floating-point
  reduction order. Therefore, retained feature/gradient equivalence for a
  controlled call does **not** promise a bitwise-identical full training
  trajectory across these settings. Record settings with each run and
  compare predictive performance using the saved common folds.

The [CLI](compare_foundations.py) resolves chunk 128 and checkpointing disabled
for TSPulse, chunk 24 for Chronos-2, and chunk 8 for the other models.
Checkpointing is enabled by default except for TSPulse. Override these using
`--model-channel-batch-size MODEL=N` and
`--model-gradient-checkpointing MODEL=on|off`, each repeatable. The global
`--channel-batch-size N`, `--gradient-checkpointing`, and
`--no-gradient-checkpointing` flags remain available on new runs. Resolved settings
are saved in each run manifest. `--device auto` selects CUDA when available
and otherwise CPU; request `--device mps` explicitly on this Mac. These
changes retain class-balanced Ridge/BCE, inner-only model selection, and unique
run folders. Per-fold metrics and predictions are saved after each outer
fold; fine-tuned fold weights require `--save-fold-models`. Larger models may
require smaller chunks and checkpointing to fit device memory.

Run the focused correctness checks from the project root using the pinned
environment; this command uses synthetic fixtures and local test checkpoints:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-foundations/bin/python -m unittest \
  sota_pd_et.tests.test_foundation_fast_paths \
  sota_pd_et.tests.test_foundation_tspulse_fast \
  sota_pd_et.tests.test_foundation_ridge \
  sota_pd_et.tests.test_foundation_training_performance \
  sota_pd_et.tests.test_foundation_loading_cache -v
```

See the [comparison guide](FOUNDATION_COMPARISON_README.md) for the complete
test command, experiment CLI, and saved artifacts. Synthetic timings depend
on model size, channels, sequence length, warm-up, device, and concurrent
work. The reported ratios apply to the stated workloads only.
