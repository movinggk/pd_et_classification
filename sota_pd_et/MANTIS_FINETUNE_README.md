# Participant-level MantisV2 fine-tuning for PD versus ET

This adds a trainable MantisV2 backbone to the existing PADS pipeline. The
implementation was tested using **synthetic participants/signals only**,
including gradient and serialization checks using the actual local pretrained
weights. **No real PADS fine-tuning/evaluation was run during implementation.**

The user subsequently completed the 132-channel `head-only` run
`20261109T032315873662Z` and `last-block` run `20261109T041659480074Z`.
Head-only gives **BA 81.57%, ET recall 26/28, PD specificity 70.29%, PR-AUC
0.6391, MCC 0.3815**. Last-block gives **BA 76.76%, ET recall 16/28, PD
specificity 96.38%, PR-AUC 0.6463, MCC 0.5534**. PR-AUC is trapezoidal;
AP is separately 0.6444 and 0.6505. Last-block does not improve BA over
head-only; it reduces PD errors while missing more ET. The new last-block run
exactly reproduces the earlier `20261108T145903609065Z` outer predictions and
saved model tensors on the same seed/splits, rather than adding independent
test participants. The [revised technical report and saved-result audit](../MantisV2_FineTune_Report/README.md)
document both modes, class imbalance, fold results and limitations. Report
generation does not retrain or perform model inference. The frozen MantisV2
and temporal/cross-task experiments remain separate.

## 1. Files and environment

- `mantis_finetune.py`: trainable participant classifier, optimizer path,
  gradient accumulation, checkpoint serialization and restoration.
- `evaluate_mantis_finetune.py`: participant-level nested selection/evaluation,
  optional paired frozen comparison and final full-cohort fit.
- `predict_mantis_finetune.py`: inference from a self-contained trained bundle.
- `tests/test_mantis_finetune.py`, `tests/test_mantis_finetune_protocol.py`:
  synthetic-only tests, with additional local-checkpoint/hardware tests when available.
- `requirements-mantis-finetune.txt`: dependencies using the existing pinned
  `mantis-tsfm==1.1.0` adapter.

Use the project's `.venv-transformer` environment, or create a Python 3.12
environment from the project root:

```sh
python3.12 -m venv .venv-mantis-finetune
.venv-mantis-finetune/bin/python -m pip install --upgrade pip
.venv-mantis-finetune/bin/python -m pip install -r sota_pd_et/requirements-mantis-finetune.txt
.venv-mantis-finetune/bin/python -m pip check
```

Substitute the new Python path in the examples below. CUDA requires a suitable
CUDA-enabled PyTorch installation. `--device auto` selects CUDA, then Apple MPS,
then CPU. Explicitly unavailable devices fail with a clear error.

Tests, without running a PADS experiment:

```sh
.venv-transformer/bin/python -m unittest \
  sota_pd_et.tests.test_mantis_finetune \
  sota_pd_et.tests.test_mantis_finetune_protocol -v
```

The actual-weight tests use generated signals. They skip if the pinned checkpoint
is absent locally and do not download it. Numerical GPU tests skip when their
hardware is unavailable. The toy nested evaluator test fabricates its entire
cohort and does not call the actual PADS data loader.

## 2. Run the default 132-channel experiment yourself

From the project root:

```sh
.venv-transformer/bin/python -m sota_pd_et.evaluate_mantis_finetune \
  --input-set all \
  --mode last-block \
  --device auto \
  --compare-to artifacts/pads_mantis_v2 \
  --fit-final
```

This uses the same 132 official preprocessed channels as the completed frozen
MantisV2 experiment. `--compare-to` is optional; omit it if you do not have that
completed baseline directory. The default split seed and split-generation
function match the existing experiment. When the original baseline split file
is present and the default 5/3/1/seed configuration is used, exact split equality
is checked automatically. `--reference-splits path/to/splits.json` explicitly
requires a particular saved plan.

Results go to a new timestamped folder under
`artifacts/pads_mantis_v2_finetuned/`. You can set a new directory with `--output`.
An existing output directory, source/data directory, pretrained checkpoint
directory or subdirectory of an existing experiment is rejected.
The evaluator does not automatically resume a partial run. Completed fold files
remain available on failure, and `manifest.json` records the failure. Use a new
output directory when starting again.

`--fit-final` performs a separate inner search and final fit on all participants
after outer evaluation. This final model's training data performance is not a
new independent evaluation. `--save-outer-models` optionally saves each outer
training model as well.

## 3. Inputs and participant labels

The checksum-verifying existing loader selects 276 PD and 28 ET participants
using the exact published diagnosis strings. PD is `0`, ET is `1`. Original
numeric label `2` includes other differential diagnoses and is not an ET-only
label. Participant IDs, diagnoses/free text, demographics, medication and
questionnaires are excluded from the signal features.

`--input-set all` uses `(N,132,976)`: 11 retained task segments × two wrists × two
sensor types × XYZ. This is the default because it matches the original frozen
MantisV2 comparison.

`--input-set three_tasks` selects the same 60 channels as the newer cross-task
trial: `Relaxed1`, `Relaxed2`, `RelaxedTask1`, `RelaxedTask2`, `StretchHold`, each
with 12 physical channels. For a matched comparison to that trial:

```sh
.venv-transformer/bin/python -m sota_pd_et.evaluate_mantis_finetune \
  --input-set three_tasks \
  --mode last-block \
  --device auto \
  --compare-to artifacts/pads_temporal_cross_task/20261108T071720522119Z \
  --fit-final
```

`--compare-to` also accepts that run's `mantis` subdirectory. It checks the
cohort/order, data hashes, exact inner/outer splits, selected input channels,
pretrained representation, zero decision threshold and outer-prediction coverage.
Comparing the 60-channel fine-tuner to the old 132-channel baseline is rejected.
Changing task input and changing model training are distinct comparisons.

The official author preprocessing is retained. Each independent channel is
resized from 976 samples to 512 by linear interpolation with
`align_corners=False`, matching the previous frozen adapter. No additional
per-record z-score is applied to raw input. Time is not concatenated across
channels/tasks. This does not repair gaps in the original raw timestamps.

## 4. What is actually fine-tuned

Initialization is the same immutable pretrained checkpoint as the frozen trial:

- Model: `paris-noah/MantisV2`.
- Revision: `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`.
- Weights SHA-256:
  `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`.
- Representation: Transformer layer index `2` (third block), combined class
  token and mean patch token, 512 features per channel.

The complete initial checkpoint is loaded strictly and checksum-verified before
the three later, unused Transformer blocks are removed. The unused pretraining
projector stays frozen. The original cached checkpoint is never trained in place.

| `--mode` | Parameters receiving gradients after warmup |
|---|---|
| `last-block` (default) | Third/last **active** Transformer block and the new participant head |
| `full` | Token generator, class token, all three active Transformer blocks, and participant head |
| `head-only` | Participant head; the encoder is permanently frozen |

`full` refers to the entire forward path through the chosen third-layer
representation. Later blocks are outside this representation's forward path.
The choice of representation is fixed in this implementation so that fine-tuning
starts from the same representation as the frozen comparison.

The default last-block adapter trains **658,176 encoder parameters**. The
132-channel linear head has **67,585 parameters**, for **725,761 trainable
parameters** after warmup. Exact counts are written to the run manifest.

The head receives one ordered concatenation per participant: 67,584 features
for all channels or 30,720 for three conditions. It applies feature
standardization, dropout 0.1 and one linear output. A channel is not treated as
an independent patient, and channel labels are not repeated as separate examples.

## 5. Fold-local normalization and training

The initial pretrained encoder is first used in a completely frozen state to
extract initial features `Z0`. Since this operation is externally pretrained and
does not fit PADS data, the initial cache may be shared across folds.

Inside **each inner/outer/final training fit**, a mean and population standard
deviation are fitted to that fit's training rows of `Z0`. Initial standard
deviations below `1e-6` use scale 1. These statistics are fixed during adaptation
and saved as model buffers. Validation/test participants never fit the statistics.

During training, the model recomputes embeddings from **raw participant signals**
using the current trainable encoder and applies the fixed fold-specific
statistics. It does not train on cached adapted embeddings, reuse adapted
encoders between folds, or fit one global feature scaler on the entire cohort.

The participant loss is class-weighted BCE with logits. Each fit recomputes
`w_c = n_train/(2*n_class)` from its own training participants. AdamW uses
separate parameter groups for encoder and head. Default settings:

| Setting | Default |
|---|---|
| Encoder learning-rate candidates | `1e-5`, `3e-5` |
| Head learning rate | `1e-4` |
| Weight-decay candidates | `0.01` |
| Candidate total epochs | `3`, `5`, `10` |
| Initial head-only warmup | 1 epoch, included in the total |
| Participant microbatch | 2 |
| Accumulation steps | 8; normally 16 participants per optimizer step |
| Encoder channel chunk | 16 independent channel records per call |
| Gradient norm limit | 1.0 |
| Arithmetic | float32; no AMP |

Frozen blocks stay in evaluation mode during last-block training, while the
trainable last block uses training mode. During warmup the whole encoder runs
without gradients in evaluation mode. The head remains trainable. In head-only
mode encoder learning-rate/warmup settings do not adapt the encoder.

Gradient accumulation divides each microbatch's summed weighted loss by the
**actual participant count in its optimizer group**. This handles the final
incomplete group without incorrectly dividing it by a full group size.

Activation checkpointing is enabled by default and recomputes encoder chunks
during backward. It reduces retained activations at extra computational cost.
`--no-activation-checkpointing` disables it. Reducing `--channel-batch-size` helps
bound temporary memory, but chunking alone does not discard all backward graphs
when checkpointing is disabled. Device/version/dropout behavior can affect
numerical reproducibility; no wall-clock or GPU-speedup claim is made.

Validation uses `torch.no_grad()` rather than `inference_mode()` so that cached
upstream rotary tensors remain usable by later gradient-enabled epochs.

## 6. Nested selection and evaluation

All channels, tasks and internal encoder records from a participant stay in the
same fold. Defaults are five stratified outer folds, three stratified inner
folds, one repeat, seed `20261017`.

For every outer training set:

1. Start every inner fit from fresh copies of the pinned pretrained weights and
   a fresh participant head.
2. Fit initial-feature statistics using only that inner training subset.
3. Train one trajectory per encoder-LR/weight-decay pair, saving validation
   predictions at the candidate epochs.
4. Select by maximum **mean inner-fold balanced accuracy**. Ties within `1e-12`
   prefer stronger weight decay, then fewer epochs, then lower encoder LR.
5. Start another fresh model on the entire outer training set, fit its own
   statistics, and train for the selected **fixed total epoch count**.
6. Predict outer test participants once. Outer-test labels/loss never select
   epochs, stopping, normalization, learning rates or thresholds.

The ET decision is fixed before evaluation: **ET iff logit > 0; otherwise PD**.
Balanced accuracy is the mean of ET sensitivity and PD specificity. AUROC and
AP use continuous logits. Scores are not calibrated clinical probabilities.
Also inspect fold-level AUROC/AP because pooled scores come from different fits.

The optional comparison to a frozen ridge head is descriptive: it compares
different training procedures, not a controlled change in the encoder alone.
After previous inspection of this small cohort, this follow-up remains
exploratory. More CV repeats do not increase the number of independent ET cases.

## 7. Saved outputs

| Output | Contents |
|---|---|
| `manifest.json` | Status, configuration, package versions, source/data hashes and parameter counts |
| `cohort.csv`, `splits.json` | Participant identities/labels and all inner/outer assignments |
| `initial_checkpoint/` | Verified copies of the initial pinned config/weights |
| `initial_frozen_features.npy` | Initial frozen embeddings, used only for training-local statistics |
| `initial_encoder_provenance.json` | Exact initial representation and cache hash |
| `repeat_XX_fold_XX/inner_search.csv` | Training-only candidate selection results |
| `repeat_XX_fold_XX/inner_oof_predictions.csv` | Selected inner validation predictions |
| `repeat_XX_fold_XX/test_predictions.csv` | Outer held-out participant predictions |
| `repeat_XX_fold_XX/training_history.csv` | Training weighted BCE and warmup flag per epoch |
| `oof_predictions.csv`, `fold_metrics.csv` | Complete outer evaluation |
| `summary.json`, `REPORT.md` | BA, AUROC/AP, ET/PD metrics and interpretation |
| `frozen_comparison.json` | Optional matched frozen versus adapted comparison |
| `final_model.pt` | Optional full-cohort fine-tuned model; self-contained for inference |
| `final_inner_splits.json`, `final_selected.json` | Separate final-fit selection |
| `evaluated_sources.zip` | Exact Python source snapshot for the run |

An outer model is additionally saved under its fold when `--save-outer-models`
is enabled. Training histories are stochastic batch losses, not independent
generalization estimates. The final model is trained on all participants and
must not be evaluated as if those participants were new test cases.

### Progress bars

Progress bars are enabled by default for the fine-tuning command:

- **Initial MantisV2 features:** independent channel records encoded, including
  the final partial extraction batch.
- **Model fits:** completed inner training trajectories and outer refits,
  including the optional final selection/refit. With the default grid and
  `--fit-final`, the total is 42 fits: 35 for evaluation and 7 for the final model.
- **Training epochs:** current fit's epoch count, latest weighted BCE and warmup
  or training phase; the description identifies the repeat/fold and inner fit.
- **Epoch participants / validation / test:** participant counts within the
  current pass. The training counter includes each participant once per epoch,
  independently of gradient accumulation.

The bars show elapsed time, throughput and estimated remaining time. Estimates
are based on work already completed and can change between fits with different
epoch budgets. The Model fits bar counts fits, rather than wall-clock work.
Bars close on failure and do not mark unfinished fits as completed.

To retain the ordinary text logs without progress bars, append `--no-progress`
to the same command. `--progress` explicitly enables them again. The inference
command also supports both flags. Library callers can use
`fit_finetune(..., progress=True)` and `predict_scores(..., progress=True)`;
these library functions are quiet by default.

## 8. Inference from your trained model

Use the identical official preprocessing and channel order for new `.npy`
participant arrays. The saved input protocol selects all 132 channels or the
matching 60-channel subset automatically. This command does not take raw CSV
recordings directly.

```sh
.venv-transformer/bin/python -m sota_pd_et.predict_mantis_finetune \
  --model artifacts/pads_mantis_v2_finetuned/YOUR_RUN/final_model.pt \
  --inputs new_participants.npy \
  --output fine_tuned_predictions.csv \
  --device auto
```

The model bundle includes the pruned encoder, participant head, initial-feature
statistics, input protocol, architecture and training metadata. Restoration uses
`torch.load(weights_only=True)` and does not download the original checkpoint.
Restore the recorded environment for numerical reproduction. Automatic training
resume/optimizer restoration is not implemented.

## 9. Sources

- [Official Mantis repository](https://github.com/vfeofanov/mantis).
- [MantisV2 paper](https://arxiv.org/abs/2602.17868).
- [Pinned pretrained revision](https://huggingface.co/paris-noah/MantisV2/tree/99fe0f548960e272fbfa4b82fd9b5b5956779dfd).
- [PADS v1.0.0 and citation/license instructions](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/).

The pretrained upstream architecture is used through the locally pinned package;
this code supplies the PADS participant head, fine-tuning policy and nested
evaluation. Performance improvement is an experimental question, not an
implementation guarantee.
