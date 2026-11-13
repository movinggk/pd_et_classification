# PADS PD-versus-ET: frozen MantisV2 transformer features

Korean mathematical explanation: [standalone LaTeX source](docs/mantis_v2_explanation_ko.tex),
[compiled PDF](../output/pdf/mantis_v2_explanation_ko.pdf), and
[compilation guide](docs/mantis_v2_readme_ko.md).

This experiment replaces the MultiROCKET–HYDRA feature map with an existing,
pretrained **MantisV2 transformer encoder**, then trains a class-balanced ridge
classifier. The encoder remains frozen throughout evaluation. MantisV2 is a
recent classification-focused time-series foundation model with public weights;
choosing it does not establish that it is the universally newest or best model,
or that it improves PD-versus-ET classification. The completed MPS experiment
is documented in [the result report](../artifacts/pads_mantis_v2/REPORT.md).

The experiment uses the same published PADS movement representation and the
same participant-level nested evaluation protocol as the
[MultiROCKET–HYDRA baseline](README.md). It is an exploratory comparison on an
already studied cohort, rather than an independent external validation.

## Install in a separate environment

Run these commands from the project root with Python 3.12:

```sh
python3.12 -m venv .venv-transformer
.venv-transformer/bin/python -m pip install --upgrade pip
.venv-transformer/bin/python -m pip install -r sota_pd_et/requirements-transformer.txt
```

The requirements file pins the Mantis package and bounds the numerical
dependencies. It is a portable dependency specification, not an exact lock for
every platform. Installation and the first checkpoint download need Internet
access. Evaluation uses local PADS files. The completed experiment uses the Apple
MPS GPU for transformer inference and the CPU for scaler and ridge fitting.
CPU inference is also supported; a GPU is not required.

The environment prepared locally for this trial reuses installed baseline
libraries through a Python `.pth` path file. That is a workspace-specific setup
detail, not a portable installation requirement. A fresh environment should use
the ordinary installation commands above. Record the resolved package versions
when reproducing an experiment:

```sh
.venv-transformer/bin/python -m pip freeze > transformer-environment-lock.txt
```

## Run the experiment

```sh
.venv-transformer/bin/python -m sota_pd_et.evaluate_transformer \
  --output artifacts/pads_mantis_v2 \
  --batch-size 128 \
  --threads 4 \
  --device mps \
  --fit-final
```

Use a fresh output directory for each run. The default input directory is
`data/pads/1.0.0`. The batch size limits the number of single-channel sequences
encoded together; the thread count limits CPU work. Reduce either setting if
the machine has less memory or fewer available CPU cores. Omit `--fit-final` to
perform nested evaluation without creating a classifier fitted on the full
cohort.

Use `--device cpu` on a machine without an Apple GPU, or `--device cuda` with
a CUDA-enabled PyTorch installation and NVIDIA GPU. The code fails explicitly
if the selected device is unavailable. In this desktop session, the sandbox
could not see MPS; the experiment ran with approved GPU access. A small parity
check, including constant records, compared CPU and MPS output before the run:
maximum absolute feature difference was `2.86102294921875e-6`, and every feature
was finite. The check is saved with the completed experiment.

## Fixed encoder and input transformation

The extractor is specified before observing this experiment's outer-test
results:

| Object | Fixed value |
| --- | --- |
| Package | `mantis-tsfm==1.1.0` |
| Architecture | `mantis.architecture.MantisV2` |
| Pretrained checkpoint | `paris-noah/MantisV2` |
| Immutable checkpoint revision | `99fe0f548960e272fbfa4b82fd9b5b5956779dfd` |
| Weight file | `model.safetensors` |
| Weight SHA-256 | `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1` |
| Transformer output layer | Index `2`, the third transformer layer |
| Output aggregation | `combined`: class token concatenated with mean patch token |
| Output per channel | 512 real-valued features |
| Encoder state | Frozen, evaluation mode, float32 MPS inference in the completed run |
| Resized input length | 512 samples per channel |

The official full backbone has 4,188,672 parameters. The intermediate-layer and
combined-token settings follow the authors' feature-extraction recommendation;
they are not selected using PADS outer-test performance. The checkpoint was
pretrained on the synthetic CauKer2M corpus. According to the authors, this
synthetic-only pretraining avoids the real-dataset overlap present in the
original Mantis pretraining corpus. This establishes provenance of the encoder,
not clinical validity or guaranteed transfer performance.

Each participant contributes one float32 array of shape `(132, 976)`. The 132
channels identify retained task-segment, wrist, sensor, and axis combinations;
they are not 132 simultaneously recorded physical axes. The loader retains the
published channel order and selects **276 PD and 28 ET participants**, with
**PD = 0 and ET = 1**. Labels are selected from the exact diagnosis strings,
not from the original numeric differential-diagnosis label. The existing loader
checks source hashes, file sizes, label agreement, unique IDs, and finite values.

Feature extraction proceeds as follows:

1. Resize each length-976 channel to length 512 using linear interpolation with
   `align_corners=False`. No cohort statistics or labels are used by this step.
2. Send each channel independently through the frozen encoder. Do not add
   per-record z-scoring before this pass. Mantis internally standardizes its
   signal and first-difference branches, while separately encoding raw patch
   means and standard deviations. Additional z-scoring would change the
   amplitude information available to those statistical branches.
3. Concatenate the 512-dimensional channel embeddings in the existing channel
   order. A participant therefore has `132 × 512 = 67,584` features. The full
   cohort feature array has shape `(304, 67584)`.

The transformer does not directly attend across different channels. Channel
relationships are available to the downstream linear classifier through the
concatenated embeddings. Interpolation changes temporal resolution; the
published sample-index representation is not converted back into raw timestamps
or corrected for timing gaps.

Because the checkpoint and interpolation are fixed and estimate no parameters
from the cohort, their features can be computed once and reused across folds.
Every operation that learns from PADS features or labels remains inside the
appropriate training fold. In particular, the feature scaler is not fitted on
the complete cohort before cross-validation.

## Participant-level nested evaluation

The default is one repeat with **five stratified outer folds**, **three
stratified inner folds**, and master seed **20261017**. The existing split
construction is reused so that the default trial can be compared with the
baseline using the same held-out participants.

For each outer fold:

1. Keep every channel and task belonging to one participant together. Reserve
   the outer test participants until the final prediction for that fold.
2. Within each inner fold, fit a standard feature scaler on inner-training
   embeddings only, then apply that fitted scaler to inner-validation embeddings.
3. Fit a separate ridge classifier for each
   `alpha ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}`. For class `c`, use the weight
   `w_c = n_train / (2 n_c)`, where `n_c` is the number of training participants
   in class `c`; recompute the weights for every fit.
4. Choose the alpha with the highest mean inner-fold balanced accuracy. Ties
   within `1e-12` select the largest alpha. The decision rule is fixed:
   **predict ET when the ridge margin is strictly greater than zero, and PD
   otherwise**. Do not optimize a threshold on outer-test labels.
5. Fit a fresh scaler and the selected ridge classifier on the complete outer
   training set. Predict each outer-test participant once.

The primary metric is balanced accuracy,
`(ET recall + PD recall) / 2`. Inspect ET recall, PD recall, the confusion matrix,
AUROC, and ET average precision alongside it. Ridge margins are decision scores,
not calibrated probabilities. The confusion matrix uses true labels as rows,
predictions as columns, and `[PD, ET]` order on both axes.

With `--fit-final`, perform a separate three-fold inner alpha search on the full
cohort, then fit a final scaler and ridge classifier on all participants. This
model has no independent test set. Its training predictions do not replace
nested-CV results.

## Saved artifacts and comparison scope

The output directory records the checkpoint and configuration, cached frozen
features, their provenance manifest, participant assignments, alpha searches,
held-out predictions, and evaluation metrics. Retain these together so that
features and predictions can be traced to their encoder, input order, source
data, and software versions.

The optional final classifier bundle contains the fitted feature scaler, ridge
classifier, and the information needed to identify its feature extractor. It
does not serialize the full PyTorch transformer inside a joblib object. Reusing
the classifier requires the matching pinned checkpoint, extractor configuration,
and upstream PADS preprocessing/channel order. A cached feature matrix can be
used directly only for the exact participant rows represented by that cache.

For a comparison with the saved MultiROCKET–HYDRA run, verify that cohort IDs,
labels, outer and inner participant assignments, ridge alpha candidates, class
weights, and the decision threshold agree. Compare held-out predictions for the
same participants. The feature extractors necessarily differ: the baseline
learns its feature maps inside each training fold, while MantisV2 uses one fixed
external checkpoint.

Only 28 participants have ET, and each outer test fold contains five or six ET
participants. Fold estimates are consequently variable, and overlapping training
sets make fold standard deviations unsuitable as confidence intervals. The
trial follows prior analysis of this cohort, so a favorable result remains
exploratory evidence. It cannot establish general clinical superiority,
prospective diagnostic utility, calibrated risk, or universal state-of-the-art
performance. Additional model or preprocessing choices made after inspecting
these outer results would further limit a confirmatory interpretation.

## Official sources

- [Mantis repository and feature-extraction instructions](https://github.com/vfeofanov/mantis).
- [MantisV2 paper](https://arxiv.org/abs/2602.17868).
- [MantisV2 checkpoint](https://huggingface.co/paris-noah/MantisV2).
- [Immutable weight-upload revision](https://huggingface.co/paris-noah/MantisV2/commit/99fe0f548960e272fbfa4b82fd9b5b5956779dfd).
- [Official weight file and checksum](https://huggingface.co/paris-noah/MantisV2/blob/99fe0f548960e272fbfa4b82fd9b5b5956779dfd/model.safetensors).
- [Mantis version 1.1.0 architecture](https://github.com/vfeofanov/mantis/blob/v1.1.0/src/mantis/architecture/version2.py).
- [CauKer2M synthetic pretraining data](https://huggingface.co/datasets/paris-noah/CauKer2M).
- [PADS v1.0.0 dataset and usage terms](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/).
