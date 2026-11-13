# PADS PD-versus-ET: MultiROCKET–HYDRA

For the seven-model pretrained-encoder comparison with optional backbone
fine-tuning, see [the comparison guide](FOUNDATION_COMPARISON_README.md).
That implementation has been tested; its PADS benchmark is left for the user
to run.

For the completed pretrained-transformer feature-extraction trial, see
[MantisV2 instructions](TRANSFORMER_README.md) and the
[MPS evaluation report](../artifacts/pads_mantis_v2/REPORT.md).

Runnable model, nested evaluation, and prediction code for the data already in
this workspace. The completed user-run experiment in `artifacts/pads_mr_hydra`
is documented with outer-evaluation figures in the Korean article below.

For Google Colab, use the single-file
[`PD_ET_Classification_Colab.ipynb`](../PD_ET_Classification_Colab.ipynb).
Upload it at [Colab](https://colab.research.google.com/), choose a Python GPU
runtime for CUDA acceleration of HYDRA, and run all cells. CPU runtimes also work.
It contains all classification pipeline sources and tests, downloads
only the checksum-verified classification inputs, installs dependencies in an
isolated environment, runs the same five-outer/three-inner participant-level
evaluation and final fit, and exports metrics, plots, predictions, source code,
the actual package lock, and the final model in a ZIP. It needs Internet access
and Python 3.11–3.14. The default transform batch size is 16 to reduce memory
use. The full dataset, local environment, and previous experiment artifacts are
not required. Training is sequential across folds; partial runs cannot resume
automatically. Download the results before Colab releases the runtime.

The notebook defaults to `HYDRA_DEVICE = 'auto'`, selecting CUDA when available
and CPU otherwise. `'cpu'` preserves the original aeon CPU path; `'cuda'`
requires a usable NVIDIA GPU and fails clearly if it is unavailable. Only
HYDRA's convolutions and competing-kernel counts are accelerated. MultiROCKET,
both scalers, and ridge stay on the CPU. The notebook includes an optional
synthetic HYDRA timing check that counts transfers; whole-experiment speedup
depends on the hardware and workload and has not been established locally.
GPU computations reuse the CPU-fitted kernels and channel indices. TF32 is
disabled, but floating-point arithmetic and winner ties can still change
features or predictions relative to CPU. Device, GPU name, and CUDA build are
recorded in the experiment manifest.

Maintainers can regenerate the notebook using Python's standard library:

```sh
python -m sota_pd_et.build_colab_notebook
```

Rebuild it after editing an embedded module; the notebook is a snapshot of those
sources and records their hashes in its metadata.

한국어 수학 설명서: [LaTeX 원고](docs/model_explanation_ko.tex),
[컴파일 안내](docs/README.md).

한국어 Beamer 요약 발표자료: [LaTeX 원고](docs/model_summary_beamer_ko.tex),
[발표용 PDF](../output/pdf/model_summary_beamer_ko.pdf),
[독립 컴파일 안내](docs/beamer_readme_ko.md).

한국어 문서의 실제 평가 그림과 재생성 안내:
[평가 결과를 포함한 PDF](../output/pdf/model_explanation_ko.pdf),
[컴파일·그림 재생성](docs/README.md).

The selected method is **MultiROCKET–HYDRA**, a strong published time-series
classification method. The [2024 classification bake-off](https://arxiv.org/abs/2304.13029)
supports the choice as a competitive general time-series method. That benchmark
does **not** establish that it is the best model for PD versus ET, or the best
multivariate model available in October 2026. This implementation is a candidate
to evaluate, not an assertion of a new SOTA result.

## Install in a separate environment

Run from the project root, using Python 3.12:

```sh
python3.12 -m venv .venv-sota
.venv-sota/bin/python -m pip install --upgrade pip
.venv-sota/bin/python -m pip install -r sota_pd_et/requirements-lock.txt
```

The existing `.venv-demo` has NumPy/SciPy versions outside aeon 1.6.0's allowed
ranges. Keep the model environment separate. PyTorch is needed for HYDRA; the
command-line default is CPU. Pass `--hydra-device auto` or `--hydra-device cuda`
to use CUDA for HYDRA after installing a compatible CUDA-enabled PyTorch build.
The Colab notebook handles that installation automatically. MPS is not a backend
for this HYDRA implementation. Installation needs Internet access; the experiment
itself uses local files.

`requirements-lock.txt` records the exact environment used for the synthetic
verification (Python 3.12.13, macOS arm64). `requirements.txt` provides the looser
dependency specification if you need a fresh platform-specific resolution. It
pins aeon because the model imports aeon's internal HYDRA sparse scaler. After
installation, you can save your own complete environment lock:

```sh
.venv-sota/bin/python -m pip freeze > sota-environment-lock.txt
```

## Run the experiment

```sh
.venv-sota/bin/python -m sota_pd_et.evaluate \
  --data-root data/pads/1.0.0 \
  --output artifacts/pads_mr_hydra \
  --outer-folds 5 \
  --inner-folds 3 \
  --repeats 1 \
  --n-jobs 4 \
  --batch-size 32 \
  --seed 20261017 \
  --fit-final
```

Choose an output directory that does not exist. The code refuses to mix runs or
overwrite source data. If `--output` is omitted, it creates a timestamped directory
under `artifacts/pd_et_mr_hydra/`. A failed/interrupted run retains completed fold
files and a failure status; it does not support automatic resumption. Use a fresh
output directory when restarting.

Omit `--fit-final` if you only want evaluation. With the defaults, evaluation fits
15 inner feature maps and five outer feature maps. Final-model creation adds
three inner maps and one fit on all participants. Each inner map is reused across
the seven alpha candidates; it is never reused between validation folds. CPU
runtime and memory usage depend on the machine. Reduce `--batch-size` to 8 or 16
if memory is limited; reduce `--n-jobs` if fewer CPU cores are available.

CUDA execution is optional: add `--hydra-device auto` to the command above to
select a GPU when available, or `--hydra-device cuda` to require one. Each fold
still fits its feature map using only its own training participants. Device
caches are omitted from the saved model; the original kernels and scaler state
stay on CPU. A saved CUDA model automatically permits CPU fallback after loading,
provided the required package versions are present. `predict` also accepts
`--hydra-device cpu`, `auto`, or `cuda` as an inference override. This does not
relax its exact environment-version check; `torch-install.json` in the notebook
ZIP records the correct CPU/CUDA wheel index for restoring the training build.

For a repeated experiment, change `--repeats 1` to `--repeats 3` and choose a new
output directory. Every participant is tested once per repeat. Repeats reuse the
same people; they do not increase the number of independent ET cases.

## Inputs and labels

- The current cohort contains **304 participants: 276 PD and 28 ET**.
- The selection uses the exact `condition` strings `Parkinson's` and
  `Essential Tremor`. The new binary target is **PD = 0, ET = 1**.
- The original numeric label `2` combines several differential diagnoses and
  must not be treated as an ET-only label.
- Each participant contributes their official little-endian float32 movement
  file, reshaped to **132 channels × 976 samples**. The complete input array has
  shape **(304, 132, 976)**. Both wrists and all retained tasks remain together.
- The channels follow the authors' released order: task segment, wrist, sensor,
  then axis. `cohort.csv` and `manifest.json` record the cohort and channel layout.
- Diagnoses and ID strings select labels/files only. The model receives no
  disease comments, original numeric labels, medication, demographics,
  questionnaire responses, timing-quality flags, or participant IDs as features.
- Eight PD-assigned participants whose comments mention ET retain their published
  PD label. The manifest records their IDs for audit only.
- Before loading, the code checks SHA-256 hashes for the participant table,
  participant JSON files, selected movement files, and two upstream scripts that
  define binary preprocessing/order. It verifies binary sizes, all selected
  movement values' finiteness, unique IDs, and table/JSON label agreement.

## Model and exact evaluation protocol

1. Construct stratified **five-fold outer cross-validation** over participants.
   These are new seeded PD/ET splits, not the original study's PD/HC/DD folds.
2. For each outer training set, construct **three stratified inner folds**.
3. In each inner training fold, fit aeon's multivariate HYDRA and MultiROCKET
   transforms. Fit HYDRA's sparse scaler and MultiROCKET's standard scaler only
   on those transformed training participants. Transform inner validation
   participants using these fitted objects.
4. Concatenate the scaled features. Fit separate ridge classifiers for
   `alpha ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}`. Every fit uses class weights
   `w_c = n_train / (2 n_c)`, where `n_c` counts training participants of class `c`.
5. Choose the alpha with the highest mean inner-fold balanced accuracy. Ties
   within `1e-12` choose the largest alpha. The class decision is fixed before
   evaluation: **ET when the signed ridge margin is strictly greater than zero;
   PD otherwise**. Thresholds are not optimized on either validation or test data.
6. Fit fresh transforms, scalers, and the selected ridge classifier on the entire
   outer training set; predict the outer test participants once.
7. Save every outer prediction, then calculate metrics using one held-out
   prediction per participant per repeat. Alpha-selection results are diagnostic
   training outputs and must not be reported as test performance.
8. If requested, perform a separate inner alpha search on the full cohort and
   fit a final model on all 304 participants. This final model has no independent
   test set; use the nested-CV results to assess the training procedure.

The feature transforms match the components/scaling of aeon's
[MultiRocketHydraClassifier](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.convolution_based.MultiRocketHydraClassifier.html).
The difference is explicit fold-local alpha selection with `RidgeClassifier`,
rather than letting `RidgeClassifierCV` choose alpha after fitting a feature map
on all outer training participants. No extra normalization is applied to individual
input channels; amplitude information remains available to the transforms.

The default configuration uses 6,250 MultiROCKET kernels, 64 HYDRA groups, and
eight competing kernels per HYDRA group. MultiROCKET uses both the original
series and differences with four pooling statistics. HYDRA contributes features
from competing kernels. The underlying transforms are provided by aeon, rather
than approximated with a newly written convolution implementation.

## Metrics and saved outputs

The primary metric is **balanced accuracy**:

`BA = (ET recall + PD recall) / 2`.

The code also reports AUROC, ET average precision, ET precision/recall/F1, PD
recall, macro F1, Matthews correlation, ordinary accuracy, and the confusion
matrix. Confusion-matrix rows are true labels and columns are predicted labels,
both in **[PD, ET]** order.

An always-PD classifier has about 90.8% ordinary accuracy on this cohort, but only
50% balanced accuracy and zero ET recall. Its results are included automatically.
The average-precision no-skill reference is ET prevalence: `28 / 304 ≈ 0.0921`.

**Scores are not probabilities.** AUROC and AP use continuous ridge margins;
positive values favor ET. This code deliberately does not report Brier score,
log loss, or calibration curves. Metrics pooled over a repeat compare margins
from different fitted models, whose numerical scales can differ. Inspect the
saved fold-level AUROC/AP and their descriptive means alongside pooled results.

| Output | Contents |
| --- | --- |
| `summary.json` | Complete nested-CV metrics, repeat summaries, fold summaries, and baseline |
| `oof_predictions.csv` | Participant ID, true target, outer fold/repeat, margin, prediction |
| `fold_metrics.csv` | One row per outer fold |
| `splits.json` | Exact outer and inner participant assignments |
| `cohort.csv` | Exact selected cohort and audit flags |
| `manifest.json` | Configuration, actual versions, source hashes, data hashes, status |
| `repeat_XX_fold_XX/alpha_search.csv` | Inner alpha-selection results |
| `repeat_XX_fold_XX/inner_oof_predictions.csv` | Selected alpha's inner predictions; not test results |
| `repeat_XX_fold_XX/test_predictions.csv` | This fold's held-out predictions |
| `repeat_XX_fold_XX/metrics.json` | This fold's metrics |
| `final_model.joblib` | Optional full-cohort model for later inference |
| `final_inner_splits.json` | Optional final model's alpha-search folds |

Reported standard deviations describe folds/repeats; they are **not confidence
intervals**, since training sets overlap. Five-fold tests contain only five or
six ET participants each, so inspect ET counts and the confusion matrix. There
is no independent external cohort in this evaluation.

The official representation already contains preprocessing and task choices made
in the original PADS study. This is an evaluation on that published
representation, not an untouched prospective evaluation of raw-signal processing.
The supplied binaries also retain sample-index timing: this code does not
reconstruct raw timestamps or correct the gaps previously documented in this
workspace. Treat any comparisons with the demonstration's resampled spectra
accordingly. Software verification alone does not establish clinical validity.

## Predict using a saved final model

Prepare a NumPy `.npy` array with shape `(N, 132, 976)` using the **same upstream
preprocessing and channel order**. Arbitrary raw recordings are not valid inputs.
Then run from the same project root and environment:

```sh
.venv-sota/bin/python -m sota_pd_et.predict \
  --model artifacts/pads_mr_hydra/final_model.joblib \
  --inputs new_participants.npy \
  --output artifacts/new_predictions.csv
```

The prediction script verifies the installed numerical/model package versions
against the saved bundle. Output includes the input row index, continuous margin,
binary prediction, and condition string. Predictions for people already in the
final model's training cohort must not be treated as held-out evaluations.

## Focused verification

```sh
.venv-sota/bin/python -m unittest discover -s sota_pd_et/tests -v
```

Tests cover diagnosis-string selection, checksum corruption, nonfinite values,
table/JSON disagreement, participant separation at both CV levels, prediction
coverage, metric orientation, regularization ties, and independent feature-map
fits. With all dependencies installed, one small synthetic test also exercises
the real batched feature transforms and model serialization. A second synthetic
integration test runs nested CV, final-model fitting, output writing, and the
prediction entry point together. These tests do not use the real PADS data.
Initial Numba compilation can make them slower than later runs.

During code preparation on 17 October 2026, all **12 tests passed** in the locked
environment. The real cohort's loader, 611 source-file hashes, and default
five-fold/three-fold participant assignments were also checked. The real-data
model training and evaluation were not run.

## Sources and attribution

- [PADS v1.0.0 and usage notes](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/),
  dataset DOI [10.13026/m0w9-zx22](https://doi.org/10.13026/m0w9-zx22).
  Data and data derivatives remain subject to **CC BY-NC-SA 4.0**. Follow the
  dataset's citation instructions for publications.
- [HYDRA paper](https://doi.org/10.1007/s10618-023-00939-3).
- [MultiROCKET paper](https://doi.org/10.1007/s10618-022-00844-1).
- [Bake off redux](https://arxiv.org/abs/2304.13029), for the general classification
  benchmark motivation and its scope.
- [aeon v1.6.0 implementation](https://github.com/aeon-toolkit/aeon/blob/v1.6.0/aeon/classification/convolution_based/_mr_hydra.py).
  The upstream library supplies the algorithm implementations and their licenses.
