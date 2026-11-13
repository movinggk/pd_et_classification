# MantisV2 fine-tuning technical report

This revised Korean report compares the completed user-run experiments under
`artifacts/pads_mantis_v2_finetuned/`:

- Head-only: `20261109T032315873662Z`.
- Last-block: `20261109T041659480074Z` (primary last-block result).
- Previous last-block: `20261108T145903609065Z` (same-seed reproduction check).

All use the official 132-channel PADS inputs, 276 PD and 28 ET participants,
and participant-level 5-outer/3-inner-fold nested cross-validation. Head-only
freezes the encoder and trains the participant head. Last-block adapts the last
of the three active blocks and the same head. Frozen MantisV2+ridge and
MultiROCKET--HYDRA are two additional matched baselines. Inputs, ordered cohort,
all nested participant splits, source code and initial weights match.

The report **does not train, construct, or run a model**. `build_report.py`
reads saved predictions and tensors, checks their provenance and training-only
normalization buffers, and authors the tables/figures. It never imports the
project training/inference functions. Original experiment artifacts are unchanged.

## Deliverables

- `report.tex`: standalone editable XeLaTeX source, with embedded plot
  coordinates, tables and bibliography; does not require PADS or model weights.
- `report_template.tex`: source template used by the report generator.
- `build_report.py`: saved-result audit and source generator; requires the
  completed run and two original baseline directories in this project layout.
- `metrics.csv`: full-precision pooled metrics, including **ET-positive
  trapezoidal PR-AUC and AP as distinct quantities**, G-mean, MCC and macro-F1.
- `fold_metrics.csv`: 20 outer-fold results for the four evaluated methods.
- `verification.json`: all three manifests, recalculated metrics, candidate
  selection checks, class-stratified paired bootstrap, reproduction checks
  and artifact hashes.
- `document_verification.json`: compiled-document and package checks.
- Final PDF: `../output/pdf/mantis_v2_finetune_report_ko.pdf`.
- Portable source bundle: `../output/code/MantisV2_FineTune_LaTeX_Report.zip`.

## Interpretation

| Method | BA | ET sensitivity | PD specificity | PR-AUC (trapezoidal) | AP | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Head-only | 81.57% | 92.86% (26/28) | 70.29% (194/276) | 0.6391 | 0.6444 | 0.3815 |
| Last-block | 76.76% | 57.14% (16/28) | 96.38% (266/276) | 0.6463 | 0.6505 | 0.5534 |
| Frozen MantisV2+ridge | 70.89% | 42.86% (12/28) | 98.91% (273/276) | 0.6229 | 0.6272 | 0.5577 |
| MultiROCKET--HYDRA | 67.13% | 35.71% (10/28) | 98.55% (272/276) | 0.5557 | 0.5776 | 0.4727 |

Last-block does **not** improve BA over head-only. It corrects 72 of the PD
errors made by head-only, while losing 10 ET detections. Its ET precision,
MCC and macro-F1 are higher, while head-only has higher BA, G-mean and pooled
AUROC. This is a tradeoff rather than uniform improvement. Both modes select
epochs internally, so the comparison is of two tuned training procedures;
no fixed-epoch causal effect of adaptation or post-hoc threshold adjustment is
estimated. Head-only has 24 fits and last-block has 42, including final selection.

The new last-block run exactly reproduces the previous run's outer predictions,
candidate summaries, selected settings, refit histories and all state tensors
in the six saved bundles. It is not an independent seed, cohort, or split and
does not add 304 independent observations or another 28 ET participants. The
previous document revision is preserved in files with `_before_head_only`
names in `output/pdf/` and `output/code/`.

ET prevalence is 9.21%; the report shows it as a horizontal no-skill PR reference.
AP and trapezoidal PR-AUC have different interpolation rules. For the degenerate
constant-score Always PD baseline, default endpoint interpolation produces a
misleading trapezoidal area of 0.5461 even though AP is only 0.0921. The report
explains this and displays a dash for its main-table PR-AUC; the raw computed
area is retained in the audit JSON/CSV. Do not interpret that value as skill.

The report's 20,000 paired bootstrap draws preserve 276 PD and 28 ET per draw
and match participant indices across methods. Their percentile intervals are
conditional on saved fits. They omit training-set/model-selection variability,
cross-validation dependence and the exploratory design history; they are not
a superiority test of the training procedures. No operating threshold is tuned
on outer predictions. The added PR-AUC and G-mean do not change original model
selection, which used inner mean BA.

## Compile the standalone source

Use a TeX distribution with XeLaTeX, `kotex`, UnBatang fonts, TikZ and PGFPlots.
From this directory:

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../tmp/pdfs/mantis_finetune report.tex
```

All tables and figures are embedded. No network access or external images are
needed to compile the source. An alternative installed Korean font can be
configured in the preamble if UnBatang is absent.

## Recompute the audit and generate source

From the project root, using the existing experiment environment:

```sh
.venv-transformer/bin/python MantisV2_FineTune_Report/build_report.py
```

`--audit-only` checks saved evidence without writing the source or audit outputs.
The script is intentionally specific to the documented run configurations and
fails on mismatching labels, splits, saved metrics, sources or normalization.
Optional `--run`, `--head-only-run` and `--previous-run` override the three paths;
their modes and the documented protocol must still match. All 18 model files
are read without constructing a model. Frozen encoder tensors are checked
against the original checkpoint; last-block must actually change the third
block, while the other encoder tensors must remain exact. All normalization
buffers must equal statistics of the corresponding training participants only.
It validates the selected candidate's saved inner predictions; nonselected
candidate-level participant predictions were not saved and cannot be fully
recomputed without repeating model training, which this script never does.

The source ZIP omits participant-level raw data, predictions, and trained model
weights. The standalone document compiles without them. Regenerating the audit
requires the original experiment folders. The audit includes source-data file
hashes and participant identifiers in data filenames from the original manifest.
