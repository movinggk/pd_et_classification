# PADS data and Fourier demonstration

Prepared and executed on **7 October 2026** against the acquired PADS v1.0.0 release. The demonstration uses original raw recordings; it does not use the authors' already preprocessed binary arrays.

## Open the demonstration

Start with [`outputs/index.html`](outputs/index.html). It is a self-contained, offline HTML explorer: the plotting library, numerical plot data, and figures are embedded. Open it in a browser without installing Python packages. The companion CSV/JSON download links require keeping the other files in `outputs/` beside it.

The executed [`PADS_demonstration.ipynb`](PADS_demonstration.ipynb) contains the code, equations, plots, and recorded outputs. Change the participant/task/wrist/sensor parameter cell and rerun later cells to inspect another recording. Use the `.venv-demo` Python interpreter when executing it.

## Included material

| Material | Scope |
| --- | --- |
| Dataset overview | All 469 participants, with the six published diagnosis counts |
| Interactive recording explorer | PD 004, ET 066, healthy 001; all 11 tasks, both wrists, both sensors |
| Fourier teaching example | Clearly marked synthetic 5 Hz + 8 Hz signal |
| Fourier analysis of real data | Axis amplitude spectra, full-record Hann periodograms, Welch PSDs |
| Full-cohort comparison | All 276 PD + 28 ET participants, `Relaxed` and `StretchHold`, both wrists and both sensors |
| Timing-quality sensitivity | Separate comparison retaining a participant/task only if both raw wrists have maximum timestamp gaps ≤ 30 ms |
| Scientific figures | Seven figures, each exported as PNG and SVG in [`outputs/figures/`](outputs/figures/) |
| Numerical verification | Synthetic-frequency/amplitude check, SciPy equivalence, Parseval identity, odd/even handling, DC/Nyquist handling, and real-data timing checks |

Examples are selected by the smallest published participant ID in each named condition, without looking at signals or spectra. They are individual examples, not representative group averages. PD 004's clinical comment specifies an akinetic-rigid phenotype; it was not selected for a strong resting tremor.

## Suggested ten-minute demonstration

1. **Minute 0–1: Define the cohort.** Show 469 total participants, 276 PD and 28 ET, and explain why 10,318 recordings do not mean 10,318 independent people. Select ET using `condition == "Essential Tremor"`; the original numeric label 2 also contains other differential diagnoses.
2. **Minutes 1–3: Show the data.** Inspect raw XYZ acceleration and angular velocity, their units, onset vibration, and actual timestamps. Explain the 100 Hz nominal rate and why recorded timing must be checked.
3. **Minutes 3–5: Explain Fourier analysis.** Show the synthetic 5 Hz and 8 Hz example first. Then choose ET 066, `StretchHold`, `RightWrist`, `Accelerometer`, and `FFT amplitude`. Relate the time trace to the frequency peaks, with the physical unit preserved.
4. **Minutes 5–7: Compare estimators.** Switch from amplitude to the full-record Hann PSD and then Welch PSD. Explain g²/Hz versus g, the 0.25 Hz Welch grid, four-second segments, and 50% overlap. Switch wrist, task, or sensor and describe the observed changes.
5. **Minutes 7–9: Show the cohort.** Display medians/IQRs and participant peak-frequency distributions. Switch between resting and postural tasks, and between all participants and the timing-quality subset. Explain the class imbalance, changing subset counts, and overlap.
6. **Minute 9–10: State what was demonstrated.** The analysis shows recorded motion and exploratory spectral summaries. It does not estimate diagnostic accuracy. Mention timestamp gaps and the need for participant-independent splits in any later classifier experiment.

## Exact analysis contract

1. Read the seven-column raw file through its observation JSON, verify its published SHA-256, and require finite values and strictly increasing timestamps.
2. Starting at `first timestamp + 0.5 s`, linearly interpolate the six sensor axes onto the metadata's uniform 100 Hz grid. Retain only grid points within recorded support; never extrapolate.
3. Subtract a least-squares linear trend separately for each interpolated axis. This is demonstration-specific preprocessing, distinct from the authors' supplied processed data.
4. Apply a **periodic Hann window**. Compute the real FFT with no zero padding. The full-record frequency spacing is `fs/N`, where `N` is the actual retained grid length.
5. Normalize one-sided amplitude by the sum of window weights. Normalize one-sided PSD by `fs × sum(window²)`. Double interior positive-frequency bins, but not DC or an even-length Nyquist bin. The notebook defines every object and states the normalization identity used by the numerical checks.
6. Compute Welch PSD using 400-sample periodic-Hann segments, 200-sample overlap, per-segment mean subtraction, arithmetic mean averaging, and no zero padding. The bin spacing is **0.25 Hz**.
7. For each sensor, sum the three axis PSDs. This avoids Fourier-transforming the nonlinear vector norm. In cohort comparisons, average the left/right axis-summed PSDs before calculating one feature set per participant, task, and sensor.
8. Define exploratory band power as `df × sum(PSD bins whose centers lie in [3,12] Hz)`, including endpoints. Reference power uses centers in `[1,20] Hz`. The fraction is exploratory/reference power. The peak is the first largest PSD bin inside `[3,12] Hz`; it is undefined for zero band power. This is an explicit discrete bin-sum convention, not an exact integral of an unknown continuous spectrum.

The displayed spectrum range is 0–20 Hz. Calculations retain the complete one-sided 0–50 Hz spectrum. FFT bin spacing is not the same as the ability to resolve nearby components; the Hann window broadens peaks.

## Measured timing limitation

Of **1,216 cohort recordings**, **561** have at least one raw timestamp gap greater than **15 ms** (1.5 nominal intervals). The largest is **646.577 ms**, in PD participant **006**, task `StretchHold`, left wrist. Interpolation over a long gap can change a spectrum and cannot recover the missing motion.

The main view retains all published PD/ET participants. The separate sensitivity view uses a maximum gap of **30 ms in each entire raw wrist recording**. This threshold is an explicitly declared exploratory quality choice made after inspecting timestamps; it is not a validated clinical cutoff and was not optimized for diagnostic separation. Retention varies by task and condition:

| Task | PD retained / total | ET retained / total |
| --- | ---: | ---: |
| `Relaxed` | 250 / 276 | 26 / 28 |
| `StretchHold` | 185 / 276 | 15 / 28 |

Consequently, a change in the filtered curves can reflect changed cohort composition as well as timing quality. Within each displayed cohort, a participant contributes once per task and sensor. The IQR is a distribution across participants, not a confidence interval.

All original diagnosis assignments are retained, including eight PD-assigned participants whose comments mention essential tremor. No clinical relabeling or comment-based exclusion was performed.

## Outputs and provenance

- [`outputs/cohort_features.csv`](outputs/cohort_features.csv): 1,216 rows = 304 participants × two tasks × two sensors; includes the timing-sensitivity flag.
- [`outputs/cohort_recording_qc.csv`](outputs/cohort_recording_qc.csv): 1,216 raw-recording checks, including timestamp rates/gaps, sample counts, FFT spacing, and source hashes.
- [`outputs/example_recording_qc.csv`](outputs/example_recording_qc.csv): 66 example-recording checks.
- [`outputs/demo_data.json`](outputs/demo_data.json): all explorer plot data and full/filtered cohort summaries.
- [`outputs/fourier_checks.json`](outputs/fourier_checks.json): numerical verification results.
- [`outputs/run_manifest.json`](outputs/run_manifest.json): parameters, source-code hashes, 1,274 distinct raw-source hashes, package versions, and notebook execution status.

All 1,274 distinct raw recordings used in the demonstration were verified against PhysioNet's SHA-256 list when loaded. The larger dataset acquisition was separately verified in [`../data/pads/acquisition/`](../data/pads/acquisition/).

## Rebuild

Run from the project root, with Python 3.12:

```sh
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r demonstration/requirements-lock.txt
.venv-demo/bin/python -m demonstration.build_demo
```

The build regenerates figures, data tables, the offline HTML, and the notebook; it then executes every notebook code cell using that same interpreter. Notebook execution needs permission to open local kernel connections. No globally registered Jupyter kernel is required.

To run only the focused Fourier checks:

```sh
.venv-demo/bin/python -m demonstration.check_fourier
```

Reusable objects are defined in [`pads_demo.py`](pads_demo.py): `PADSDataset`, `RawRecording`, `SpectralConfig`, `Analysis`, `analyze_recording`, `fourier_spectra`, `vector_psd`, and `spectral_features`.

## Sources and license

Data and protocol: [PADS v1.0.0 on PhysioNet](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/), DOI [10.13026/m0w9-zx22](https://doi.org/10.13026/m0w9-zx22). Data license: **CC BY-NC-SA 4.0**; follow the official dataset citation instructions for publications. Source-data derivatives in this demonstration retain that attribution and license.

Spectral estimator references: [SciPy periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html) and [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
