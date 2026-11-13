"""Build the standalone PADS explorer, scientific figures, and executed notebook."""

import csv
import base64
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pads-demo-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import get_plotlyjs

from .pads_demo import (PADSDataset, SpectralConfig, analyze_recording, vector_psd,
                        spectral_features, fourier_spectra, SENSORS, TASKS, COLORS, CONDITIONS)
from .check_fourier import run_checks

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIGURES = OUT / "figures"
CONFIG = SpectralConfig()
COHORT_TASKS = ("Relaxed", "StretchHold")
GAP_SENSITIVITY_SECONDS = 0.03
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def overview(dataset):
    counts = {condition: sum(row["condition"] == condition for row in dataset.participants)
              for condition in sorted({row["condition"] for row in dataset.participants})}
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    names = list(counts)
    bars = ax.barh(names, list(counts.values()), color=[COLORS.get(c, "#99a8b8") for c in names])
    ax.bar_label(bars, padding=5)
    ax.set_xlim(0, 310)
    ax.set_xlabel("Independent participants")
    ax.set_title("PADS cohort: 469 participants, including 276 PD and 28 ET")
    ax.grid(axis="x", alpha=0.15)
    save_figure(fig, "cohort_overview")
    return counts


def example_payload(analysis):
    # Full-resolution signals are embedded: browser zooming does not hide an
    # extra display-only decimation step. Analysis always uses full resolution.
    display = analysis.recording
    mask_fft = analysis.fft_frequency <= 20
    mask_welch = analysis.welch_frequency <= 20
    return {
        "participant_id": display.participant_id, "condition": display.condition,
        "task": display.task, "wrist": display.wrist, "qc": analysis.qc,
        "time": analysis.time.tolist(),
        "detrended": analysis.detrended_values.T.tolist(),
        "fft_frequency": analysis.fft_frequency[mask_fft].tolist(),
        "fft_amplitude": analysis.fft_amplitude[mask_fft].T.tolist(),
        "fft_psd": analysis.fft_psd[mask_fft].T.tolist(),
        "welch_frequency": analysis.welch_frequency[mask_welch].tolist(),
        "welch_psd": analysis.welch_psd[mask_welch].T.tolist(),
        "features": {sensor: spectral_features(*vector_psd(analysis, sensor), CONFIG) for sensor in SENSORS},
    }


def plot_examples(examples):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout="constrained")
    for i, condition in enumerate(CONDITIONS):
        analysis = examples[(condition, "Relaxed", "LeftWrist")]
        raw = analysis.recording
        for j, (sensor, (columns, unit)) in enumerate(SENSORS.items()):
            ax = axes[i, j]
            for k, axis in enumerate("XYZ"):
                ax.plot(raw.time, raw.values[:, columns.start + k], lw=0.75, alpha=0.85, label=axis)
            ax.axvspan(0, CONFIG.crop_seconds, color="#fed7aa", alpha=0.7, label="Omitted onset")
            ax.set_title(f"{condition} · {raw.participant_id} · {sensor}")
            ax.set_xlabel("Recorded time (s)")
            ax.set_ylabel(unit)
            ax.legend(ncol=4, fontsize=8, loc="upper right")
            ax.grid(alpha=0.12)
    fig.suptitle("Raw PADS recordings · Relaxed task · left wrist\nExamples selected by smallest ID in each condition", fontsize=14)
    save_figure(fig, "example_waveforms")

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout="constrained")
    for i, condition in enumerate(CONDITIONS):
        analysis = examples[(condition, "Relaxed", "LeftWrist")]
        for j, (sensor, (_, unit)) in enumerate(SENSORS.items()):
            ax = axes[i, j]
            f, p = vector_psd(analysis, sensor, "fft")
            fw, pw = vector_psd(analysis, sensor, "welch")
            ax.semilogy(f, p, color=COLORS[condition], alpha=0.45, lw=0.7, label="Full-record Hann periodogram")
            ax.semilogy(fw, pw, color=COLORS[condition], lw=1.6, label="Welch: 4 s, 50% overlap")
            ax.axvspan(3, 12, color="#cbd5e1", alpha=0.18)
            ax.set_xlim(0, 20)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"Sum of axis PSDs ({unit})²/Hz")
            ax.set_title(f"{condition} · {analysis.recording.participant_id} · {sensor}")
            ax.grid(alpha=0.12)
            ax.legend(fontsize=8)
    fig.suptitle("Fourier analysis of the same recordings\n0.5 s crop · uniform 100 Hz grid · linear detrend · no zero padding", fontsize=14)
    save_figure(fig, "example_fourier")


def cohort_analysis(dataset, examples):
    rows, qc_rows, spectra = [], [], {}
    subjects = [row for row in dataset.participants if row["condition"] in CONDITIONS[:2]]
    for number, row in enumerate(subjects, 1):
        for task in COHORT_TASKS:
            analyses = []
            for wrist in ("LeftWrist", "RightWrist"):
                analysis = examples.get((row["condition"], task, wrist))
                if analysis is None or analysis.recording.participant_id != row["id"]:
                    analysis = analyze_recording(dataset.load_recording(row["id"], task, wrist), CONFIG)
                analyses.append(analysis)
                qc_rows.append(analysis.qc)
            for sensor, (_, unit) in SENSORS.items():
                f, left = vector_psd(analyses[0], sensor)
                fr, right = vector_psd(analyses[1], sensor)
                np.testing.assert_array_equal(f, fr)
                # Each participant contributes once per task/sensor. Average
                # the left/right PSDs before computing their spectral features.
                mean_psd = (left + right) / 2
                maximum_gap = max(a.qc["maximum_timestamp_gap_seconds"] for a in analyses)
                accepted = maximum_gap <= GAP_SENSITIVITY_SECONDS
                spectra.setdefault((task, sensor, row["condition"]), []).append((mean_psd, accepted))
                rows.append({"participant_id": row["id"], "condition": row["condition"],
                             "task": task, "sensor": sensor, "wrist_aggregation": "mean of left and right axis-summed PSDs",
                             "maximum_raw_timestamp_gap_across_wrists_seconds": maximum_gap,
                             "passes_30ms_gap_sensitivity": accepted,
                             "power_unit": f"({unit})^2", **spectral_features(f, mean_psd, CONFIG)})
        if number % 50 == 0 or number == len(subjects):
            print(f"Analyzed {number}/{len(subjects)} PD/ET participants for two tasks, both wrists, both sensors.", flush=True)
    write_csv(OUT / "cohort_features.csv", rows)
    write_csv(OUT / "cohort_recording_qc.csv", qc_rows)
    summaries, sensitivity = {}, {}
    for key, values in spectra.items():
        for target, use_quality_filter in [(summaries, False), (sensitivity, True)]:
            stacked = np.stack([p for p, accepted in values if accepted or not use_quality_filter])
            target["|".join(key)] = {"n": len(stacked), "frequency": f.tolist(),
                                    "q25": np.quantile(stacked, 0.25, axis=0).tolist(),
                                    "median": np.median(stacked, axis=0).tolist(),
                                    "q75": np.quantile(stacked, 0.75, axis=0).tolist()}
    return rows, qc_rows, summaries, sensitivity


def plot_cohort(rows, summaries):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    for i, task in enumerate(COHORT_TASKS):
        for j, (sensor, (_, unit)) in enumerate(SENSORS.items()):
            ax = axes[i, j]
            for condition in CONDITIONS[:2]:
                item = summaries["|".join((task, sensor, condition))]
                f = np.array(item["frequency"])
                ax.semilogy(f, item["median"], color=COLORS[condition], label=f"{condition}, n={item['n']}", lw=1.8)
                ax.fill_between(f, item["q25"], item["q75"], color=COLORS[condition], alpha=0.16)
            ax.set_xlim(1, 20)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"Mean-wrist axis-summed PSD ({unit})²/Hz")
            ax.set_title(f"{task} · {sensor}")
            ax.grid(alpha=0.12)
            ax.legend(fontsize=9)
    fig.suptitle("All PD/ET participants: median Welch PSD and interquartile range\nOne PSD per participant, task and sensor; both wrists averaged", fontsize=14)
    save_figure(fig, "cohort_spectra")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    rng = np.random.default_rng(20261007)
    for i, task in enumerate(COHORT_TASKS):
        for j, sensor in enumerate(SENSORS):
            ax = axes[i, j]
            for k, condition in enumerate(CONDITIONS[:2]):
                values = [r["peak_frequency_hz"] for r in rows if r["task"] == task and r["sensor"] == sensor
                          and r["condition"] == condition and r["peak_frequency_hz"] is not None]
                ax.boxplot([values], positions=[k], widths=0.38, showfliers=False, patch_artist=True,
                           boxprops={"facecolor": COLORS[condition], "alpha": 0.15},
                           medianprops={"color": COLORS[condition], "linewidth": 2})
                ax.scatter(k + rng.uniform(-0.15, 0.15, len(values)), values, s=9,
                           alpha=0.4, color=COLORS[condition])
            ax.set_xticks([0, 1], ["PD (n=276)", "ET (n=28)"])
            ax.set_ylim(2.7, 12.3)
            ax.set_ylabel("Largest PSD bin within 3–12 Hz (Hz)")
            ax.set_title(f"{task} · {sensor}")
            ax.grid(axis="y", alpha=0.12)
    fig.suptitle("Exploratory peak-frequency distributions overlap\nA forced spectral maximum is not proof of a tremor or a diagnostic rule", fontsize=14)
    save_figure(fig, "cohort_features")


def plot_fft_teaching():
    fs, n = 100.0, 1000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 5 * t) + 0.4 * np.sin(2 * np.pi * 8 * t)
    f, amplitude, _ = fourier_spectra(x[:, None], fs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].plot(t, x, lw=1, color="#334155")
    axes[0].set_xlim(0, 2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (arbitrary units)")
    axes[0].set_title("Synthetic signal: 5 Hz + 0.4 × 8 Hz")
    axes[1].plot(f, amplitude[:, 0], color="#2563eb")
    axes[1].set_xlim(0, 15)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Hann-corrected amplitude")
    axes[1].set_title("FFT separates the two components")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle("Fourier teaching example · synthetic data, separate from PADS", fontsize=14)
    save_figure(fig, "fft_teaching")


def plot_timing_quality(qc_rows, sensitivity):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    gaps_ms = np.array([row["maximum_timestamp_gap_seconds"] * 1000 for row in qc_rows])
    axes[0].hist(gaps_ms, bins=np.geomspace(gaps_ms.min() * 0.99, gaps_ms.max() * 1.01, 32), color="#526e95")
    axes[0].set_xscale("log")
    axes[0].axvline(30, color="#dc6844", ls="--", label="30 ms sensitivity threshold")
    axes[0].set_xlabel("Largest gap in each raw recording (ms, logarithmic axis)")
    axes[0].set_ylabel("Raw recording count")
    axes[0].set_title("Measured timestamp quality · 1,216 recordings")
    axes[0].legend(fontsize=8)
    labels, accepted_counts, excluded_counts = [], [], []
    for task in COHORT_TASKS:
        for condition, total in [(CONDITIONS[0], 276), (CONDITIONS[1], 28)]:
            accepted = sensitivity["|".join((task, "Accelerometer", condition))]["n"]
            labels.append(f"{task} · {'PD' if condition == CONDITIONS[0] else 'ET'}")
            accepted_counts.append(accepted)
            excluded_counts.append(total - accepted)
    positions = np.arange(4)
    axes[1].barh(positions, accepted_counts, color="#2563eb", label="Both raw wrists have max gap ≤ 30 ms")
    axes[1].barh(positions, excluded_counts, left=accepted_counts, color="#cbd5e1", label="Excluded in sensitivity view")
    axes[1].set_yticks(positions, labels)
    for y, n in zip(positions, accepted_counts):
        axes[1].text(n + 3, y, str(n), va="center", fontsize=9)
    axes[1].set_xlim(0, 315)
    axes[1].set_xlabel("Participants per task")
    axes[1].set_title("Timing-quality sensitivity cohort sizes")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Interpolation cannot reconstruct unobserved movement in sampling gaps", fontsize=13)
    save_figure(fig, "timing_quality")


def build_html(payload):
    template = (HERE / "explorer_template.html").read_text()
    replacements = {"__PLOTLY_JS__": get_plotlyjs(),
                    "__PAYLOAD__": json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")}
    for token, figure in {"__OVERVIEW__": "cohort_overview", "__TEACHING__": "fft_teaching",
                          "__COHORT__": "cohort_spectra", "__WAVEFORMS__": "example_waveforms",
                          "__FOURIER__": "example_fourier", "__TIMING__": "timing_quality"}.items():
        replacements[token] = "data:image/png;base64," + base64.b64encode((FIGURES / f"{figure}.png").read_bytes()).decode()
    # One substitution pass prevents any inserted JS/data from being processed
    # as a template. The file embeds charts, plot library, data and all images.
    import re
    html = re.sub("|".join(map(re.escape, replacements)), lambda match: replacements[match.group()], template)
    (OUT / "index.html").write_text(html)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    dataset = PADSDataset()
    checks = run_checks()
    write_json(OUT / "fourier_checks.json", checks)
    counts = overview(dataset)
    examples = {}
    for condition, subject in dataset.example_ids().items():
        for task in TASKS:
            for wrist in ("LeftWrist", "RightWrist"):
                examples[(condition, task, wrist)] = analyze_recording(dataset.load_recording(subject, task, wrist), CONFIG)
    print("Computed 66 example recordings: three participants, 11 tasks, two wrists.", flush=True)
    plot_examples(examples)
    plot_fft_teaching()
    rows, cohort_qc, summaries, sensitivity = cohort_analysis(dataset, examples)
    plot_cohort(rows, summaries)
    plot_timing_quality(cohort_qc, sensitivity)
    qc_summary = {
        "cohort_recordings": len(cohort_qc),
        "recordings_with_gaps_over_1_5_intervals": sum(q["gaps_over_1_5_nominal_intervals"] > 0 for q in cohort_qc),
        "maximum_timestamp_gap_seconds": max(q["maximum_timestamp_gap_seconds"] for q in cohort_qc),
        "elapsed_time_rate_hz_range": [min(q["elapsed_time_rate_hz"] for q in cohort_qc),
                                       max(q["elapsed_time_rate_hz"] for q in cohort_qc)],
        "gap_flag_rule": "Raw adjacent timestamps differ by more than 1.5 / nominal_fs seconds.",
        "gap_policy": "Report and retain; no timestamp-quality exclusions were applied.",
        "sensitivity_threshold_seconds": GAP_SENSITIVITY_SECONDS,
        "sensitivity_policy": "Separate view retains a participant/task only if both raw wrists have maximum gap ≤ 30 ms.",
        "sensitivity_participant_counts": {key: value["n"] for key, value in sensitivity.items()},
    }
    payload = {"examples": dataset.example_ids(), "tasks": list(TASKS),
               "recordings": {"|".join((a.recording.participant_id, a.recording.task, a.recording.wrist)): example_payload(a)
                              for a in examples.values()},
               "cohort": summaries, "cohort_sensitive": sensitivity, "features": rows, "qc_summary": qc_summary}
    write_json(OUT / "demo_data.json", payload)
    write_csv(OUT / "example_recording_qc.csv", [a.qc for a in examples.values()])
    build_html(payload)
    print("Built offline HTML explorer and seven PNG/SVG scientific figures.", flush=True)
    from .notebook_builder import build_notebook
    notebook = build_notebook(HERE)
    all_qc = {q["source_relative_path"]: q for q in cohort_qc}
    all_qc.update({a.qc["source_relative_path"]: a.qc for a in examples.values()})
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PADS 1.0.0", "dataset_doi": "10.13026/m0w9-zx22", "license": "CC BY-NC-SA 4.0",
        "analysis_source": "Original raw time-series text; supplied preprocessed binaries not used.",
        "data_root": str(dataset.root), "config": asdict(CONFIG),
        "example_selection_rule": "Minimum published participant ID per condition, without inspecting spectra.",
        "example_ids": dataset.example_ids(), "example_recordings": len(examples), "condition_counts": counts,
        "cohort_participants": 304, "cohort_condition_counts": {"Parkinson's": 276, "Essential Tremor": 28},
        "cohort_tasks": list(COHORT_TASKS), "cohort_recordings": len(cohort_qc), "cohort_feature_rows": len(rows),
        "cohort_wrist_aggregation": "Arithmetic mean of the two wrist axis-summed PSDs before computing features.",
        "unique_source_recordings": len(all_qc),
        "all_used_raw_recording_checksums_verified": True,
        "source_checksums": {name: q["source_sha256"] for name, q in sorted(all_qc.items())},
        "timestamp_qc": qc_summary, "fourier_checks": checks, "notebook": notebook,
        "versions": {name: version(name) for name in ("numpy", "scipy", "matplotlib", "plotly", "nbformat", "nbclient", "ipykernel")},
        "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(HERE.glob("*.py"))},
        "source_files_modified": False, "models_trained": False,
        "interpretation": "Descriptive demonstration, not diagnostic performance validation.",
    }
    write_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"html": str(OUT / "index.html"), "notebook": notebook,
                      "timestamp_qc": qc_summary, "unique_source_recordings": len(all_qc)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
