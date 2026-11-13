"""Load PADS raw data and calculate explicitly normalized Fourier spectra."""

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/pads/1.0.0"
SENSORS = {"Accelerometer": (slice(0, 3), "g"), "Gyroscope": (slice(3, 6), "rad/s")}
TASKS = ("Relaxed", "RelaxedTask", "StretchHold", "LiftHold", "HoldWeight", "PointFinger",
         "DrinkGlas", "CrossArms", "TouchIndex", "TouchNose", "Entrainment")
CONDITIONS = ("Parkinson's", "Essential Tremor", "Healthy")
COLORS = {"Parkinson's": "#2563eb", "Essential Tremor": "#dc6844", "Healthy": "#148476"}


@dataclass(frozen=True)
class SpectralConfig:
    crop_seconds: float = 0.5
    welch_segment_samples: int = 400
    exploratory_band_hz: tuple[float, float] = (3.0, 12.0)
    reference_band_hz: tuple[float, float] = (1.0, 20.0)


@dataclass
class RawRecording:
    participant_id: str
    condition: str
    task: str
    wrist: str
    sampling_rate_hz: float
    time: np.ndarray
    values: np.ndarray  # rows by six sensor axes
    source: Path
    source_sha256: str


@dataclass
class Analysis:
    recording: RawRecording
    time: np.ndarray
    interpolated_values: np.ndarray
    detrended_values: np.ndarray
    fft_frequency: np.ndarray
    fft_amplitude: np.ndarray
    fft_psd: np.ndarray
    welch_frequency: np.ndarray
    welch_psd: np.ndarray
    qc: dict


class PADSDataset:
    def __init__(self, root=DEFAULT_DATA_ROOT):
        self.root = Path(root).resolve()
        with (self.root / "preprocessed/file_list.csv").open(newline="") as stream:
            self.participants = list(csv.DictReader(stream))
        self.by_id = {row["id"]: row for row in self.participants}
        self.checksums = {}
        for line in (self.root / "SHA256SUMS.txt").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            self.checksums[name] = digest

    def example_ids(self):
        """Smallest published ID in each condition; no signal-based selection."""
        return {condition: min(r["id"] for r in self.participants if r["condition"] == condition)
                for condition in CONDITIONS}

    def load_recording(self, participant_id, task="Relaxed", wrist="LeftWrist"):
        participant_id = str(participant_id).zfill(3)
        participant = self.by_id[participant_id]
        metadata_path = self.root / f"movement/observation_{participant_id}.json"
        observation = json.loads(metadata_path.read_text())
        session = next((s for s in observation["session"] if s["record_name"] == task), None)
        if session is None:
            raise ValueError(f"Task {task!r} is not present for participant {participant_id}.")
        record = next((r for r in session["records"] if r["device_location"] == wrist), None)
        if record is None:
            raise ValueError(f"Wrist {wrist!r} is not present for participant {participant_id}.")
        relative = Path("movement") / record["file_name"]
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Recording reference escapes the dataset directory.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.checksums[relative.as_posix()]:
            raise ValueError(f"Source checksum mismatch: {relative}")
        data = np.loadtxt(path, delimiter=",", dtype=np.float64)
        expected_channels = ["Time", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",
                             "Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z"]
        if record["channels"] != expected_channels or data.shape != (session["rows"], 7):
            raise ValueError(f"Unexpected recording schema: {relative}")
        if not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
            raise ValueError(f"Nonfinite values or nonincreasing timestamps: {relative}")
        return RawRecording(participant_id, participant["condition"], task, wrist,
                            float(observation["sampling_rate"]), data[:, 0], data[:, 1:], path, digest)


def fourier_spectra(values, fs):
    """Return full-record Hann amplitude and one-sided PSD, with no zero padding.

    `values` is a real (time, channels) array. Windowed PSD integrates (by the
    sum of bins times df) to the window-weighted mean square, not the unweighted
    sample variance. DC and the even-length Nyquist bin are not doubled.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 4 or not np.isfinite(values).all() or fs <= 0:
        raise ValueError("Expected finite real data with at least four rows and positive fs.")
    n = values.shape[0]
    window = signal.windows.hann(n, sym=False)
    transformed = np.fft.rfft(values * window[:, None], axis=0)
    factors = np.full(n // 2 + 1, 2.0)
    factors[0] = 1.0
    if n % 2 == 0:
        factors[-1] = 1.0
    amplitude = np.abs(transformed) * factors[:, None] / window.sum()
    psd = np.abs(transformed) ** 2 * factors[:, None] / (fs * np.sum(window ** 2))
    return np.fft.rfftfreq(n, 1.0 / fs), amplitude, psd


def analyze_recording(recording, config=SpectralConfig()):
    if config.crop_seconds < 0:
        raise ValueError("Crop duration must be nonnegative.")
    t, raw, fs = recording.time, recording.values, recording.sampling_rate_hz
    start = t[0] + config.crop_seconds
    # Fixed 100 Hz grid, inside original support. Never extrapolate. Interpolation
    # accounts for timestamp jitter; it cannot recover unobserved motion in gaps.
    n = int(np.floor((t[-1] - start) * fs)) + 1
    if n < config.welch_segment_samples or config.welch_segment_samples < 4:
        raise ValueError("Recording is too short for the requested Welch segment.")
    grid = start + np.arange(n) / fs
    grid = grid[grid <= t[-1]]
    values = np.column_stack([np.interp(grid, t, raw[:, j]) for j in range(6)])
    detrended = signal.detrend(values, axis=0, type="linear")
    f_fft, amplitude, fft_psd = fourier_spectra(detrended, fs)
    length = config.welch_segment_samples
    overlap = length // 2
    f_welch, welch_psd = signal.welch(
        detrended, fs=fs, window=signal.windows.hann(length, sym=False),
        nperseg=length, noverlap=overlap, nfft=length, detrend="constant",
        return_onesided=True, scaling="density", axis=0, average="mean")
    delta = np.diff(t)
    qc = {
        "participant_id": recording.participant_id, "condition": recording.condition,
        "task": recording.task, "wrist": recording.wrist,
        "source_relative_path": recording.source.relative_to(DEFAULT_DATA_ROOT).as_posix()
            if recording.source.is_relative_to(DEFAULT_DATA_ROOT) else str(recording.source),
        "source_sha256": recording.source_sha256,
        "nominal_sampling_rate_hz": fs,
        "median_timestamp_interval_seconds": float(np.median(delta)),
        "median_interval_rate_hz": float(1 / np.median(delta)),
        "elapsed_time_rate_hz": float((len(t) - 1) / (t[-1] - t[0])),
        "maximum_timestamp_gap_seconds": float(delta.max()),
        "gaps_over_1_5_nominal_intervals": int(np.sum(delta > 1.5 / fs)),
        "raw_samples": len(t), "analysis_samples": len(grid),
        "analysis_start_seconds": float(grid[0]), "analysis_end_seconds": float(grid[-1]),
        "crop_seconds": config.crop_seconds, "fft_grid_spacing_hz": float(fs / len(grid)),
        "welch_segment_samples": length, "welch_overlap_samples": overlap,
        "welch_segments": int(1 + (len(grid) - length) // (length - overlap)),
        "welch_grid_spacing_hz": float(fs / length),
    }
    return Analysis(recording, grid, values, detrended, f_fft, amplitude, fft_psd,
                    f_welch, welch_psd, qc)


def vector_psd(analysis, sensor="Accelerometer", method="welch"):
    """Sum three axis PSDs; do not Fourier-transform the nonlinear vector norm."""
    axes, _ = SENSORS[sensor]
    if method == "welch":
        return analysis.welch_frequency, analysis.welch_psd[:, axes].sum(axis=1)
    if method == "fft":
        return analysis.fft_frequency, analysis.fft_psd[:, axes].sum(axis=1)
    raise ValueError("method must be 'welch' or 'fft'")


def spectral_features(frequency, psd, config=SpectralConfig()):
    """Features of a supplied one-sided PSD. Bands include their endpoints.

    Powers use sum of included bin densities times df, a defined rectangular
    bin convention. The peak is an algorithmic maximum, not a tremor diagnosis.
    """
    df = float(frequency[1] - frequency[0])
    band = (frequency >= config.exploratory_band_hz[0]) & (frequency <= config.exploratory_band_hz[1])
    reference = (frequency >= config.reference_band_hz[0]) & (frequency <= config.reference_band_hz[1])
    if not np.any(band) or not np.all(np.isfinite(psd)) or np.any(psd < 0):
        raise ValueError("Invalid PSD or empty analysis band.")
    band_power = float(psd[band].sum() * df)
    reference_power = float(psd[reference].sum() * df)
    if band_power <= 0 or reference_power <= 0:
        return {"peak_frequency_hz": None, "band_power": band_power,
                "reference_power": reference_power, "band_power_fraction": None}
    return {"peak_frequency_hz": float(frequency[band][np.argmax(psd[band])]),
            "band_power": band_power, "reference_power": reference_power,
            "band_power_fraction": float(band_power / reference_power)}
