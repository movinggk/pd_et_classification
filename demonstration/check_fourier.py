"""Numerical checks of the spectral normalization and resampling contract."""

import json
from pathlib import Path
import numpy as np
from scipy import signal

from .pads_demo import (PADSDataset, SpectralConfig, analyze_recording, fourier_spectra,
                        vector_psd, spectral_features)


def run_checks():
    checks = []
    fs = 100.0
    rng = np.random.default_rng(20261007)
    for n in (400, 401):
        x = rng.normal(size=(n, 3))
        f, amplitude, p = fourier_spectra(x, fs)
        window = signal.windows.hann(n, sym=False)
        sf, sp = signal.periodogram(x, fs=fs, window=window, nfft=n,
                                   detrend=False, return_onesided=True, scaling="density", axis=0)
        np.testing.assert_allclose(f, sf)
        np.testing.assert_allclose(p, sp, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(p.sum(axis=0) * fs / n,
                                   ((x * window[:, None]) ** 2).sum(axis=0) / (window ** 2).sum(),
                                   rtol=1e-12)
        checks.append(f"Length {n}: one-sided PSD equals SciPy periodogram and satisfies windowed Parseval identity.")
    n = 400
    t = np.arange(n) / fs
    f, amplitude, p = fourier_spectra((2 * np.sin(2 * np.pi * 6 * t))[:, None], fs)
    peak = int(np.argmax(amplitude[:, 0]))
    assert f[peak] == 6.0
    np.testing.assert_allclose(amplitude[peak, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(p.sum() * fs / n, 2.0, atol=1e-12)
    checks.append("Known 6 Hz sine: recovered 6 Hz peak, amplitude 2, and integrated power 2.")
    _, a_dc, _ = fourier_spectra(np.ones((n, 1)), fs)
    _, a_nyq, _ = fourier_spectra(((-1.0) ** np.arange(n))[:, None], fs)
    np.testing.assert_allclose(a_dc[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(a_nyq[-1, 0], 1.0, atol=1e-12)
    checks.append("DC and even-length Nyquist amplitudes are not doubled.")
    dataset = PADSDataset()
    recording = dataset.load_recording("066", "StretchHold", "RightWrist")
    analysis = analyze_recording(recording)
    np.testing.assert_allclose(np.diff(analysis.time), 1 / fs, atol=1e-12)
    assert analysis.time[0] >= recording.time[0] + 0.5
    assert analysis.time[-1] <= recording.time[-1]
    assert analysis.time.shape[0] == analysis.detrended_values.shape[0]
    assert analysis.welch_frequency[1] == 0.25
    checks.append("Real ET recording: uniform 100 Hz grid, 0.5 s crop, no extrapolation, and 0.25 Hz Welch spacing.")
    f, total = vector_psd(analysis)
    np.testing.assert_allclose(total, analysis.welch_psd[:, :3].sum(axis=1))
    features = spectral_features(f, total)
    assert 0 <= features["band_power_fraction"] <= 1
    checks.append("Three-axis PSD sum and exploratory/reference band-power fraction are consistent.")
    return {"passed": True, "check_count": len(checks), "checks": checks}


if __name__ == "__main__":
    report = run_checks()
    destination = Path(__file__).resolve().parent / "outputs/fourier_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
