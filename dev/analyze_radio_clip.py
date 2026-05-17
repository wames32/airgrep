"""Inspect the real NBFM debug clip's spectral properties so the
augmenter's bandpass / noise profiles target realistic values.
"""
from pathlib import Path
import numpy as np
import soundfile as sf

WAV = Path(__file__).resolve().parents[1] / "debug_audio" / "cycle_20260412_223734_144.35MHz.wav"

def main():
    x, sr = sf.read(str(WAV))
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    print(f"File: {WAV.name}  sr={sr}  dur={dur:.1f}s  samples={len(x)}")
    print(f"Peak: {np.max(np.abs(x)):.3f}  RMS: {np.sqrt(np.mean(x**2)):.3f}")

    # Rough SNR via: find quiet sections (lowest 20% RMS in 100ms windows) = noise
    win = int(0.1 * sr)
    n_win = len(x) // win
    rms = np.array([np.sqrt(np.mean(x[i*win:(i+1)*win]**2)) for i in range(n_win)])
    noise_rms = np.mean(np.sort(rms)[: max(1, n_win // 5)])
    signal_rms = np.mean(np.sort(rms)[-max(1, n_win // 5):])
    if noise_rms > 1e-6:
        snr_db = 20 * np.log10(signal_rms / noise_rms)
    else:
        snr_db = float("inf")
    print(f"Approx noise RMS: {noise_rms:.4f}   signal RMS: {signal_rms:.4f}")
    print(f"Approx SNR:       {snr_db:.1f} dB")

    # Spectral content: FFT whole-clip, find -3 dB bandwidth
    from numpy.fft import rfft, rfftfreq
    X = np.abs(rfft(x))
    f = rfftfreq(len(x), 1 / sr)
    # Smooth to ~50 Hz bins
    bin_width = 50
    n_bins = int((sr / 2) / bin_width)
    bands = np.interp(
        np.linspace(f.min(), f.max(), n_bins),
        f, X,
    )
    peak = bands.max()
    above = np.where(bands > 0.1 * peak)[0]
    if len(above):
        f_lo = above[0] * bin_width
        f_hi = above[-1] * bin_width
        print(f"Spectral energy >10% of peak between ~{f_lo} Hz and ~{f_hi} Hz")

    # Report energy fraction in key bands
    def band_energy(lo, hi):
        m = (f >= lo) & (f < hi)
        return (X[m] ** 2).sum()
    total = (X ** 2).sum()
    for lo, hi in [(0, 100), (100, 300), (300, 3000), (3000, 5000), (5000, 8000)]:
        print(f"  {lo:>4}-{hi:<4} Hz: {100*band_energy(lo, hi)/total:5.2f}%")

if __name__ == "__main__":
    main()
