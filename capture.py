"""SDR signal processing: IQ capture, FM demodulation, filtering, decimation.

This module provides the DSP functions used by pipeline.py to turn raw
RTL-SDR IQ samples into audio WAV files.  It has no CLI -- use pipeline.py
to run the full monitor, or import these functions directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import decimate, firwin, lfilter

if TYPE_CHECKING:
    from rtlsdr import RtlSdr as RtlSdrType


def is_wbfm(freq_mhz: float) -> bool:
    """Return True if the frequency falls in the FM broadcast band."""
    return 88.0 <= freq_mhz <= 108.0


def fm_demodulate(iq: np.ndarray) -> np.ndarray:
    """FM discriminator demodulation via instantaneous phase difference."""
    product = iq[1:] * np.conj(iq[:-1])
    return np.angle(product)


def detect_signal(iq: np.ndarray) -> float:
    """Noise squelch score -- lower values indicate a signal is present.

    Measures high-frequency noise energy after FM demodulation.
    Real FM signals suppress HF noise; pure noise produces maximum HF energy.
    Typical values: noise ~1.0-1.5, signal ~0.05-0.3.
    """
    demod = np.angle(iq[1:] * np.conj(iq[:-1]))
    hf = np.diff(demod)
    return float(np.sqrt(np.mean(hf ** 2)))


def de_emphasis_filter(audio: np.ndarray, sample_rate: float, tau: float = 75e-6) -> np.ndarray:
    """Apply a single-pole IIR de-emphasis filter (75us for US FM broadcast)."""
    dt = 1.0 / sample_rate
    alpha = dt / (tau + dt)
    return lfilter([alpha], [1, -(1 - alpha)], audio)


def nbfm_lowpass(audio: np.ndarray, sample_rate: float, cutoff: float = 4000.0) -> np.ndarray:
    """Lowpass filter for NBFM — removes noise above voice bandwidth.

    NBFM voice occupies ~300-3000 Hz.  After FM demod + decimation to
    48 kHz, frequencies from 4 kHz up to 24 kHz are pure noise that
    dominates the peak amplitude and makes voice inaudible after
    normalization.  Cutting at 4 kHz fixes this.
    """
    taps = firwin(101, cutoff, fs=sample_rate)
    return lfilter(taps, 1.0, audio)


def factorize_decimation(factor: int, max_stage: int = 13) -> list[int]:
    """Break a decimation factor into stages, each <= max_stage.

    Uses prime factorization so the product of stages == factor exactly.
    """
    stages = []
    for p in (2, 3, 5, 7, 11, 13):
        while factor % p == 0:
            stages.append(p)
            factor //= p
    if factor > 1:
        stages.append(factor)
    return stages


def decimate_audio(audio: np.ndarray, factor: int) -> np.ndarray:
    """Decimate in stages, each <= 13 (scipy fir limit)."""
    stages = factorize_decimation(factor)
    for stage in stages:
        ftype = "iir" if stage > 13 else "fir"
        audio = decimate(audio, stage, ftype=ftype)
    return audio


def fft_scan(
    sdr: RtlSdrType,
    start_mhz: float,
    end_mhz: float,
    sample_rate: float,
    channel_khz: float = 5.0,
    threshold_db: float = 10.0,
    fft_size: int = 16384,
    on_chunk: 'Callable[[float, float], None] | None' = None,
) -> list[tuple[float, float]]:
    """Wideband FFT scan — find active channels across a frequency range.

    Instead of retuning per-channel (slow), this captures one wideband IQ
    chunk per ``sample_rate``-wide slice and uses an FFT to detect energy
    in every channel simultaneously.

    For 144-148 MHz at 960 kHz sample rate: 5 chunks × ~15 ms ≈ 75 ms
    versus 800 retunes × 12 ms ≈ 9.6 s with the per-channel approach.

    Parameters
    ----------
    sdr : RtlSdr
        Open RTL-SDR device handle (sample_rate and gain already set).
    start_mhz, end_mhz : float
        Frequency range to scan.
    sample_rate : float
        SDR sample rate in Hz (determines bandwidth per chunk).
    channel_khz : float
        Channel width in kHz for binning (default 5 kHz for NBFM).
    threshold_db : float
        How many dB above the noise floor a channel must be to count
        as active (default 6 dB ≈ 4× power above noise).
    fft_size : int
        FFT length.  16384 at 960 kHz → ~59 Hz bin resolution.

    Returns
    -------
    list of (freq_mhz, power_db)
        Active channels sorted by descending power.
    """
    usable_bw = sample_rate * 0.8          # ignore edges (alias/rolloff)
    step_hz = usable_bw                     # hop by usable bandwidth
    center = start_mhz * 1e6 + sample_rate / 2  # first chunk center

    active: list[tuple[float, float]] = []

    while center - sample_rate / 2 < end_mhz * 1e6:
        sdr.center_freq = center

        # Report scan progress (chunk low edge → high edge in MHz)
        if on_chunk is not None:
            lo = (center - sample_rate / 2) / 1e6
            hi = (center + sample_rate / 2) / 1e6
            on_chunk(lo, hi)

        # Flush PLL settle + capture
        sdr.read_samples(4096)              # flush stale IQ (~4 ms)
        iq = sdr.read_samples(fft_size)     # one FFT frame

        # Power spectral density via FFT
        window = np.hanning(len(iq))
        spectrum = np.fft.fftshift(np.fft.fft(iq * window))
        psd = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-20)

        # Map FFT bins to absolute frequencies
        freqs = np.fft.fftshift(
            np.fft.fftfreq(len(iq), d=1.0 / sample_rate)
        ) + center

        # Only consider the usable center 80% (avoid aliasing at edges)
        edge = int(len(psd) * 0.1)
        psd = psd[edge:-edge]
        freqs = freqs[edge:-edge]

        # Bin into channels
        channel_hz = channel_khz * 1000.0
        bins_per_channel = max(1, int(channel_hz / (sample_rate / len(iq))))

        n_full = (len(psd) // bins_per_channel) * bins_per_channel
        psd_trimmed = psd[:n_full].reshape(-1, bins_per_channel)
        freqs_trimmed = freqs[:n_full].reshape(-1, bins_per_channel)

        channel_power = psd_trimmed.mean(axis=1)
        channel_center = freqs_trimmed.mean(axis=1)

        # Noise floor = median channel power
        noise_floor = float(np.median(channel_power))

        # Find channels above threshold
        for pwr, freq_hz in zip(channel_power, channel_center):
            if pwr - noise_floor >= threshold_db:
                freq_m = freq_hz / 1e6
                # Only include if within requested range
                if start_mhz <= freq_m <= end_mhz:
                    active.append((freq_m, float(pwr)))

        center += step_hz

    # Sort by power (strongest first), deduplicate nearby freqs
    active.sort(key=lambda x: -x[1])
    deduped: list[tuple[float, float]] = []
    for freq, pwr in active:
        if not any(abs(freq - f) < channel_khz / 1000.0 for f, _ in deduped):
            deduped.append((freq, pwr))

    return deduped


def capture_iq(sdr: RtlSdrType, num_samples: int) -> np.ndarray:
    """Read IQ samples from the SDR in chunks."""
    chunk_size = 131072  # 128K samples per read
    collected = []
    remaining = num_samples

    while remaining > 0:
        n = min(chunk_size, remaining)
        samples = sdr.read_samples(n)
        collected.append(samples)
        remaining -= len(samples)

    return np.concatenate(collected)


def squelch_gate(
    iq: np.ndarray,
    audio: np.ndarray,
    sample_rate: float,
    frame_ms: float = 50.0,
    threshold: float = 0.4,
) -> np.ndarray:
    """Zero out audio frames where the IQ shows no signal (noise only).

    Uses the same HF-noise metric as detect_signal().  Frames scoring
    above *threshold* are gated to zero so that subsequent peak
    normalization is driven by the voice, not the noise floor.
    """
    frame_samples = int(sample_rate * frame_ms / 1000)
    # audio is 1 sample shorter than iq (demod drops first sample)
    n = min(len(iq), len(audio))
    n_frames = n // frame_samples

    gated = audio.copy()
    for i in range(n_frames):
        s = i * frame_samples
        e = s + frame_samples
        score = detect_signal(iq[s:e])
        if score >= threshold:
            gated[s:e] = 0.0

    # Gate the remaining tail (too short to score reliably)
    tail_start = n_frames * frame_samples
    if tail_start < len(gated):
        gated[tail_start:] = 0.0

    return gated


def process(iq: np.ndarray, sample_rate: float, audio_rate: int, wbfm: bool) -> np.ndarray:
    """Full processing pipeline: demod -> gate -> filter -> decimate -> normalize."""
    audio = fm_demodulate(iq)

    if wbfm:
        audio = de_emphasis_filter(audio, sample_rate)
    else:
        # Gate noise at full sample rate (before decimation) using IQ
        audio = squelch_gate(iq, audio, sample_rate)

    dec_factor = int(sample_rate / audio_rate)
    if dec_factor > 1:
        audio = decimate_audio(audio, dec_factor)

    if not wbfm:
        audio = nbfm_lowpass(audio, audio_rate)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)
