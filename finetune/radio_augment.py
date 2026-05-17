"""Radio-degradation augmenter.

Turn a clean 16 kHz speech clip into something that plausibly came out
of our NBFM SDR pipeline (see ../capture.py).  The pipeline already
applies: FM demod -> squelch gate -> 4 kHz lowpass -> decimate to 48 kHz
-> peak normalize.  Our *clean* LibriSpeech input has none of the channel
distortions that occur before that pipeline, so we simulate them here.

Degradations applied (all randomised within bounds):
  1. Bandpass 300-3100 Hz (NBFM voice band; this is the largest single
     factor making radio speech sound "radio-ey" and also the main source
     of the phoneme confusion that breaks Gemma 4).
  2. Soft clipping (tanh) — overmodulation at the transmitter.
  3. Companding mismatch — slight treble hiss.
  4. Additive noise — pink + white mix at randomised SNR (5 - 25 dB).
  5. Slow gain drift — mic distance / fading.
  6. Pre-emphasis / de-emphasis near-mismatch.
  7. Peak normalization (matches what pipeline.py does).

The augmenter is deterministic given a seed so we can reproduce any
specific augmented clip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfilt, lfilter, firwin


SR = 16000  # Gemma 4 audio encoder expects 16 kHz mono


# ── Noise generation ─────────────────────────────────────────────────


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n samples of pink (1/f) noise via spectral shaping."""
    # White noise in frequency domain
    X = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    # 1/sqrt(f) amplitude shaping -> 1/f power
    f = np.arange(len(X))
    f[0] = 1  # avoid div by zero for DC
    X = X / np.sqrt(f)
    x = np.fft.irfft(X, n)
    x = x / (np.std(x) + 1e-9)
    return x.astype(np.float32)


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n).astype(np.float32)
    return x / (np.std(x) + 1e-9)


# ── Filters ──────────────────────────────────────────────────────────


def bandpass(x: np.ndarray, lo: float, hi: float, sr: int = SR, order: int = 6) -> np.ndarray:
    nyq = sr / 2
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def lowpass(x: np.ndarray, cutoff: float, sr: int = SR, order: int = 6) -> np.ndarray:
    nyq = sr / 2
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def preemph(x: np.ndarray, coef: float) -> np.ndarray:
    return np.append(x[0], x[1:] - coef * x[:-1]).astype(np.float32)


def deemph(x: np.ndarray, coef: float) -> np.ndarray:
    return lfilter([1.0], [1.0, -coef], x).astype(np.float32)


# ── Effects ──────────────────────────────────────────────────────────


def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    """Tanh-style soft clipping. drive=1 => subtle, drive=5 => crunchy."""
    return np.tanh(x * drive) / np.tanh(drive)


def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise to hit target SNR (computed vs speech RMS over non-silent frames)."""
    # RMS of "voiced" portion (top 50% by frame energy)
    frame = int(0.05 * SR)  # 50 ms
    n = len(signal) // frame
    if n < 2:
        sig_rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    else:
        rms = np.array([np.sqrt(np.mean(signal[i*frame:(i+1)*frame]**2)) for i in range(n)])
        voiced = rms[rms > np.median(rms)]
        sig_rms = np.mean(voiced) if len(voiced) else rms.mean()
    sig_rms = max(sig_rms, 1e-6)

    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / (np.sqrt(np.mean(noise ** 2)) + 1e-9))
    return (signal + noise).astype(np.float32)


def gain_drift(x: np.ndarray, rng: np.random.Generator, max_db: float = 6.0) -> np.ndarray:
    """Natural, aperiodic amplitude drift using low-pass-filtered noise.

    Real fading is a random walk, not a pure sine.  We generate white noise,
    low-pass it aggressively (cutoff 0.3-1.5 Hz), normalize to +/- 1, then
    scale to the target dB range.
    """
    n = len(x)
    noise = rng.standard_normal(n).astype(np.float32)
    cutoff_hz = rng.uniform(0.3, 1.5)
    # 2-pole Butterworth lowpass
    sos = butter(2, cutoff_hz / (SR / 2), btype="low", output="sos")
    drift = sosfilt(sos, noise)
    # Normalize to +/- 1
    m = np.max(np.abs(drift))
    if m > 0:
        drift = drift / m
    drift_db = max_db * drift
    drift_lin = 10 ** (drift_db / 20)
    return (x * drift_lin).astype(np.float32)


# ── Voice activity detection (simple energy-based) ───────────────────


def vad_mask(
    x: np.ndarray,
    sr: int = SR,
    frame_ms: float = 20.0,
    hangover_ms: float = 200.0,
    onset_ms: float = 60.0,
) -> np.ndarray:
    """Return a sample-level mask in [0, 1]: 1 where voice, 0 where silent.

    Simple energy threshold (speech is ~15+ dB louder than LibriSpeech's
    room-silence floor), then extend forward/backward so we don't clip
    word starts/ends, and crossfade for smooth transitions.
    """
    frame = int(sr * frame_ms / 1000)
    n_frames = len(x) // frame
    if n_frames < 2:
        return np.ones_like(x, dtype=np.float32)

    rms = np.array([
        np.sqrt(np.mean(x[i*frame:(i+1)*frame] ** 2))
        for i in range(n_frames)
    ])
    # Threshold: 4x the 20th-percentile RMS (robust noise-floor estimate)
    floor = np.percentile(rms, 20) + 1e-6
    thr = max(4.0 * floor, 0.01 * rms.max())
    active = rms > thr

    # Hangover: keep active for N ms after last frame > thr
    hang = max(1, int(hangover_ms / frame_ms))
    onset = max(1, int(onset_ms / frame_ms))
    extended = active.copy()
    # Forward extension (hangover)
    count = 0
    for i in range(len(extended)):
        if active[i]:
            count = hang
        elif count > 0:
            extended[i] = True
            count -= 1
    # Backward extension (onset)
    count = 0
    for i in range(len(extended) - 1, -1, -1):
        if active[i]:
            count = onset
        elif count > 0:
            extended[i] = True
            count -= 1

    # Expand to sample-level mask, then smooth with a short triangular kernel
    mask = np.repeat(extended.astype(np.float32), frame)
    # Pad to full length
    if len(mask) < len(x):
        mask = np.concatenate([mask, np.full(len(x) - len(mask), mask[-1])])
    else:
        mask = mask[:len(x)]
    # Smooth transitions over ~30 ms
    smooth_n = int(0.03 * sr)
    if smooth_n > 1:
        kernel = np.ones(smooth_n, dtype=np.float32) / smooth_n
        mask = np.convolve(mask, kernel, mode="same")
    return mask


def peak_norm(x: np.ndarray) -> np.ndarray:
    p = np.max(np.abs(x))
    return (x / p).astype(np.float32) if p > 0 else x


# ── Top-level augmenter ──────────────────────────────────────────────


@dataclass
class RadioAugmentConfig:
    """Probability and range of each effect. Ranges are sampled uniformly.

    The `noise_mode` flag controls how static relates to voice activity:
      - "constant":  steady static throughout, independent of voice (default).
      - "squelched": static only while voice is active; true silence elsewhere.
                     Matches the output of our SDR squelch gate.
      - "ducked":    steady static everywhere *except* during voice — speaker
                     comes through clean, static returns when they stop.
                     Models noise-blanker / aggressive-DSP radios.
    """
    bandpass_lo: tuple[float, float] = (280.0, 350.0)
    bandpass_hi: tuple[float, float] = (2700.0, 3200.0)

    # SNR in dB — lower is noisier
    snr_db: tuple[float, float] = (6.0, 24.0)
    # Fraction pink (rest is white) in the noise mix
    pink_fraction: tuple[float, float] = (0.5, 1.0)

    # Probability each effect is applied
    p_soft_clip: float = 0.6
    soft_clip_drive: tuple[float, float] = (1.5, 4.0)

    # Voice-only drift (mic distance / breathing). Applied to signal
    # BEFORE noise mixing so the noise floor stays steady.
    p_gain_drift: float = 0.5
    gain_drift_max_db: tuple[float, float] = (2.0, 6.0)

    # Simulate pre-emph / de-emph mismatch (leftover treble boost)
    p_treble_tilt: float = 0.7
    treble_coef: tuple[float, float] = (0.2, 0.7)  # preemph coef — treble boost

    # Simulate a final squelch/lowpass stage (matches pipeline.py 4 kHz lowpass)
    final_lowpass_hz: float = 4000.0

    # See class docstring.
    noise_mode: str = "constant"  # "constant" | "squelched" | "ducked"
    # In "ducked" mode, how much we attenuate static under the voice (dB).
    duck_depth_db: tuple[float, float] = (18.0, 30.0)


def augment(
    clean: np.ndarray,
    sr: int,
    cfg: RadioAugmentConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Apply random radio-channel degradations to a clean speech clip.

    Parameters
    ----------
    clean : np.ndarray
        Mono float32 speech at 16 kHz.
    sr : int
        Sample rate. Must be 16000.
    cfg : RadioAugmentConfig
    seed : int
        Seed for deterministic output.

    Returns
    -------
    np.ndarray
        Float32, 16 kHz, peak-normalized to [-1, 1].
    """
    assert sr == SR, f"Expected 16 kHz, got {sr}"
    cfg = cfg or RadioAugmentConfig()
    rng = np.random.default_rng(seed)

    x = clean.astype(np.float32).copy()

    # 1. Bandpass to NBFM voice band
    lo = rng.uniform(*cfg.bandpass_lo)
    hi = rng.uniform(*cfg.bandpass_hi)
    x = bandpass(x, lo, hi)

    # 2. Optional treble tilt (pre-emph mismatch)
    if rng.random() < cfg.p_treble_tilt:
        coef = rng.uniform(*cfg.treble_coef)
        x = preemph(x, coef)
        x = deemph(x, coef * 0.7)  # 70% undo — leaves slight treble bias

    # 3. Optional soft clip
    if rng.random() < cfg.p_soft_clip:
        p = np.max(np.abs(x))
        if p > 0:
            x = x / p
        drive = rng.uniform(*cfg.soft_clip_drive)
        x = soft_clip(x, drive)

    # 4. Voice-only gain drift (applied to signal BEFORE noise so noise
    #    floor stays steady — fixes the periodic "wobble" reported by the
    #    human reviewer).
    if rng.random() < cfg.p_gain_drift:
        max_db = rng.uniform(*cfg.gain_drift_max_db)
        x = gain_drift(x, rng, max_db=max_db)

    # 5. Build the noise track (pink + white mix)
    snr = rng.uniform(*cfg.snr_db)
    p_pink = rng.uniform(*cfg.pink_fraction)
    pink = pink_noise(len(x), rng) * p_pink
    white = white_noise(len(x), rng) * (1 - p_pink)
    noise = pink + white

    # 6. Mix signal + noise according to the noise_mode.
    if cfg.noise_mode == "constant":
        y = add_noise_at_snr(x, noise, snr)

    elif cfg.noise_mode == "squelched":
        # Real NBFM squelch gates the channel closed between transmissions:
        # true silence outside of voice, voice+static when the carrier is up.
        mask = vad_mask(x)
        # Scale noise so SNR is correct when we DO have voice
        voiced = add_noise_at_snr(x, noise, snr)
        # Outside of voice, go to hard silence (matches capture.py squelch_gate).
        y = voiced * mask

    elif cfg.noise_mode == "ducked":
        # Static is the *background*; it ducks under the voice when the
        # speaker is talking, then comes back up. The voice itself comes
        # through clean, not buried in static.
        mask = vad_mask(x)
        duck_db = rng.uniform(*cfg.duck_depth_db)
        duck_lin = 10 ** (-duck_db / 20)
        # Scale noise so *background* RMS matches our target SNR vs voice RMS.
        sig_rms = max(np.sqrt(np.mean(x ** 2) + 1e-12), 1e-6)
        target_noise_rms = sig_rms / (10 ** (snr / 20))
        noise = noise * (target_noise_rms / (np.sqrt(np.mean(noise ** 2)) + 1e-9))
        # Duck the noise by (duck_depth) dB where mask is 1.
        noise_env = 1.0 - mask * (1.0 - duck_lin)
        y = x + noise * noise_env

    else:
        raise ValueError(f"Unknown noise_mode: {cfg.noise_mode}")

    # 7. Final lowpass (matches pipeline.py's nbfm_lowpass).
    y = lowpass(y, cfg.final_lowpass_hz)

    # 8. Peak normalise.
    return peak_norm(y)


def augment_batch(
    clean: np.ndarray, sr: int, n: int, cfg: RadioAugmentConfig | None = None,
    base_seed: int = 0,
) -> list[np.ndarray]:
    """Return n distinct random augmentations of the same clean clip."""
    return [augment(clean, sr, cfg, seed=base_seed + i) for i in range(n)]
