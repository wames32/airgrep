"""Generate sample augmented clips for human review.

Streams a handful of LibriSpeech examples (avoids waiting for the full
100h download), augments each with a range of radio-degradation
severities, and writes WAVs + a manifest to samples/.

Review target: the user will listen to these and tell us if they sound
plausibly like what comes out of the SDR NBFM pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import io

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

from radio_augment import augment, RadioAugmentConfig

OUT_DIR = Path(__file__).parent / "samples"
N_CLIPS = 5  # distinct sentences
VARIANTS_PER_CLIP = 4  # distinct radio conditions


def presets() -> list[tuple[str, RadioAugmentConfig]]:
    """Plausibly-realistic conditions ranging clean->very noisy, plus two
    gated modes that reflect how NBFM squelch / noise-blanker radios behave.
    """
    return [
        ("clean_link",    RadioAugmentConfig(
            snr_db=(22.0, 26.0), p_soft_clip=0.2, p_gain_drift=0.0)),
        ("nominal_ham",   RadioAugmentConfig(
            snr_db=(14.0, 20.0))),
        ("weak_signal",   RadioAugmentConfig(
            snr_db=(6.0, 12.0), p_gain_drift=0.8,
            gain_drift_max_db=(4.0, 8.0))),
        ("overmodulated", RadioAugmentConfig(
            snr_db=(10.0, 16.0), p_soft_clip=1.0,
            soft_clip_drive=(3.0, 5.0))),
        # Real NBFM squelch: silence between transmissions, noise+voice when keyed.
        ("squelched",     RadioAugmentConfig(
            snr_db=(10.0, 18.0), noise_mode="squelched")),
        # Aggressive noise-blanker: static is the floor, voice comes through clean.
        ("ducked",        RadioAugmentConfig(
            snr_db=(10.0, 16.0), noise_mode="ducked",
            duck_depth_db=(18.0, 28.0))),
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Streaming a few LibriSpeech clean samples (raw-bytes mode)...")
    ds = load_dataset(
        "openslr/librispeech_asr", "clean",
        split="validation", streaming=True,
    )
    # Disable torchcodec-based decoding — we'll decode bytes ourselves with soundfile
    ds = ds.cast_column("audio", Audio(decode=False))

    manifest = []
    preset_list = presets()

    it = iter(ds)
    for clip_idx in range(N_CLIPS):
        ex = next(it)
        # With decode=False, ex["audio"] is {"bytes": ..., "path": ...}
        audio_info = ex["audio"]
        if audio_info.get("bytes"):
            buf = io.BytesIO(audio_info["bytes"])
            audio, sr = sf.read(buf)
        else:
            audio, sr = sf.read(audio_info["path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        text = ex["text"]
        clip_id = ex.get("id", f"clip{clip_idx}")
        print(f"\n[{clip_idx+1}/{N_CLIPS}] {clip_id}  ({len(audio)/sr:.1f}s)")
        print(f"  Transcript: {text}")

        # Save the clean reference
        clean_name = f"{clip_idx:02d}_{clip_id}_00_clean.wav"
        sf.write(OUT_DIR / clean_name, audio, sr)
        manifest.append({
            "file": clean_name, "kind": "clean",
            "preset": None, "transcript": text, "clip_id": clip_id,
        })

        # Save each variant
        for v_idx, (preset_name, cfg) in enumerate(preset_list):
            aug = augment(audio, sr, cfg, seed=1000 * clip_idx + v_idx)
            # randomly pick 1 or 2 extra seeds for within-preset variety
            v_name = f"{clip_idx:02d}_{clip_id}_{v_idx+1:02d}_{preset_name}.wav"
            sf.write(OUT_DIR / v_name, aug, sr)
            manifest.append({
                "file": v_name, "kind": "augmented",
                "preset": preset_name, "transcript": text, "clip_id": clip_id,
            })
            print(f"    -> {preset_name}")

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(manifest)} files + manifest.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
