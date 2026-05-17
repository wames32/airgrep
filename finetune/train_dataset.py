"""Training dataset: LibriSpeech clips, concatenated + augmented on the fly.

Design:
  - Read the manifest once to get (id, path, text, speaker, duration).
  - Group utterances by speaker so concatenations stay in-voice.
  - On __getitem__, pick a base utterance, then with P_CONCAT probability
    extend with 1-2 more utterances from the same speaker (separated by
    short inter-transmission silence) until total <= ~27s (under the
    Gemma 4 30s audio cap).
  - Run the concatenated waveform through a random radio-augmentation
    preset, sample fresh every step. This means the model sees a new
    noise realisation every epoch.
  - Return (audio_np_16k, transcript_str).

Concatenation creates natural multi-over transmissions with silence
between them, which is exactly what our SDR captures look like after the
squelch gate.
"""
from __future__ import annotations

import io
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from radio_augment import (
    augment, RadioAugmentConfig,
)


CACHE_DIR = Path(__file__).parent / "librispeech_cache"
MANIFEST = CACHE_DIR / "manifest.jsonl"
MANIFEST_PRETTY = CACHE_DIR / "manifest_pretty.jsonl"
SR = 16000
MAX_AUDIO_S = 27.0  # leave some headroom under Gemma 4's 30s cap
MIN_CONCAT_GAP_S = 0.6
MAX_CONCAT_GAP_S = 2.0
P_CONCAT = 0.4  # probability of building a multi-over clip


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest_merged(
    base: Path = MANIFEST, pretty: Path = MANIFEST_PRETTY
) -> list[dict]:
    """Load the base manifest, then merge in text_pretty from the beautifier
    output where available. Rows without a valid pretty transcript fall back
    to the original ALL-CAPS text.
    """
    rows = load_manifest(base)
    if not pretty.exists():
        print(f"[load_manifest_merged] No pretty manifest at {pretty}; using ALL-CAPS text.")
        return rows
    pretty_map: dict[str, str] = {}
    with open(pretty, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tp = r.get("text_pretty")
            if tp:
                pretty_map[r["id"]] = tp
    n_pretty = 0
    for r in rows:
        if r["id"] in pretty_map:
            r["text"] = pretty_map[r["id"]]
            n_pretty += 1
    print(f"[load_manifest_merged] {n_pretty}/{len(rows)} rows use beautified text.")
    return rows


def group_by_speaker(rows: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["speaker"]].append(r)
    return buckets


# ── Preset pool ─────────────────────────────────────────────────────────


def random_preset(rng: random.Random) -> RadioAugmentConfig:
    """Sample a random radio-condition preset for this step.

    Weighted to match realistic ham distribution: more 'nominal' than
    edge cases, but every mode represented.
    """
    mode = rng.choices(
        population=["nominal", "weak", "overmod", "clean", "squelched", "ducked"],
        weights=     [0.35,      0.20,    0.15,     0.15,    0.10,        0.05],
        k=1,
    )[0]

    if mode == "clean":
        return RadioAugmentConfig(
            snr_db=(20.0, 26.0), p_soft_clip=0.2, p_gain_drift=0.0,
        )
    if mode == "nominal":
        return RadioAugmentConfig(
            snr_db=(14.0, 22.0),
        )
    if mode == "weak":
        return RadioAugmentConfig(
            snr_db=(4.0, 12.0), p_gain_drift=0.8,
            gain_drift_max_db=(4.0, 8.0),
        )
    if mode == "overmod":
        return RadioAugmentConfig(
            snr_db=(10.0, 16.0), p_soft_clip=1.0,
            soft_clip_drive=(3.0, 5.5),
        )
    if mode == "squelched":
        return RadioAugmentConfig(
            snr_db=(10.0, 20.0), noise_mode="squelched",
        )
    if mode == "ducked":
        return RadioAugmentConfig(
            snr_db=(10.0, 16.0), noise_mode="ducked",
            duck_depth_db=(18.0, 28.0),
        )
    raise ValueError(mode)


# ── Audio IO ────────────────────────────────────────────────────────────


def load_flac(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"Expected {SR} Hz, got {sr} Hz in {path}"
    return audio.astype(np.float32)


def concat_with_silence(
    clips: list[np.ndarray], rng: random.Random
) -> np.ndarray:
    out = []
    for i, c in enumerate(clips):
        if i > 0:
            gap_s = rng.uniform(MIN_CONCAT_GAP_S, MAX_CONCAT_GAP_S)
            out.append(np.zeros(int(gap_s * SR), dtype=np.float32))
        out.append(c)
    return np.concatenate(out)


# ── Main dataset ────────────────────────────────────────────────────────


@dataclass
class Example:
    audio: np.ndarray  # 16 kHz float32
    text: str          # concatenated transcript


class RadioASRDataset(Dataset):
    """Map-style dataset. __getitem__ returns an Example."""

    def __init__(
        self,
        manifest_rows: list[dict],
        cache_dir: Path = CACHE_DIR,
        seed: int | None = None,
        length_override: int | None = None,
    ):
        self.rows = manifest_rows
        self.cache_dir = cache_dir
        self.speakers = group_by_speaker(manifest_rows)
        self._len = length_override or len(manifest_rows)
        self._seed_base = seed if seed is not None else 0

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Example:
        # Deterministic-ish per-step seed so behaviour is reproducible per run.
        rng = random.Random(self._seed_base + idx)
        np_rng = np.random.default_rng(self._seed_base + idx)

        base = self.rows[idx % len(self.rows)]
        base_audio = load_flac(self.cache_dir / base["path"])
        clips = [base_audio]
        texts = [base["text"]]
        total_dur = len(base_audio) / SR

        if rng.random() < P_CONCAT:
            speaker_pool = [r for r in self.speakers[base["speaker"]] if r["id"] != base["id"]]
            rng.shuffle(speaker_pool)
            for cand in speaker_pool[:4]:  # cap lookups
                # Account for the silence gap we'll insert
                gap = (MIN_CONCAT_GAP_S + MAX_CONCAT_GAP_S) / 2
                new_total = total_dur + gap + cand["duration"]
                if new_total > MAX_AUDIO_S:
                    continue
                clip = load_flac(self.cache_dir / cand["path"])
                clips.append(clip)
                texts.append(cand["text"])
                total_dur = new_total
                if len(clips) >= 3:  # cap at 3 overs per training clip
                    break

        audio = concat_with_silence(clips, rng)
        # Defensive final truncation
        if len(audio) > int(MAX_AUDIO_S * SR):
            audio = audio[: int(MAX_AUDIO_S * SR)]

        cfg = random_preset(rng)
        aug_audio = augment(audio, SR, cfg, seed=self._seed_base + idx)
        text = " ".join(texts).strip()

        return Example(audio=aug_audio, text=text)


def build_dataset(
    val_speakers: int = 6, seed: int = 0
) -> tuple[RadioASRDataset, RadioASRDataset]:
    """Split manifest into train/val by holding out entire speakers.

    Speaker-disjoint split is standard ASR practice so we measure
    generalisation to unheard voices.
    """
    rows = load_manifest_merged()
    by_spk = group_by_speaker(rows)
    speakers = sorted(by_spk.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)
    val_spk = set(speakers[:val_speakers])

    train = [r for r in rows if r["speaker"] not in val_spk]
    val = [r for r in rows if r["speaker"] in val_spk]
    print(f"Train: {len(train)} utterances ({len(set(r['speaker'] for r in train))} speakers)")
    print(f"Val:   {len(val)} utterances ({len(set(r['speaker'] for r in val))} speakers)")

    return (
        RadioASRDataset(train, seed=seed),
        RadioASRDataset(val, seed=seed + 1),
    )


if __name__ == "__main__":
    # Smoke test: load one example and write it for human inspection.
    rows = load_manifest()
    print(f"Loaded manifest: {len(rows)} rows")
    ds = RadioASRDataset(rows, seed=42)
    for i in range(3):
        ex = ds[i]
        print(f"  [{i}] dur={len(ex.audio)/SR:.1f}s  text={ex.text[:80]!r}...")
        sf.write(f"samples/train_preview_{i}.wav", ex.audio, SR)
    print("Wrote samples/train_preview_{0..2}.wav")
