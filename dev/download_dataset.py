"""Download LibriSpeech train-clean-100 via HuggingFace datasets.

Why LibriSpeech:
  - 100 hours of clean 16 kHz English speech — perfect ASR target rate for Gemma 4.
  - Verbatim transcripts (standard ASR benchmark).
  - Diverse speakers (many hundreds of readers).
  - Open license (CC BY 4.0).

Why train-clean-100 specifically and not the full 960 h:
  - We will augment each clip many ways (varied noise, SNR, radio FX).
  - 100 h raw * N augmentations >> enough for LoRA on ~7M params.
  - Downloads in a reasonable time (~6 GB, ~15 min on decent bandwidth).
"""
from __future__ import annotations

from datasets import load_dataset

SPLIT = "train.clean.100"
OUT = "librispeech_clean_100"

def main():
    print(f"Loading LibriSpeech {SPLIT} (~6 GB, streamed to cache)...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split=SPLIT)
    print(f"Loaded: {len(ds)} examples")
    print(f"First example keys: {list(ds[0].keys())}")
    ex = ds[0]
    print(f"  text: {ex['text'][:100]!r}")
    print(f"  audio sampling_rate: {ex['audio']['sampling_rate']}")
    print(f"  audio array shape:   {ex['audio']['array'].shape}")
    print(f"  audio duration (s):  {len(ex['audio']['array']) / ex['audio']['sampling_rate']:.2f}")

    # Stats
    total_dur = sum(len(ds[i]["audio"]["array"]) for i in range(min(100, len(ds)))) / 16000
    print(f"\nFirst 100 clips total duration: {total_dur:.1f} s")

if __name__ == "__main__":
    main()
