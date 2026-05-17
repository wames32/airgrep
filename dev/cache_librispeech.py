"""One-shot: extract LibriSpeech train-clean-100 to indexed FLAC + manifest.

After this runs, we have:
  librispeech_cache/
    flacs/{speaker}-{chapter}-{utterance}.flac     (raw bytes from the dataset)
    manifest.jsonl                                  (one JSON per line)

Manifest rows look like:
  {"id": "103-1240-0000", "speaker": "103", "chapter": "1240",
   "utterance": "0000", "path": "flacs/103-1240-0000.flac",
   "text": "CHAPTER ONE ...", "duration": 14.23}

This avoids torchcodec / FFmpeg (we keep raw FLAC bytes and decode at train
time with soundfile, which works fine on Windows).
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import soundfile as sf
from datasets import load_dataset, Audio

CACHE_DIR = Path(__file__).parent / "librispeech_cache"
FLAC_DIR = CACHE_DIR / "flacs"
MANIFEST = CACHE_DIR / "manifest.jsonl"

SPLIT = "train.100"  # ~100h, ~28.5k utterances


def main():
    FLAC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming LibriSpeech {SPLIT} (bytes mode, torchcodec bypassed)...")
    ds = load_dataset(
        "openslr/librispeech_asr", "clean",
        split=SPLIT, streaming=True,
    ).cast_column("audio", Audio(decode=False))

    n_done = 0
    n_skipped = 0
    total_dur = 0.0
    t0 = time.perf_counter()

    with open(MANIFEST, "w", encoding="utf-8") as mf:
        for ex in ds:
            ai = ex["audio"]
            raw = ai.get("bytes")
            if raw is None:
                # fallback: dataset gave us a path
                raw = Path(ai["path"]).read_bytes()

            clip_id = ex["id"]
            out_path = FLAC_DIR / f"{clip_id}.flac"
            if out_path.exists():
                # Idempotent — resume from prior run.
                info = sf.info(str(out_path))
                dur = info.frames / info.samplerate
                n_skipped += 1
            else:
                out_path.write_bytes(raw)
                try:
                    info = sf.info(io.BytesIO(raw))
                    dur = info.frames / info.samplerate
                except Exception as e:
                    print(f"  WARN: bad audio for {clip_id}: {e}")
                    out_path.unlink(missing_ok=True)
                    continue

            speaker, chapter, utt = clip_id.split("-")
            row = {
                "id": clip_id,
                "speaker": speaker,
                "chapter": chapter,
                "utterance": utt,
                "path": f"flacs/{clip_id}.flac",
                "text": ex["text"],
                "duration": round(dur, 3),
            }
            mf.write(json.dumps(row) + "\n")
            total_dur += dur
            n_done += 1

            if n_done % 500 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {n_done:5d} utterances   "
                      f"total audio: {total_dur/3600:.2f}h   "
                      f"elapsed: {elapsed:.0f}s   "
                      f"skipped (already cached): {n_skipped}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {n_done} utterances cached "
          f"({total_dur/3600:.2f}h audio, {elapsed:.0f}s elapsed, "
          f"{n_skipped} resumed from previous run).")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
