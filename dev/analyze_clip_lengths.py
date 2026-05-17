"""Sample clip-length distribution from LibriSpeech train-clean-100.

Also check Gemma 4's max audio context from the processor/config.
"""
from __future__ import annotations

import io
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio


def librispeech_length_stats(n: int = 500):
    print(f"Streaming {n} LibriSpeech train-clean-100 clips to measure durations...")
    ds = load_dataset(
        "openslr/librispeech_asr", "clean",
        split="train.100", streaming=True,
    ).cast_column("audio", Audio(decode=False))

    durs = []
    it = iter(ds)
    for _ in range(n):
        ex = next(it)
        ai = ex["audio"]
        if ai.get("bytes"):
            info = sf.info(io.BytesIO(ai["bytes"]))
        else:
            info = sf.info(ai["path"])
        durs.append(info.frames / info.samplerate)

    durs = np.array(durs)
    print(f"\nClip duration stats over {len(durs)} samples (seconds):")
    print(f"  min:   {durs.min():.2f}")
    print(f"  p10:   {np.percentile(durs, 10):.2f}")
    print(f"  p25:   {np.percentile(durs, 25):.2f}")
    print(f"  median:{np.percentile(durs, 50):.2f}")
    print(f"  mean:  {durs.mean():.2f}")
    print(f"  p75:   {np.percentile(durs, 75):.2f}")
    print(f"  p90:   {np.percentile(durs, 90):.2f}")
    print(f"  p99:   {np.percentile(durs, 99):.2f}")
    print(f"  max:   {durs.max():.2f}")

    # Distribution buckets
    print("\nBuckets:")
    buckets = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 9999)]
    for lo, hi in buckets:
        n_in = int(((durs >= lo) & (durs < hi)).sum())
        print(f"  {lo:3d}-{hi:4d}s:  {n_in:4d}  ({100*n_in/len(durs):5.1f}%)")


def gemma4_audio_limits():
    print("\n" + "=" * 60)
    print("Gemma 4 audio processor / config limits")
    print("=" * 60)
    from transformers import AutoProcessor, AutoConfig
    proc = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
    cfg = AutoConfig.from_pretrained("google/gemma-4-E4B-it")

    fe = getattr(proc, "feature_extractor", None)
    if fe is not None:
        for attr in ["sampling_rate", "feature_size", "hop_length",
                     "n_fft", "chunk_length", "nb_max_frames",
                     "max_audio_length", "max_audio_seconds"]:
            if hasattr(fe, attr):
                print(f"  feature_extractor.{attr}: {getattr(fe, attr)}")

    ac = getattr(cfg, "audio_config", None)
    if ac is not None:
        for attr in [
            "attention_chunk_size", "attention_context_left",
            "attention_context_right", "num_hidden_layers", "hidden_size",
            "input_feat_size", "subsampling_conv_channels", "output_proj_dims",
        ]:
            if hasattr(ac, attr):
                print(f"  audio_config.{attr}: {getattr(ac, attr)}")

    # Run the FE on a known-length dummy clip to see what N_tokens comes out
    print("\nToken-rate probe:")
    for secs in [2, 5, 10, 15, 20, 29, 30, 45, 60]:
        wav = np.zeros(16000 * secs, dtype=np.float32)
        try:
            out = proc(audio=wav, text="x", sampling_rate=16000, return_tensors="pt")
        except Exception as e:
            print(f"  {secs:2d}s: ERROR {e}")
            continue
        feat = out.get("input_features", None)
        if feat is not None:
            print(f"  {secs:2d}s audio -> feature shape {tuple(feat.shape)}  "
                  f"(~{feat.shape[1]/secs:.1f} mel-frames/sec)")


if __name__ == "__main__":
    librispeech_length_stats(500)
    gemma4_audio_limits()
