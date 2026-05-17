#!/usr/bin/env python3
"""AirGrep ASR benchmark — measure Word Error Rate across radio conditions.

Runs every clip in samples/ through the ASR model and compares
against ground-truth transcripts from the manifest.  Outputs a table of
WER by radio condition and an overall score.

Usage:
    python benchmark.py                          # fine-tuned model (default)
    python benchmark.py --model-path /path/to   # specific checkpoint
    python benchmark.py --base google/gemma-4-e4b  # compare with base model

Results are printed to stdout AND saved to benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# WER computation (no external dependency needed)
# ---------------------------------------------------------------------------

def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance at the word level."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
    return dp[m]


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Compute WER between reference and hypothesis strings.

    Returns (wer, errors, ref_word_count).
    Both strings are lowered and stripped before comparison.
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    if not ref_words:
        return (0.0 if not hyp_words else 1.0), len(hyp_words), 0
    errors = _edit_distance(ref_words, hyp_words)
    return errors / len(ref_words), errors, len(ref_words)


# ---------------------------------------------------------------------------
# ASR inference (reuses llm.py internals)
# ---------------------------------------------------------------------------

def transcribe_clip(model, processor, wav_path: str) -> str:
    """Run ASR on a single clip. Returns the transcript string."""
    from llm import _load_audio_16k, build_asr_prompt
    import torch

    audio = _load_audio_16k(wav_path)

    # Skip silence (same energy gate as the main pipeline)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.005:
        return "[no signal]"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": build_asr_prompt()},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {
        k: (v.to(model.device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    transcript = processor.tokenizer.decode(
        new_ids, skip_special_tokens=True
    ).strip()

    # Post-ASR safety net: detect echoed prompt
    prompt = build_asr_prompt()
    if transcript.strip() == prompt.strip():
        return "[no signal]"

    return transcript


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AirGrep ASR benchmark — WER by radio condition.",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="wames123/airgrep-asr-gemma-4-e4b",
        help="HF repo id or local path to model checkpoint",
    )
    parser.add_argument(
        "--samples-dir", type=str,
        default=str(Path(__file__).parent / "samples"),
        help="Directory containing sample WAVs and manifest.json",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--cpu", action="store_true", dest="force_cpu",
        help="Force CPU inference (no GPU). Slow — use only for testing.",
    )
    parser.add_argument(
        "--max-gpu-memory", type=str, default=None, dest="max_gpu_memory",
        help="Cap GPU VRAM usage (e.g. '7GiB'). Spills to CPU RAM for small GPUs.",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    manifest_path = samples_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter to actual WAV entries (skip train_preview etc.)
    entries = [
        e for e in manifest
        if (samples_dir / e["file"]).exists()
    ]
    print(f"Benchmark: {len(entries)} clips from {manifest_path.parent.name}/")
    print(f"Model:     {args.model_path}")
    print()

    # Load model once
    from llm import _get_model
    print("Loading model...")
    t0 = time.perf_counter()
    model, processor = _get_model(
        args.model_path,
        force_cpu=args.force_cpu,
        max_gpu_memory=args.max_gpu_memory,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

    # Run benchmark
    results = []
    by_condition: dict[str, list[float]] = defaultdict(list)
    total_errors = 0
    total_words = 0

    print(f"{'File':<52} {'Condition':<16} {'WER':>6}  Hypothesis")
    print("-" * 120)

    for entry in entries:
        wav_path = str(samples_dir / entry["file"])
        reference = entry["transcript"]
        condition = entry.get("preset") or entry.get("kind", "unknown")

        t1 = time.perf_counter()
        hypothesis = transcribe_clip(model, processor, wav_path)
        elapsed = time.perf_counter() - t1

        wer, errors, ref_words = word_error_rate(reference, hypothesis)

        results.append({
            "file": entry["file"],
            "condition": condition,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(wer, 4),
            "errors": errors,
            "ref_words": ref_words,
            "time_s": round(elapsed, 2),
        })

        by_condition[condition].append(wer)
        total_errors += errors
        total_words += ref_words

        # Truncate hypothesis for display
        hyp_display = hypothesis[:60] + "..." if len(hypothesis) > 60 else hypothesis
        print(f"{entry['file']:<52} {condition:<16} {wer:>5.1%}  {hyp_display}")

    # Summary
    print("\n" + "=" * 80)
    print(f"{'CONDITION':<20} {'CLIPS':>6} {'AVG WER':>8} {'MIN':>6} {'MAX':>6}")
    print("-" * 80)

    summary = {}
    for condition in sorted(by_condition.keys()):
        wers = by_condition[condition]
        avg = sum(wers) / len(wers)
        summary[condition] = {
            "clips": len(wers),
            "avg_wer": round(avg, 4),
            "min_wer": round(min(wers), 4),
            "max_wer": round(max(wers), 4),
        }
        print(f"{condition:<20} {len(wers):>6} {avg:>7.1%} {min(wers):>5.1%} {max(wers):>5.1%}")

    overall_wer = total_errors / total_words if total_words > 0 else 0.0
    print("-" * 80)
    print(f"{'OVERALL':<20} {len(results):>6} {overall_wer:>7.1%}")
    print("=" * 80)

    # Save results
    output = {
        "model_path": args.model_path,
        "total_clips": len(results),
        "overall_wer": round(overall_wer, 4),
        "by_condition": summary,
        "clips": results,
    }
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
