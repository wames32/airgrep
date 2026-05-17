"""Measure pretrained Gemma 4 E4B performance on the debug radio clip.

This is the baseline to compare the fine-tuned adapter against.
Saves output JSON for later comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E4B-it"
WAV = Path(__file__).resolve().parents[1] / "debug_audio" / "cycle_20260412_223734_144.35MHz.wav"
OUT = Path(__file__).parent / "baseline_output.json"
TRIALS = 3


def load_audio_16k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    # Truncate to 29s (Gemma 4 audio cap)
    x = x[: 29 * 16000]
    return x


def main():
    print(f"Loading {MODEL_ID} (bf16)...")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
    )
    model.eval()

    audio = load_audio_16k(WAV)
    print(f"Audio: {WAV.name}  {len(audio)/16000:.1f}s")

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": (
                "You are transcribing audio from a radio receiver. "
                "The audio may contain amateur (ham) radio transmissions with callsigns. "
                "Transcribe exactly what you hear. Output only the transcription."
            )},
        ],
    }]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    results = []
    for t in range(TRIALS):
        torch.manual_seed(t)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,     # deterministic greedy
                temperature=1.0,
            )
        dt = time.perf_counter() - t0
        # Decode only the newly generated tokens
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"  [{t+1}/{TRIALS}] ({dt:.1f}s) {text!r}")
        results.append({"trial": t + 1, "elapsed_s": round(dt, 2), "text": text})

    record = {
        "model": MODEL_ID,
        "audio_file": str(WAV.name),
        "expected": "This is KJ7ZBB looking for a signal report",
        "results": results,
    }
    OUT.write_text(json.dumps(record, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
