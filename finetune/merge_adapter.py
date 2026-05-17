"""Merge LoRA adapter into base Gemma 4 bf16 checkpoint.

Output: a normal HF checkpoint at runs/merged_bf16/ that's binary-compatible
with the original Gemma 4 layout — just with our LoRA deltas folded into the
audio_tower + embed_audio Linear weights.

After saving, runs eval_adapter-style inference on the merged model to verify
that the merge preserved the fine-tune (should still output KJ7 ZBB).
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E4B-it"
ADAPTER = Path(__file__).parent / "runs" / "adapter" / "step_1000"
MERGED = Path(__file__).parent / "runs" / "merged_bf16"
WAV = Path(__file__).resolve().parents[1] / "debug_audio" / "cycle_20260412_223734_144.35MHz.wav"


def load_audio_16k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    assert sr == 16000, f"unexpected sr {sr}"
    return x[: 29 * 16000]


def main():
    print(f"Loading base {MODEL_ID} (bf16)...")
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
    )
    print(f"  base loaded in {time.perf_counter()-t0:.1f}s")

    print(f"Loading adapter {ADAPTER}...")
    peft_model = PeftModel.from_pretrained(base, str(ADAPTER))

    print("Merging LoRA into base weights (this takes a few min)...")
    t0 = time.perf_counter()
    merged = peft_model.merge_and_unload()
    print(f"  merged in {time.perf_counter()-t0:.1f}s")

    print(f"Saving merged checkpoint to {MERGED} ...")
    MERGED.mkdir(parents=True, exist_ok=True)
    # Bypass save_pretrained: safetensors 0.7.0 hits a Windows ctypes int
    # overflow on individual tensors >2 GB (per_layer_token_embd is ~5 GB).
    # Manually shard with torch.save (.bin) which has no such limit.
    # convert_hf_to_gguf.py reads pytorch_model-*.bin sharded layouts.
    import json
    state_dict = merged.state_dict()
    shard_max_bytes = 1_900 * 1024 * 1024
    shards: list[dict] = [{}]
    cur_size = 0
    weight_map: dict[str, str] = {}
    for k, v in state_dict.items():
        sz = v.numel() * v.element_size()
        if cur_size + sz > shard_max_bytes and shards[-1]:
            shards.append({})
            cur_size = 0
        shards[-1][k] = v.detach().cpu()
        cur_size += sz
    n = len(shards)
    print(f"  writing {n} shards...")
    for i, shard in enumerate(shards, 1):
        fname = f"pytorch_model-{i:05d}-of-{n:05d}.bin"
        for k in shard:
            weight_map[k] = fname
        torch.save(shard, MERGED / fname)
        gb = sum(t.numel() * t.element_size() for t in shard.values()) / 1e9
        print(f"    [{i}/{n}] {fname}  ({len(shard)} tensors, {gb:.2f} GB)")
    total = sum(t.numel() * t.element_size() for sh in shards for t in sh.values())
    (MERGED / "pytorch_model.bin.index.json").write_text(json.dumps({
        "metadata": {"total_size": total},
        "weight_map": weight_map,
    }, indent=2))
    merged.config.save_pretrained(MERGED)
    if hasattr(merged, "generation_config") and merged.generation_config is not None:
        merged.generation_config.save_pretrained(MERGED)
    proc.save_pretrained(MERGED)
    print(f"  saved.")

    # Sanity: run inference on merged model
    print("\nSanity check — generating on debug clip with MERGED model...")
    audio = load_audio_16k(WAV)
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
    inputs = {k: v.to(merged.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    merged.eval()
    with torch.inference_mode():
        out = merged.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=1.0)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"  merged output: {text!r}")

    expected_substr = "KJ7"
    if expected_substr in text and ("ZBB" in text or "Z BB" in text):
        print(f"  ✓ MERGE PRESERVED FINE-TUNE (saw '{expected_substr}' + Z* callsign)")
    else:
        print(f"  ✗ WARNING: merged output looks like base model — check for precision drift")

    del merged, peft_model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
