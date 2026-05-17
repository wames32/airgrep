"""Inspect Gemma 3n E4B architecture to verify audio encoder module names.

Goal: enumerate the exact Linear module names under model.audio_tower and
model.embed_audio so we can target them with PEFT LoRA.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E4B-it"


def main() -> None:
    print(f"Loading {MODEL_ID} (bf16, cpu meta for introspection only)...")
    # Load in bf16 on CPU just for inspection — we only need module structure.
    # We'll reload properly for actual training.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    # Find top-level
    top_names = [n for n, _ in model.named_children()]
    print(f"\nTop-level children of model: {top_names}")

    # Inner model
    if hasattr(model, "model"):
        inner_names = [n for n, _ in model.model.named_children()]
        print(f"Children of model.model: {inner_names}")

    # Audio tower
    if hasattr(model.model, "audio_tower"):
        at = model.model.audio_tower
        print(f"\naudio_tower class: {type(at).__name__}")
        print(f"audio_tower children: {[n for n, _ in at.named_children()]}")

        # Count linears and their suffixes
        linears = [(n, tuple(m.weight.shape)) for n, m in at.named_modules() if isinstance(m, torch.nn.Linear)]
        print(f"\naudio_tower contains {len(linears)} nn.Linear modules.")
        # Print unique suffixes
        suffixes = sorted({n.split(".", 2)[-1] if n.count(".") >= 2 else n for n, _ in linears})
        print("Unique linear-module suffixes (after stripping conformer.{i}.):")
        stripped = sorted({
            ".".join(p for p in n.split(".") if not p.isdigit())
            for n, _ in linears
        })
        for s in stripped:
            print(f"  {s}")

        # Number of conformer layers
        if hasattr(at, "conformer"):
            n_layers = len(at.conformer)
            print(f"\naudio_tower.conformer has {n_layers} layers")

    # embed_audio
    if hasattr(model.model, "embed_audio"):
        ea = model.model.embed_audio
        print(f"\nembed_audio class: {type(ea).__name__}")
        ea_linears = [n for n, m in ea.named_modules() if isinstance(m, torch.nn.Linear)]
        print(f"embed_audio linears: {ea_linears}")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    audio_params = sum(p.numel() for p in model.model.audio_tower.parameters()) if hasattr(model.model, "audio_tower") else 0
    print(f"\nTotal model params:  {total_params/1e9:.2f} B")
    print(f"audio_tower params:  {audio_params/1e6:.1f} M ({100*audio_params/total_params:.1f}%)")

    # Expected input
    print(f"\nmodel.config.audio_config: {model.config.audio_config}")


if __name__ == "__main__":
    main()
