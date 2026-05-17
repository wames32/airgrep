"""Feasibility test: LoRA on Gemma 4 E4B audio encoder.

Objectives:
 1. Load the model in 4-bit (QLoRA).
 2. Attach LoRA adapters to the audio_tower + embed_audio linears ONLY.
    Everything else stays frozen.
 3. Run a real forward pass on (audio, text_target).
 4. Run backward pass, verify gradients flow only through LoRA params,
    and that the audio-tower LoRA params specifically receive gradient.
 5. Report VRAM usage and trainable param count.

If all assertions pass, we have confirmed that audio-encoder LoRA is
trainable on this setup and can proceed to data prep.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/gemma-4-E4B-it"
WAV_PATH = Path(__file__).resolve().parents[1] / "debug_audio" / "cycle_20260412_223734_144.35MHz.wav"
TARGET_TEXT = "This is KJ7ZBB looking for a signal report"

# Target modules — matched as regex suffixes against module names.
# Covers every Linear in the audio_tower and the audio projector.
AUDIO_TARGETS = [
    # Per-layer attention
    r".*audio_tower\.layers\.\d+\.self_attn\.q_proj\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.k_proj\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.v_proj\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.post\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.relative_k_proj$",
    # Per-layer feed-forwards
    r".*audio_tower\.layers\.\d+\.feed_forward1\.ffw_layer_1\.linear$",
    r".*audio_tower\.layers\.\d+\.feed_forward1\.ffw_layer_2\.linear$",
    r".*audio_tower\.layers\.\d+\.feed_forward2\.ffw_layer_1\.linear$",
    r".*audio_tower\.layers\.\d+\.feed_forward2\.ffw_layer_2\.linear$",
    # Per-layer light conv
    r".*audio_tower\.layers\.\d+\.lconv1d\.linear_start\.linear$",
    r".*audio_tower\.layers\.\d+\.lconv1d\.linear_end\.linear$",
    # Tower-level projections
    r".*audio_tower\.subsample_conv_projection\.input_proj_linear$",
    r".*audio_tower\.output_proj$",
    # Audio -> text projector
    r".*embed_audio\.embedding_projection$",
]


def main() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Note: Gemma 4's audio encoder uses torch.finfo(weight.dtype) in its
    # clipped-linear forward pass, which is incompatible with bitsandbytes
    # 4-bit quantization (uint8 weights). llm_int8_skip_modules is silently
    # ignored by bnb for 4-bit. Simplest correct path: bf16 full model.
    # 7.94B * 2 bytes ≈ 16 GB; fits 25 GB VRAM with LoRA optimizer state.
    print(f"Loading {MODEL_ID} in bf16 (full precision)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    # Freeze everything; PEFT will unfreeze LoRA params.
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable()

    # Verify our regex patterns actually match existing modules before building LoRA.
    all_linear_names = [
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    ]
    patterns = [re.compile(p) for p in AUDIO_TARGETS]
    matched = [n for n in all_linear_names if any(p.match(n) for p in patterns)]
    print(f"\n{len(matched)} audio linear modules matched for LoRA.")
    # sanity: should be 12 layers * 11 per-layer + 3 tower-level + 1 projector = 136
    # (but relative_k_proj is one of the 11 per-layer)
    if len(matched) < 120:
        # Print a few for debugging
        audio_linears = [n for n in all_linear_names if "audio" in n]
        print("First 20 audio linears in model (for debug):")
        for n in audio_linears[:20]:
            print(f"  {n}")
        raise RuntimeError("Too few matches — regex patterns wrong.")

    # PEFT expects target_modules as list of strings; with regex via re: prefix
    # or as suffix list. Simpler: pass matched module names directly.
    # But PEFT's target_modules matches on SUFFIX by default, which is what we want.
    # We'll pass the minimal suffix patterns.
    suffix_targets = [
        "q_proj.linear", "k_proj.linear", "v_proj.linear", "post.linear",
        "relative_k_proj",
        "ffw_layer_1.linear", "ffw_layer_2.linear",
        "linear_start.linear", "linear_end.linear",
        "input_proj_linear",
        "output_proj",
        "embedding_projection",
    ]
    # ^ These suffixes would match language layers too (e.g. q_proj). We need
    # to restrict. PEFT LoraConfig supports `target_modules` as a list of
    # strings where each string matches the FULL qualified name via regex when
    # prefixed with "re:"... actually PEFT uses suffix match OR exact match.
    # To be safe, use the explicit matched names we computed above.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=matched,  # exact names
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Report trainable params
    trainable, total = 0, 0
    audio_trainable = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if "audio_tower" in n or "embed_audio" in n:
                audio_trainable += p.numel()
    print(
        f"\nTrainable params: {trainable/1e6:.2f} M / {total/1e9:.2f} B "
        f"({100*trainable/total:.3f}%)"
    )
    print(f"Of those, in audio stack: {audio_trainable/1e6:.2f} M "
          f"({100*audio_trainable/trainable:.1f}%)")
    assert audio_trainable / trainable > 0.95, (
        "Audio stack should dominate trainable params. "
        "Something is targeting the wrong modules."
    )

    # ── Real forward pass with audio + text target ────────────────────
    print(f"\nBuilding inputs from {WAV_PATH.name} + target text...")
    import soundfile as sf
    wav, sr = sf.read(str(WAV_PATH))
    # Gemma 4 audio expects 16 kHz mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    # Truncate to 30s to keep test cheap
    max_samples = 30 * sr
    if len(wav) > max_samples:
        wav = wav[:max_samples]

    # Build a chat-style message with audio + target response
    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio": wav},
            {"type": "text", "text": "Transcribe this audio."},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": TARGET_TEXT}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    # Move to GPU; audio tensors stay float32 per processor convention
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    # Labels = input_ids (standard SFT; we compute LM loss over the whole seq,
    # good enough for a smoke test)
    labels = inputs["input_ids"].clone()
    inputs["labels"] = labels

    print("Forward pass...")
    torch.cuda.synchronize()
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"  loss: {loss.item():.4f}")

    print("Backward pass...")
    loss.backward()
    torch.cuda.synchronize()

    # Check gradients on audio LoRA params
    grad_audio = 0
    grad_audio_count = 0
    grad_other = 0
    for n, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            g = p.grad.detach().abs().sum().item()
            if "audio_tower" in n or "embed_audio" in n:
                grad_audio += g
                grad_audio_count += 1
            else:
                grad_other += g
    print(f"\nGradient flow check:")
    print(f"  audio LoRA params with nonzero grad count: {grad_audio_count}")
    print(f"  sum |grad| on audio stack: {grad_audio:.4f}")
    print(f"  sum |grad| elsewhere:     {grad_other:.4f}")
    assert grad_audio > 0, "No gradient reached audio-tower LoRA!"

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak VRAM: {peak:.2f} GB")
    print("\n✓ Audio-encoder LoRA forward+backward works.")


if __name__ == "__main__":
    main()
