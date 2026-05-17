"""Fine-tune Gemma 4 E4B audio encoder (LoRA) on radio-augmented speech.

Architecture decision log:
  - Full bf16 model, no 4-bit. bnb Linear4bit is incompatible with the
    audio encoder's torch.finfo(weight.dtype) clipping code.
  - LoRA r=16 alpha=32 on every Linear in audio_tower + embed_audio.
  - Language model frozen; only 7.0M params train.
  - Label masking: loss only on the assistant's transcript tokens.
    The user prompt (instructions) and audio soft-tokens get label = -100
    so they don't contribute to cross-entropy.
  - Gradient checkpointing enabled to fit the 30s training clips.
  - Optimizer: AdamW 8-bit (bitsandbytes) for memory efficiency on LoRA.
  - LR: 2e-4 (standard LoRA); cosine schedule; 100 warmup steps.
  - Batch size 1 + grad-accum 8 => effective batch 8.
  - Periodic eval on held-out speakers; save adapter every N steps.

Run:
  python train.py [--steps 2000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from train_dataset import build_dataset, Example, SR

MODEL_ID = "google/gemma-4-E4B-it"
OUT_DIR = Path(__file__).parent / "runs"
ADAPTER_DIR = OUT_DIR / "adapter"

# ── LoRA targets: every Linear in audio_tower + embed_audio ─────────
AUDIO_REGEXES = [
    r".*audio_tower\.layers\.\d+\.self_attn\.q_proj\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.k_proj\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.v_proj\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.post\.linear$",
    r".*audio_tower\.layers\.\d+\.self_attn\.relative_k_proj$",
    r".*audio_tower\.layers\.\d+\.feed_forward1\.ffw_layer_1\.linear$",
    r".*audio_tower\.layers\.\d+\.feed_forward1\.ffw_layer_2\.linear$",
    r".*audio_tower\.layers\.\d+\.feed_forward2\.ffw_layer_1\.linear$",
    r".*audio_tower\.layers\.\d+\.feed_forward2\.ffw_layer_2\.linear$",
    r".*audio_tower\.layers\.\d+\.lconv1d\.linear_start\.linear$",
    r".*audio_tower\.layers\.\d+\.lconv1d\.linear_end\.linear$",
    r".*audio_tower\.subsample_conv_projection\.input_proj_linear$",
    r".*audio_tower\.output_proj$",
    r".*embed_audio\.embedding_projection$",
]


def build_model_and_processor():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
    )
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable()

    patterns = [re.compile(p) for p in AUDIO_REGEXES]
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and any(p.match(n) for p in patterns)]
    print(f"LoRA targets: {len(names)} audio linears.")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", target_modules=names, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_train/1e6:.2f}M params")
    return model, processor


# ── Collator: audio + chat-formatted text, with label masking ───────


def build_training_inputs(processor, ex: Example):
    """Build one training example.

    Returns a dict with keys:
      input_features, input_features_mask, input_ids, attention_mask, labels
    The `labels` tensor has -100 on prompt/audio tokens, real token IDs
    on transcript+end-of-turn tokens.
    """
    # Phase 1: encode prompt (user turn, no assistant yet) to discover how
    # many prompt tokens exist. We'll mask those in labels.
    prompt_msgs = [{
        "role": "user", "content": [
            {"type": "audio", "audio": ex.audio},
            {"type": "text", "text": "Transcribe this audio."},
        ]
    }]
    prompt_out = processor.apply_chat_template(
        prompt_msgs, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )
    prompt_len = prompt_out["input_ids"].shape[-1]

    # Phase 2: encode full conversation (user + assistant target).
    full_msgs = prompt_msgs + [{
        "role": "assistant",
        "content": [{"type": "text", "text": ex.text}],
    }]
    full_out = processor.apply_chat_template(
        full_msgs, add_generation_prompt=False,
        tokenize=True, return_tensors="pt", return_dict=True,
    )

    input_ids = full_out["input_ids"]
    labels = input_ids.clone()
    # Mask prompt (audio + instruction + role markers) — don't train on them.
    labels[:, :prompt_len] = -100
    full_out["labels"] = labels
    return full_out


def move_to(d, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in d.items()}


# ── Train loop ───────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    steps: int = 2000
    grad_accum: int = 8
    lr: float = 2e-4
    warmup: int = 100
    eval_every: int = 200
    save_every: int = 500
    log_every: int = 10
    eval_samples: int = 32


def evaluate(model, processor, val_ds, cfg: TrainConfig) -> float:
    model.eval()
    losses = []
    rng = np.random.default_rng(0)
    indices = rng.choice(len(val_ds), size=min(cfg.eval_samples, len(val_ds)), replace=False)
    with torch.inference_mode():
        for idx in indices:
            ex = val_ds[int(idx)]
            batch = build_training_inputs(processor, ex)
            batch = move_to(batch, model.device)
            out = model(**batch)
            losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="Quick 50-step sanity run")
    ap.add_argument("--resume", type=str, default=None, help="Resume from adapter dir")
    args = ap.parse_args()

    cfg = TrainConfig(steps=50 if args.smoke else args.steps)
    if args.smoke:
        cfg.eval_every = 25
        cfg.save_every = 50
        cfg.log_every = 5
        cfg.eval_samples = 8
        cfg.warmup = 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print("Building model + processor...")
    model, processor = build_model_and_processor()
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_adapter(args.resume, adapter_name="default")

    print("Building datasets...")
    train_ds, val_ds = build_dataset()

    # Optimizer (8-bit AdamW for memory efficiency on LoRA params).
    import bitsandbytes as bnb
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = bnb.optim.AdamW8bit(trainable, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, cfg.warmup, cfg.steps)

    model.train()
    log_path = OUT_DIR / "train_log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step = 0
    micro_losses = []
    order = np.random.default_rng(0).permutation(len(train_ds))
    order_pos = 0

    while step < cfg.steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.grad_accum):
            # Wrap-around through the dataset
            if order_pos >= len(order):
                order = np.random.default_rng(step).permutation(len(train_ds))
                order_pos = 0
            ex_idx = int(order[order_pos]); order_pos += 1

            ex = train_ds[ex_idx]
            batch = build_training_inputs(processor, ex)
            batch = move_to(batch, model.device)
            out = model(**batch)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        sched.step()
        step += 1
        micro_losses.append(accum_loss)

        if step % cfg.log_every == 0:
            lr_now = sched.get_last_lr()[0]
            peak = torch.cuda.max_memory_allocated() / 1e9
            mean_loss = float(np.mean(micro_losses[-cfg.log_every:]))
            elapsed = time.perf_counter() - t0
            print(f"  step {step:4d}/{cfg.steps}  loss {mean_loss:.4f}  "
                  f"lr {lr_now:.2e}  vram {peak:.1f}GB  {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step, "loss": mean_loss, "lr": lr_now,
                "peak_vram_gb": round(peak, 2), "elapsed_s": round(elapsed, 1),
            }) + "\n")
            log_f.flush()

        if step % cfg.eval_every == 0:
            val_loss = evaluate(model, processor, val_ds, cfg)
            print(f"  >>> step {step}: val loss {val_loss:.4f}")
            log_f.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
            log_f.flush()

        if step % cfg.save_every == 0 or step == cfg.steps:
            save_to = ADAPTER_DIR / f"step_{step}"
            model.save_pretrained(save_to)
            print(f"  >>> saved adapter to {save_to}")

    log_f.close()
    total = time.perf_counter() - t0
    print(f"\nDone. {cfg.steps} steps in {total:.0f}s "
          f"({total/cfg.steps:.1f}s/step).")


if __name__ == "__main__":
    main()
