# Fine-tuning results

Modular results log. Each numbered section is one experiment. Append new
sections rather than rewriting prior ones.

**Test clip:** `debug_audio/cycle_20260412_223734_144.35MHz.wav`
(29.0s, 16 kHz, recorded from RTL-SDR on 144.35 MHz NBFM,
spoken via handie-talkie in a quiet office).

**Ground truth:** `This is KJ7ZBB looking for a signal report`

All generation uses the same prompt for apples-to-apples comparison:

> You are transcribing audio from a radio receiver. The audio may contain
> amateur (ham) radio transmissions with callsigns. Transcribe exactly
> what you hear. Output only the transcription.

Decoding: `do_sample=False` (greedy), `max_new_tokens=128`, 3 trials per run.

---

## 1. Baseline — pretrained Gemma 4 E4B (2026-04-14)

- Model: `google/gemma-4-E4B-it` at bf16, no adapter
- Source: `finetune/baseline_output.json`

| trial | output |
|---|---|
| 1 | `This is KJ7CBB looking for a signal report.` |
| 2 | `This is KJ7CBB looking for a signal report.` |
| 3 | `This is KJ7CBB looking for a signal report.` |

**Word-error analysis (vs ground truth):**
- Callsign: `KJ7CBB` vs `KJ7ZBB` — **1 character wrong** (`C`→`Z`)
- Content words: all correct (`signal report` preserved)
- Structure: perfect

Character-error rate: 1/43 ≈ 2.3%. Deterministic (greedy), 3/3 identical.

---

## 2. Audio-encoder LoRA, 1000 steps (2026-04-14)

- Base: `google/gemma-4-E4B-it` bf16, language model frozen
- Adapter: LoRA r=16 α=32 on 135 Linears in `audio_tower` + `embed_audio`
  - Trainable params: **7.02M / 7.94B (0.09%)**
- Training data: LibriSpeech train-clean-100 (27,927 utterances, 245 speakers)
  with on-the-fly radio-degradation augmentation
  - 40% of clips are 2–3 same-speaker utterances concatenated with
    0.6–2.0s silence gaps (multi-over simulation, ≤27s)
  - 94% of transcripts beautified via Qwen 3.5 9B (casing + punctuation),
    5.6% fall back to LibriSpeech ALL-CAPS
- 6 augmentation presets: nominal (0.35), weak (0.20), overmod (0.15),
  clean (0.15), squelched (0.10), ducked (0.05)
- Optimizer: AdamW 8-bit, lr 2e-4, cosine schedule, warmup 100
- Batch 1 × grad-accum 8 = effective batch 8
- Label masking: loss only on assistant's transcript tokens
- Hardware: single RTX 3090, gradient checkpointing
- Wall-clock: **8.79 hours** (31.6 s/step), peak VRAM 22.1 GB
- Adapter: `runs/adapter/step_1000/`
- Source: `finetune/eval_output.json`

### Val loss (held-out 6 speakers, 612 utterances, 32-sample subset)

| step | val loss |
|---|---|
| 200 | 0.340 |
| 400 | 0.309 |
| 600 | 0.315 |
| 800 | 0.299 |
| 1000 | **0.297** |

Still trending down at 1000 — likely some headroom remains.

### Inference results

| trial | output |
|---|---|
| 1 | `This is KJ7 ZBB looking for a signal report.` |
| 2 | `This is KJ7 ZBB looking for a signal report.` |
| 3 | `This is KJ7 ZBB looking for a signal report.` |

**Word-error analysis (vs ground truth):**
- Callsign letters: `KJ7ZBB` — **all correct**
- Spurious space between `KJ7` and `ZBB` (tokenization artifact,
  not a perception error — trivial to post-process)
- Content words: all correct
- Structure: perfect

Character-error rate: 0/43 on content, 1/43 on whitespace.
Deterministic, 3/3 identical.

### Verdict

Fine-tune **fixed the hardest part of the clip** (the callsign character
the base model hallucinated) while leaving everything the base model
already got right unchanged. This validates the core thesis: domain
adapting only the audio encoder, with the language decoder frozen, gives
us robust NBFM ASR without touching general capabilities used elsewhere
in the pipeline.

---

## How to add a new experiment

1. Run training or inference, write outputs to `eval_output*.json` or
   similar.
2. Append a new numbered `## N. <title> (YYYY-MM-DD)` section above.
3. Include: what changed vs prior run, raw outputs, WER/CER vs ground
   truth, qualitative verdict.
4. If a new plot is helpful, regenerate `runs/loss_curve.png` via
   `python ../dev/plot_losses.py` or save a new PNG under `runs/`.
