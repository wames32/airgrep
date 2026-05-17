---
license: apache-2.0
base_model: google/gemma-4-E4B-it
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - audio
  - asr
  - speech-recognition
  - lora
  - peft
  - radio
  - sdr
  - gemma-4
language:
  - en
datasets:
  - openslr/librispeech_asr
metrics:
  - wer
---

# airgrep-asr-gemma-4-e4b

Audio-encoder LoRA fine-tune of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
specialized for **noisy narrowband-FM radio speech** — amateur radio,
public-service bands, and degraded RF links where vanilla speech models
struggle with callsigns, weak signals, and squelch artifacts.

This is the model behind [AirGrep](https://github.com/wames123/airgrep) —
an SDR monitor that transcribes radio traffic and alerts on
user-specified content in plain language.

## Quick use

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import soundfile as sf

processor = AutoProcessor.from_pretrained("wames123/airgrep-asr-gemma-4-e4b")
model = AutoModelForImageTextToText.from_pretrained(
    "wames123/airgrep-asr-gemma-4-e4b",
    dtype="bfloat16",
    device_map="auto",
)

audio, sr = sf.read("clip.wav")  # 16 kHz mono, <= 29 s

messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text":
            "You are transcribing audio from a radio receiver. "
            "The audio may contain amateur (ham) radio transmissions "
            "with callsigns. Transcribe exactly what you hear. "
            "Output only the transcription."},
    ],
}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_tensors="pt", return_dict=True,
).to(model.device)

out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(processor.tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

| | |
|---|---|
| Base model | `google/gemma-4-E4B-it` (frozen decoder) |
| Method | LoRA (r=16, α=32) on the audio encoder only |
| Trainable params | 7.02M / 7.94B (0.09%) |
| Training data | LibriSpeech `train-clean-100` (27,927 utterances, 245 speakers) |
| Augmentation | 6-preset synthetic radio degrader: nominal NBFM, weak signal, overmodulation, clean-link, squelched, ducked |
| Optimizer | AdamW 8-bit, lr 2e-4, cosine schedule, 100 warmup steps |
| Batch | 1 × grad-accum 8 (effective batch 8) |
| Steps | 1000 |
| Hardware | 1× RTX 3090, gradient checkpointing |
| Wall clock | 8.8 hours |
| Val loss | 0.340 → 0.297 (still trending down) |

40% of training clips concatenate 2–3 same-speaker utterances with
0.6–2.0s silence gaps to simulate multi-over exchanges. 94% of
transcripts are case-and-punctuation-restored (via Qwen 3.5 9B); 6%
fall back to LibriSpeech's native all-caps.

Full training script, augmenter, and reproducibility notes are in the
[AirGrep repo](https://github.com/wames123/airgrep/tree/main/finetune).

## Evaluation — Word Error Rate by radio condition

35 clips (5 utterances × 7 conditions):

| Condition | Base Gemma 4 E4B | Fine-tuned | Δ |
|-----------|:---:|:---:|:---:|
| Clean | 9.8% | 13.2% | −3.4% |
| Clean (link sim) | 15.7% | 11.5% | **+4.2%** |
| Nominal Ham | 15.0% | 13.2% | +1.8% |
| Overmodulated | 19.2% | 14.1% | **+5.1%** |
| Weak Signal | 21.6% | 19.1% | +2.5% |
| Ducked | 14.8% | 11.5% | +3.3% |
| Squelched | 25.4% | 18.2% | **+7.2%** |
| **Overall** | **14.5%** | **12.0%** | **+2.5%** |

Gains concentrate on the most degraded conditions (squelched, over-
modulated, weak signal) — exactly the conditions that matter for
real-world radio monitoring. The slight regression on perfectly clean
audio is expected: the LoRA trades clean-speech specialization for
robustness to radio artifacts.

**Canonical callsign test clip** (hand-spoken `"This is KJ7ZBB looking
for a signal report"`, NBFM 144.35 MHz, RTL-SDR capture):

| Model | Transcript |
|---|---|
| Base Gemma 4 E4B | `This is KJ7CBB looking for a signal report.` (C → Z error) |
| Fine-tuned (this model) | `This is KJ7 ZBB looking for a signal report.` (0 content errors) |

## Intended use

- Amateur radio / ham radio monitoring
- NOAA weather radio, public-service bands
- SDR-based situational awareness (emergency frequencies, etc.)
- Research on radio-degraded ASR

## Out-of-scope / limitations

- **English only.** Training data was English LibriSpeech; other
  languages will regress vs. the base model.
- **29-second audio cap** per inference call (Gemma 4 audio encoder
  limit).
- **Single-speaker at a time.** No diarization.
- **Not intended for decoding encrypted or scrambled transmissions.**
- **Not a safety-critical system.** Do not use as the sole channel for
  emergency response; always verify alerts against the live audio.
- Clean-speech WER is ~3% worse than the base model. If your use case
  is studio audio, prefer the base model.

## License

Apache 2.0 (inherited from Gemma 4). See
[Gemma 4 launch post](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/).
Gemma is a trademark of Google LLC.

## Citation

If this model is useful in your work, a link back to the AirGrep repo
is appreciated:

```
@software{airgrep2026,
  author = {wames123},
  title = {AirGrep: radio monitoring with fine-tuned Gemma 4 E4B},
  year = {2026},
  url = {https://github.com/wames123/airgrep}
}
```
