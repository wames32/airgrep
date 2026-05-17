"""Load the merged HF checkpoint (no PEFT) and verify it still produces
'KJ7 ZBB' on the debug clip. Confirms the merge math preserved the fine-tune.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MERGED = Path(__file__).parent / "runs" / "merged_bf16"
WAV = Path(__file__).resolve().parents[1] / "debug_audio" / "cycle_20260412_223734_144.35MHz.wav"


def load_audio_16k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x[: 29 * 16000]


print(f"Loading merged checkpoint from {MERGED}...")
proc = AutoProcessor.from_pretrained(str(MERGED))
model = AutoModelForImageTextToText.from_pretrained(
    str(MERGED), dtype=torch.bfloat16, device_map={"": 0},
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

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=1.0)
new_tokens = out[0, inputs["input_ids"].shape[1]:]
text = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
print(f"\nMERGED model output: {text!r}")

if "KJ7" in text and "ZBB" in text.replace(" ", ""):
    print("[OK] MERGE PRESERVED FINE-TUNE")
else:
    print("[FAIL] Output does not match fine-tuned behaviour")
