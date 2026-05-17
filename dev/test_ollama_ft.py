"""Test the fine-tuned Gemma 4 E4B served by Ollama on the debug clip.

Compares gemma4-ft:e4b (our merged LoRA) vs gemma4:e4b (base).
Expected on debug clip: HF-merged outputs 'This is KJ7 ZBB looking for a signal report.'
Base outputs 'This is KJ7CBB looking for a signal report.'
"""
import base64
import time
from pathlib import Path
from ollama import Client

WAV = Path(__file__).resolve().parents[1] / "debug_audio" / "cycle_20260412_223734_144.35MHz.wav"

PROMPT_SYSTEM = (
    "You are transcribing audio from a radio receiver. "
    "The audio may contain amateur (ham) radio transmissions with callsigns. "
    "Transcribe exactly what you hear. Output only the transcription."
)

def run(model: str, audio_b64: str, n: int = 3):
    client = Client(host="http://localhost:11434")
    print(f"\n=== {model} ===")
    for i in range(1, n + 1):
        messages = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": "Transcribe this audio.", "images": [audio_b64]},
        ]
        t0 = time.perf_counter()
        try:
            r = client.chat(model, messages=messages, think=False, options={"num_ctx": 8000})
            text = r.message.content.strip()
            print(f"  [{i}] ({time.perf_counter()-t0:.1f}s) {text!r}")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            time.sleep(5)

def main():
    print(f"Audio: {WAV.name}")
    print(f"Expected (FT): 'This is KJ7 ZBB looking for a signal report.'")
    audio_b64 = base64.b64encode(WAV.read_bytes()).decode("ascii")
    run("gemma4-ft:e4b", audio_b64, n=3)
    run("gemma4:e4b", audio_b64, n=2)

if __name__ == "__main__":
    main()
