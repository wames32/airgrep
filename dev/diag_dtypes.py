"""Diagnostic: check if audio_tower stays in bf16 with llm_int8_skip_modules."""
import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"

cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["audio_tower", "embed_audio", "vision_tower", "embed_vision", "lm_head"],
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=cfg, dtype=torch.bfloat16, device_map={"": 0},
)

def report(prefix):
    mods = dict(model.named_modules())
    for key_prefix in ["model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear",
                       "model.audio_tower.output_proj",
                       "model.language_model.layers.0.self_attn.q_proj",
                       "model.embed_audio.embedding_projection"]:
        if key_prefix in mods:
            m = mods[key_prefix]
            t = type(m).__name__
            w = getattr(m, "weight", None)
            wd = str(w.dtype) if w is not None else "no weight"
            print(f"  [{prefix}] {key_prefix}  type={t}  weight.dtype={wd}")
        else:
            print(f"  [{prefix}] {key_prefix}  NOT FOUND")

print("=== After load (pre-PEFT) ===")
report("raw")
