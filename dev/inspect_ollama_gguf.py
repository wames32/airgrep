"""Dump structure of Ollama's gemma4:e4b GGUF blob.

Goals:
  1. Confirm whether audio encoder tensors are present at all.
  2. List the audio-encoder tensor names + shapes (so we know what
     the Ollama mmproj naming convention is).
  3. Dump metadata keys (so we know what model-arch flags Ollama set).
"""
from __future__ import annotations

from pathlib import Path
from gguf import GGUFReader

BLOB = Path("C:/Users/wafor/.ollama/models/blobs/"
            "sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a")

reader = GGUFReader(str(BLOB), "r")

print("=" * 70)
print("METADATA KEYS")
print("=" * 70)
for field in reader.fields.values():
    # Pretty-print scalar/string fields; skip large arrays
    val_repr = "<array>"
    try:
        if len(field.data) == 1:
            v = field.parts[field.data[0]]
            if hasattr(v, "tobytes"):
                try:
                    val_repr = v.tobytes().decode("utf-8", errors="replace")
                except Exception:
                    val_repr = str(v[:8]) + ("..." if len(v) > 8 else "")
            else:
                val_repr = str(v)
        else:
            val_repr = f"<{len(field.data)} items>"
    except Exception as e:
        val_repr = f"<err: {e}>"
    print(f"  {field.name:60s}  {val_repr[:80]}")

print()
print("=" * 70)
print(f"TENSOR COUNT: {len(reader.tensors)}")
print("=" * 70)

# Bucket tensors by prefix to see structure
buckets: dict[str, list[tuple[str, tuple, str]]] = {}
for t in reader.tensors:
    prefix = t.name.split(".")[0] if "." in t.name else "(root)"
    buckets.setdefault(prefix, []).append((t.name, tuple(t.shape), str(t.tensor_type.name)))

for prefix in sorted(buckets):
    items = buckets[prefix]
    print(f"\n[{prefix}]  ({len(items)} tensors)")
    for name, shape, ttype in items[:12]:
        print(f"  {name:65s}  {str(shape):20s}  {ttype}")
    if len(items) > 12:
        print(f"  ... and {len(items) - 12} more")

# Specifically search for anything audio-like
print()
print("=" * 70)
print("AUDIO-RELATED TENSORS (substring match: audio, conformer, mel)")
print("=" * 70)
audio_keywords = ["audio", "conformer", "mel", "speech", "asr", "a.blk", "mm.a"]
audio_tensors = [t for t in reader.tensors
                 if any(k in t.name.lower() for k in audio_keywords)]
if audio_tensors:
    for t in audio_tensors[:50]:
        print(f"  {t.name:65s}  {tuple(t.shape)}  {t.tensor_type.name}")
    if len(audio_tensors) > 50:
        print(f"  ... and {len(audio_tensors) - 50} more")
else:
    print("  NONE FOUND — this GGUF appears to be text-only.")
