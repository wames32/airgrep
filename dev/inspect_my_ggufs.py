"""Dump tensor names + shapes from our converted text and mmproj GGUFs.

Side-by-side with Ollama's bundled gemma4:e4b layout so we can plan the
tensor renaming required to bundle into a single GGUF Ollama will accept.
"""
from __future__ import annotations
from pathlib import Path
from gguf import GGUFReader

RUNS = Path(__file__).parent / "runs"
TEXT = RUNS / "gemma4-e4b-finetune.gguf"
MMPROJ = RUNS / "mmproj-gemma4-e4b-finetune.gguf"

for label, path in [("TEXT", TEXT), ("MMPROJ", MMPROJ)]:
    print("=" * 70)
    print(f"{label}: {path.name}")
    print("=" * 70)
    r = GGUFReader(str(path), "r")

    print(f"\nMETADATA ({len(r.fields)} keys):")
    for f in r.fields.values():
        try:
            if len(f.data) == 1:
                v = f.parts[f.data[0]]
                if hasattr(v, "tobytes"):
                    try:
                        s = v.tobytes().decode("utf-8", errors="replace")
                    except Exception:
                        s = str(v[:8])
                else:
                    s = str(v)
            else:
                s = f"<{len(f.data)} items>"
        except Exception as e:
            s = f"<err: {e}>"
        print(f"  {f.name:55s}  {s[:60]}")

    print(f"\nTENSORS ({len(r.tensors)}):")
    buckets: dict[str, list] = {}
    for t in r.tensors:
        prefix = ".".join(t.name.split(".")[:2]) if "." in t.name else t.name
        buckets.setdefault(prefix, []).append((t.name, tuple(t.shape), t.tensor_type.name))
    for prefix in sorted(buckets):
        items = buckets[prefix]
        print(f"\n  [{prefix}] ({len(items)} tensors)")
        for name, shape, ttype in items[:6]:
            print(f"    {name:60s}  {str(shape):20s}  {ttype}")
        if len(items) > 6:
            print(f"    ... +{len(items) - 6} more")
    print()
