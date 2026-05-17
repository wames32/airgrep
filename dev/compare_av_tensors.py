"""Compare audio + vision tensor shapes between Ollama's gemma4:e4b blob
and our bundled GGUF, looking for mismatches that would trip the Ollama
native gemma4 runner's GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2).
"""
from pathlib import Path
from gguf import GGUFReader

OLLAMA = Path("C:/Users/wafor/.ollama/models/blobs/"
              "sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a")
OURS = Path(__file__).parent / "runs" / "gemma4-e4b-finetune-bundled.gguf"

ra = GGUFReader(str(OLLAMA), "r")
rb = GGUFReader(str(OURS), "r")

# Build {name: (shape, type)} maps for av tensors
def av_map(reader):
    m = {}
    for t in reader.tensors:
        if t.name.startswith(("a.", "v.", "mm.")):
            m[t.name] = (tuple(int(x) for x in t.shape), t.tensor_type.name)
    return m

A = av_map(ra)
B = av_map(rb)

print(f"Ollama av tensors: {len(A)}")
print(f"Ours   av tensors: {len(B)}")

print("\n== In ours but not in Ollama ==")
extra = sorted(set(B) - set(A))
for n in extra[:30]:
    print(f"  {n:60s} {B[n]}")
print(f"  ...total extra: {len(extra)}")

print("\n== In Ollama but not in ours ==")
miss = sorted(set(A) - set(B))
for n in miss[:30]:
    print(f"  {n:60s} {A[n]}")
print(f"  ...total missing: {len(miss)}")

print("\n== Shape/type mismatches ==")
both = set(A) & set(B)
mm = []
for n in sorted(both):
    if A[n] != B[n]:
        mm.append((n, A[n], B[n]))
for n, a, b in mm[:50]:
    print(f"  {n:60s}  ollama={a}  ours={b}")
print(f"  ...total mismatches: {len(mm)}")
