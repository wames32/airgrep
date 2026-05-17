"""Bundle text + mmproj GGUFs into a single Ollama-compatible Gemma 4 GGUF.

Ollama's bundled gemma4:e4b is one file containing both the text decoder
(`blk.*` etc) AND the audio+vision encoders (`a.*`, `v.*`, `mm.*`). When you
pass a separate mmproj file, Ollama's gemma4 native engine doesn't engage and
inference falls back to legacy llama.cpp which lacks the gemma4 architecture.

So: copy all metadata + tensors from BOTH GGUFs into one new GGUF.
Tensor naming already matches Ollama's layout (verified with inspect_my_ggufs.py)
so this is a structural merge with no renaming.
"""
from __future__ import annotations

import time
from pathlib import Path

from gguf import (
    GGMLQuantizationType,
    GGUFReader,
    GGUFValueType,
    GGUFWriter,
)

RUNS = Path(__file__).parent / "runs"
TEXT = RUNS / "gemma4-e4b-finetune.gguf"
MMPROJ = RUNS / "mmproj-gemma4-e4b-finetune.gguf"
OUT = RUNS / "gemma4-e4b-finetune-bundled.gguf"

# These KV keys are managed by GGUFWriter itself or are the reader's own
# header bookkeeping — never copy them from a source file.
SKIP_KV = {
    "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
}

# When merging mmproj metadata into the text-arch file, drop keys that would
# collide with the text architecture's view of the model.
MMPROJ_DROP = {
    "general.architecture",   # mmproj's is "clip"; we keep text's "gemma4"
    "general.type",           # mmproj's is "mmproj"; keep text's "model"
    "general.name",
    "general.size_label",
    "general.file_type",
    "general.quantization_version",
    "general.sampling.top_k",
    "general.sampling.top_p",
    "general.sampling.temp",
}


def field_value(field):
    """Decode a ReaderField into a python value (or list for arrays)."""
    if len(field.types) == 0:
        return None
    main_type = field.types[0]
    if main_type == GGUFValueType.STRING:
        return field.parts[field.data[0]].tobytes().decode("utf-8")
    if main_type == GGUFValueType.ARRAY:
        sub_type = field.types[1] if len(field.types) > 1 else None
        if sub_type == GGUFValueType.STRING:
            return [field.parts[i].tobytes().decode("utf-8") for i in field.data]
        return [field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist")
                else field.parts[i] for i in field.data]
    # Scalar: single numpy element
    return field.parts[field.data[0]].tolist()[0]


def emit_kv(writer: GGUFWriter, name: str, field):
    """Write one KV pair to writer, preserving GGUF type."""
    main_type = field.types[0]
    if main_type == GGUFValueType.ARRAY:
        sub_type = field.types[1]
        if sub_type == GGUFValueType.STRING:
            vals = [field.parts[i].tobytes().decode("utf-8") for i in field.data]
            writer.add_array(name, vals)
        else:
            # numeric array — flatten parts to python list
            vals = []
            for i in field.data:
                p = field.parts[i]
                if hasattr(p, "tolist"):
                    vals.extend(p.tolist())
                else:
                    vals.append(p)
            writer.add_array(name, vals)
        return
    val = field_value(field)
    if main_type == GGUFValueType.STRING:
        writer.add_string(name, val)
    elif main_type == GGUFValueType.BOOL:
        writer.add_bool(name, bool(val))
    elif main_type == GGUFValueType.UINT8:
        writer.add_uint8(name, int(val))
    elif main_type == GGUFValueType.INT8:
        writer.add_int8(name, int(val))
    elif main_type == GGUFValueType.UINT16:
        writer.add_uint16(name, int(val))
    elif main_type == GGUFValueType.INT16:
        writer.add_int16(name, int(val))
    elif main_type == GGUFValueType.UINT32:
        writer.add_uint32(name, int(val))
    elif main_type == GGUFValueType.INT32:
        writer.add_int32(name, int(val))
    elif main_type == GGUFValueType.UINT64:
        writer.add_uint64(name, int(val))
    elif main_type == GGUFValueType.INT64:
        writer.add_int64(name, int(val))
    elif main_type == GGUFValueType.FLOAT32:
        writer.add_float32(name, float(val))
    elif main_type == GGUFValueType.FLOAT64:
        writer.add_float64(name, float(val))
    else:
        raise ValueError(f"unhandled type for KV {name}: {main_type}")


def main():
    print(f"Reading text:   {TEXT.name}")
    text_reader = GGUFReader(str(TEXT), "r")
    print(f"Reading mmproj: {MMPROJ.name}")
    mmproj_reader = GGUFReader(str(MMPROJ), "r")

    print(f"  text:   {len(text_reader.fields)} KV, {len(text_reader.tensors)} tensors")
    print(f"  mmproj: {len(mmproj_reader.fields)} KV, {len(mmproj_reader.tensors)} tensors")

    print(f"\nWriting bundled output: {OUT.name}")
    writer = GGUFWriter(str(OUT), arch="gemma4", use_temp_file=False)

    # 1) KV from text (skip GGUF.* and general.architecture — writer adds them)
    text_seen = set()
    for fname, field in text_reader.fields.items():
        if fname in SKIP_KV:
            continue
        if fname == "general.architecture":
            # GGUFWriter already emits this via the constructor's arch=
            text_seen.add(fname)
            continue
        emit_kv(writer, fname, field)
        text_seen.add(fname)
    print(f"  copied {len(text_seen)} KV from text")

    # 2) KV from mmproj (skip duplicates and clip-only general.* keys)
    mmproj_added = 0
    for fname, field in mmproj_reader.fields.items():
        if fname in SKIP_KV:
            continue
        if fname in MMPROJ_DROP:
            continue
        if fname in text_seen:
            continue  # duplicate — text wins
        emit_kv(writer, fname, field)
        mmproj_added += 1
    print(f"  copied {mmproj_added} KV from mmproj")

    # 3) Tensors from text
    t0 = time.perf_counter()
    for i, t in enumerate(text_reader.tensors):
        writer.add_tensor(
            t.name, t.data,
            raw_shape=tuple(int(x) for x in t.shape),
            raw_dtype=t.tensor_type,
        )
    print(f"  registered {len(text_reader.tensors)} text tensors")

    # 4) Tensors from mmproj
    for t in mmproj_reader.tensors:
        writer.add_tensor(
            t.name, t.data,
            raw_shape=tuple(int(x) for x in t.shape),
            raw_dtype=t.tensor_type,
        )
    print(f"  registered {len(mmproj_reader.tensors)} mmproj tensors")
    print(f"  total tensors: {len(text_reader.tensors) + len(mmproj_reader.tensors)}")

    print(f"\nWriting GGUF (this takes a while — copies {OUT.stat().st_size if OUT.exists() else '~16'} GB of data)...")
    t0 = time.perf_counter()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    elapsed = time.perf_counter() - t0
    sz_gb = OUT.stat().st_size / (1024**3)
    print(f"\nDONE in {elapsed:.1f}s — wrote {OUT.name} ({sz_gb:.2f} GB)")


if __name__ == "__main__":
    main()
