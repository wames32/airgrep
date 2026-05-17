"""Use qwen3.5:9b via Ollama to restore casing + punctuation on LibriSpeech.

Why: LibriSpeech transcripts are ALL CAPS NO PUNCTUATION, which is
ugly and fights against Gemma 4's pretraining distribution. Training on
natural-looking text makes the fine-tune both easier (closer to the
decoder's pretraining prior) and produces more readable output.

Method:
  - Batch N transcripts per LLM call (much faster than one at a time).
  - Strict prompt: preserve every word, only add case + punctuation.
  - Verify output: reject batches where decapitalising the LLM output
    doesn't match the original (word-identity check). Retry such batches
    one-at-a-time for robustness.

Output: writes an enriched manifest with a new 'text_pretty' field.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ollama import Client

CACHE_DIR = Path(__file__).parent / "librispeech_cache"
MANIFEST_IN = CACHE_DIR / "manifest.jsonl"
MANIFEST_OUT = CACHE_DIR / "manifest_pretty.jsonl"

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3.5:9b"

SYSTEM_PROMPT = (
    "You restore capitalization and punctuation in transcripts. "
    "Follow these rules STRICTLY:\n"
    "1. Preserve every single word exactly as given. Do not add, remove, or substitute any word.\n"
    "2. Add natural punctuation (periods, commas, question marks, apostrophes in contractions).\n"
    "3. Apply natural capitalization: start of sentences, proper nouns, 'I'.\n"
    "4. Do NOT add quotation marks around the whole output.\n"
    "5. Do NOT add any explanation, preamble, or commentary.\n"
    "6. For batch inputs (numbered lines like '1. TEXT'), output exactly the same count of lines "
    "in the same numbered format. Each number goes on its own line, followed by the restored text.\n"
)


def build_batch_prompt(texts: list[str]) -> str:
    lines = [f"{i+1}. {t}" for i, t in enumerate(texts)]
    return (
        "Restore capitalization and punctuation for each numbered line below. "
        "Output exactly the same number of lines in the same numbered format. "
        "Do not merge lines. Do not add any extra text.\n\n"
        + "\n".join(lines)
    )


_num_line = re.compile(r"^\s*(\d+)\s*[.)\-:]\s*(.*)$")


def parse_batch_response(raw: str, expected_n: int) -> list[str] | None:
    out: dict[int, str] = {}
    current_idx: int | None = None
    current_buf: list[str] = []
    for line in raw.splitlines():
        m = _num_line.match(line)
        if m:
            if current_idx is not None:
                out[current_idx] = " ".join(current_buf).strip()
            current_idx = int(m.group(1))
            current_buf = [m.group(2)]
        else:
            if current_idx is not None and line.strip():
                current_buf.append(line.strip())
    if current_idx is not None:
        out[current_idx] = " ".join(current_buf).strip()

    if len(out) != expected_n:
        return None
    ordered = [out.get(i + 1, "") for i in range(expected_n)]
    if any(not s for s in ordered):
        return None
    return ordered


_word_re = re.compile(r"[A-Z']+")


def words(s: str) -> list[str]:
    return _word_re.findall(s.upper())


def verify(original: str, pretty: str) -> bool:
    """Words must match exactly (case-insensitive, apostrophes preserved).

    We accept minor variation in whether the LLM decomposes a contraction,
    so only treat alphabetic sequences with apostrophes as words.
    """
    a = words(original)
    b = words(pretty)
    return a == b


# ── Ollama call (with retry) ────────────────────────────────────────


def call_ollama(client: Client, prompt: str, timeout_s: int = 120) -> str:
    resp = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0, "num_ctx": 4096},
        think=False,
    )
    return (resp.message.content or "").strip()


def beautify_batch(client: Client, texts: list[str]) -> list[str | None]:
    """Return pretty versions (None for failures)."""
    prompt = build_batch_prompt(texts)
    for attempt in range(2):
        try:
            raw = call_ollama(client, prompt)
        except Exception as e:
            print(f"    ollama error: {e}")
            return [None] * len(texts)
        parsed = parse_batch_response(raw, len(texts))
        if parsed is None:
            continue
        # Per-item verification
        result: list[str | None] = []
        for orig, pretty in zip(texts, parsed):
            result.append(pretty if verify(orig, pretty) else None)
        return result
    return [None] * len(texts)


def beautify_one(client: Client, text: str) -> str | None:
    """Singleton fallback for items that failed in a batch."""
    prompt = (
        "Restore capitalization and punctuation for the line below. "
        "Output only the restored line, no preamble, no quotes.\n\n"
        f"1. {text}"
    )
    try:
        raw = call_ollama(client, prompt)
    except Exception:
        return None
    m = _num_line.match(raw.splitlines()[0] if raw else "")
    candidate = m.group(2).strip() if m else raw.strip()
    return candidate if verify(text, candidate) else None


# ── Main ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N rows (for testing)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent Ollama calls (qwen3.5:9b is small enough)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already present in manifest_pretty.jsonl")
    args = ap.parse_args()

    rows = []
    with open(MANIFEST_IN, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded manifest: {len(rows)} rows")

    already: set[str] = set()
    if args.resume and MANIFEST_OUT.exists():
        with open(MANIFEST_OUT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("text_pretty"):
                        already.add(r["id"])
                except json.JSONDecodeError:
                    pass
        print(f"Resume: {len(already)} rows already beautified.")

    todo = [r for r in rows if r["id"] not in already]
    print(f"To process: {len(todo)}")

    client = Client(host=OLLAMA_HOST)

    batches = [todo[i : i + args.batch_size]
               for i in range(0, len(todo), args.batch_size)]
    print(f"Batches: {len(batches)} (batch_size={args.batch_size}, workers={args.workers})")

    # Open in append mode so resume works
    out_f = open(MANIFEST_OUT, "a", encoding="utf-8")

    n_done = 0
    n_failed = 0
    t0 = time.perf_counter()

    def process_batch(batch_rows):
        texts = [r["text"] for r in batch_rows]
        pretties = beautify_batch(client, texts)
        # Retry failures one-at-a-time
        for i, p in enumerate(pretties):
            if p is None:
                pretties[i] = beautify_one(client, texts[i])
        out = []
        for r, p in zip(batch_rows, pretties):
            row = dict(r)
            row["text_pretty"] = p  # may be None
            out.append(row)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = [exe.submit(process_batch, b) for b in batches]
        for fut in as_completed(futures):
            result_rows = fut.result()
            for r in result_rows:
                out_f.write(json.dumps(r) + "\n")
                if r["text_pretty"] is None:
                    n_failed += 1
                n_done += 1
            out_f.flush()
            if n_done % 100 < len(result_rows):
                elapsed = time.perf_counter() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - n_done) / rate if rate > 0 else 0
                print(f"  {n_done:5d}/{len(todo)}  failed={n_failed}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s")

    out_f.close()
    print(f"\nDone. {n_done} processed, {n_failed} failures.")
    print(f"Wrote {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
