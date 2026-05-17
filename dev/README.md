# dev/

Historical diagnostic, data-prep, and investigation scripts. **Not used by
the runtime** — `app.py`, `pipeline.py`, `benchmark.py` do not import from
this directory.

They are kept as a paper trail of the engineering work behind AirGrep:

- **ASR diagnostics** (base-vs-fine-tune comparison, adapter eval, LoRA
  forward-pass sanity checks): `baseline_inference.py`, `eval_adapter.py`,
  `test_lora_forward.py`, `verify_merged.py`, `diag_dtypes.py`,
  `inspect_model.py`
- **Training-data prep** (one-time, requires LibriSpeech + Ollama/Qwen for
  transcript beautification): `cache_librispeech.py`, `download_dataset.py`,
  `beautify_transcripts.py`, `generate_samples.py`
- **Audio analysis**: `analyze_clip_lengths.py`, `analyze_radio_clip.py`
- **Loss plotting**: `plot_losses.py`
- **Ollama / GGUF conversion investigation** (abandoned — llama.cpp's
  HF→GGUF converter rejects Gemma 4's audio-tower tensor layout; see the
  project CLAUDE.md for the full writeup): `bundle_gguf.py`,
  `inspect_my_ggufs.py`, `inspect_ollama_gguf.py`, `compare_av_tensors.py`,
  `test_ollama_ft.py`

Most of these reference paths and environments specific to the author's
machine. They are here for reference, not for re-execution.
