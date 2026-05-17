# Installation

AirGrep runs on Windows, Linux, and macOS.  The model is the same — only
the RTL-SDR dongle setup differs by OS.  If you don't have a dongle, you
can still run everything in `--demo` mode against the WAV samples
included in `samples/`.

- [1. Python environment](#1-python-environment)
- [2. PyTorch (GPU vs CPU)](#2-pytorch-gpu-vs-cpu)
- [3. Everything else](#3-everything-else)
- [4. Model weights](#4-model-weights)
- [5. Quick sanity check (no dongle needed)](#5-quick-sanity-check)
- [6. RTL-SDR dongle setup](#6-rtl-sdr-dongle-setup)
- [7. Memory notes for Gemma 4 E4B](#7-memory-notes)
- [Troubleshooting](#troubleshooting)

---

## 1. Python environment

Python **3.11** is what AirGrep was developed and tested on.  Other 3.10+
versions likely work but aren't verified.

```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash / cmd):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

## 2. PyTorch (GPU vs CPU)

PyTorch is installed separately because the CUDA build depends on your
GPU / driver.  **Do this before `pip install -r requirements.txt`** or
pip will grab the wrong torch.

**GPU (recommended) — NVIDIA with CUDA 13:**
```bash
pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

**GPU — older CUDA 12.x:** substitute `cu121` or `cu124` in the URL
above and match your driver.

**CPU only:**
```bash
pip install torch==2.10.0
```
AirGrep works on CPU but inference is **minutes per clip** instead of
seconds.  Use `--cpu` on every command (see [§5](#5-quick-sanity-check))
and plan on >= 32 GB system RAM (the bf16 model weights are ~16 GB and
need to fit in RAM with headroom).  Live SDR monitoring isn't practical
in this mode; use `--demo` against pre-captured WAVs.

## 3. Everything else

```bash
pip install -r requirements.txt
```

This installs transformers, textual, numpy/scipy/librosa/soundfile, and
pyrtlsdr.  On Linux and macOS, pyrtlsdr's `librtlsdr` system dependency
needs to be installed separately:

- **Ubuntu/Debian:** `sudo apt install librtlsdr-dev`
- **Fedora:** `sudo dnf install rtl-sdr-devel`
- **macOS (Homebrew):** `brew install librtlsdr`
- **Windows:** bundled `.dll` ships with pyrtlsdr; no separate install
  — but see [§6](#6-rtl-sdr-dongle-setup) for the driver swap.

## 4. Model weights

The fine-tuned checkpoint
[`wames123/airgrep-asr-gemma-4-e4b`](https://huggingface.co/wames123/airgrep-asr-gemma-4-e4b)
downloads automatically on first run — it's the default `--model-path`.
Transformers caches it under `~/.cache/huggingface/`. The download is
~16 GB; to speed it up:

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1   # Windows PowerShell: $env:HF_HUB_ENABLE_HF_TRANSFER=1
```

If you'd rather pre-download to a specific directory (e.g. for offline
use or a shared cache):

```bash
pip install huggingface_hub
huggingface-cli download wames123/airgrep-asr-gemma-4-e4b \
    --local-dir ./airgrep-asr-gemma-4-e4b
python app.py --model-path ./airgrep-asr-gemma-4-e4b -f 144.35
```

## 5. Quick sanity check

Before plugging in an SDR dongle, confirm the model and pipeline work
against the bundled sample:

```bash
python app.py --demo samples/00_2277-149896-0000_02_nominal_ham.wav \
              --watch "anything interesting"
```

You should see the TUI launch, the model transcribe the clip, and the
evaluation pass decide whether to alert.  Press `q` to quit.

**No GPU?** Add `--cpu` (≥ 32 GB system RAM required, demo only — see
[§2](#2-pytorch-gpu-vs-cpu)).  First clip takes 2–5 minutes; subsequent
ones ~30 s.

**Tight on VRAM?** See [§7](#7-memory-notes).

## 6. RTL-SDR dongle setup

Skip this section if you're running `--demo` only.

### Linux / macOS

Plug in the dongle.  That's usually it.  If `python -c "from rtlsdr import
RtlSdr; RtlSdr()"` fails with a permissions error on Linux, add yourself
to the `plugdev` group or copy the udev rules from
<https://github.com/osmocom/rtl-sdr/blob/master/rtl-sdr.rules>.

### Windows — install the WinUSB driver with Zadig

Windows defaults to installing a DVB-T TV tuner driver when you plug an
RTL-SDR dongle in, which blocks library access.  You need to replace it
with a generic WinUSB driver using Zadig.

**One-time steps:**

1. **Download Zadig** from <https://zadig.akeo.ie/> (standalone `.exe`,
   no installer).
2. **Plug in the RTL-SDR dongle** before launching Zadig.
3. **Run `zadig.exe`** as Administrator.
4. In the menu bar: **Options → List All Devices**.
5. In the dropdown, **select** `Bulk-In, Interface (Interface 0)` —
   that's the RTL-SDR.  The name may also appear as `RTL2832U` or
   `RTL2838UHIDIR`.  **Do not** pick anything that says "HID" or
   "Composite."  If you have any doubt, **unplug the dongle and re-plug**
   — only the correct device disappears and reappears.
6. On the right side, select **`WinUSB`** as the driver to install
   (use the arrow buttons to pick it if it's not already selected).
7. Click **Replace Driver** (or **Install Driver** the first time).
   It takes 30–60 seconds.
8. **Unplug and re-plug** the dongle once it finishes.

Verify by running a quick Python snippet:

```bash
python -c "from rtlsdr import RtlSdr; s=RtlSdr(); print('OK', s.get_device_serial_addresses()); s.close()"
```

You should see `OK [...]` with no exception.  If you get a `LibUsbError`,
re-run Zadig and confirm you picked the right device.

**Gotcha:** if you ever plug the dongle into a *different USB port*,
Windows may try to reinstall the old TV-tuner driver.  Just re-run
Zadig on that port.

## 7. Memory notes

The Gemma 4 E4B architecture advertises a ~4B-parameter memory
footprint at inference via Per-Layer Embedding (PLE) offload.  **In
HuggingFace Transformers as of v5.5, this offload is not automatic.**
Without intervention, the model loads the full ~8B parameters into
VRAM (~16 GB in bf16).

**Minimum supported configuration: a single ~16 GB consumer GPU**
(e.g. RTX 4080, RTX 3090, RTX 4090).  This is what AirGrep is
developed and tested on.

**Tuning headroom on a 16 GB card:** `--max-gpu-memory <size>` caps
the model's VRAM allocation and spills the remainder to CPU RAM via
`accelerate`.  Useful if other processes need GPU memory or if you're
hitting OOM on long-running sweeps.  Expect slower inference from
PCIe transfers.

**Smaller GPUs (< 16 GB) are not currently supported.**  4-bit
quantization via bitsandbytes does *not* work with the Gemma 4 audio
encoder due to a dtype bug in `Linear4bit`, and the PLE-offload path
that would shrink the footprint to ~4B parameters isn't wired up
in Transformers v5.5.  If support for smaller GPUs lands in a future
Transformers release, AirGrep should pick it up without code changes.

**No GPU at all:** see [§2](#2-pytorch-gpu-vs-cpu) — `--cpu` mode
works for `--demo` only and needs ≥ 32 GB system RAM.

## Troubleshooting

**`LibUsbError` / `Could not open device` on Windows:** re-run Zadig
([§6](#6-rtl-sdr-dongle-setup)).

**`CUDA out of memory` while loading the model:** see
[§7](#7-memory-notes).  Either add `--max-gpu-memory <size>` or pass
`--cpu`.

**`OSError: cannot load library 'librtlsdr'` on Linux/macOS:** install
the system package (see [§3](#3-everything-else)).

**Model download is slow / keeps restarting:** use
`huggingface-cli download ... --resume-download` or set
`HF_HUB_ENABLE_HF_TRANSFER=1` and `pip install hf_transfer`.

**First inference takes 2+ minutes even on GPU:** normal.  Transformers
compiles kernels on the first forward pass.  Subsequent calls are fast.

**ASR output echoes the system prompt:** means the audio is effectively
silent.  The pre-LLM RMS energy gate (`SILENCE_RMS_THRESHOLD=0.005` in
`llm.py`) normally catches this — check `llm_debug.log` for the measured
RMS value.
