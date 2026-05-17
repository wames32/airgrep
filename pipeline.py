#!/usr/bin/env python3
"""AirGrep headless monitor — grep for the airwaves (no TUI).

Each capture cycle is stateless: a fresh prompt + one audio clip.
This avoids false positives from prior audio leaking into context and
keeps memory usage minimal for edge deployment.

Architecture:
    SDR IQ capture  ->  FM demod / filter / decimate  ->  WAV
        ->  Pass 1: Gemma 4 E4B transcribes audio (fine-tuned ASR)
        ->  Pass 2: Gemma 4 E4B evaluates transcript (tool-calling)  ->  alert_user

Usage examples:
    # Monitor public radio for weather alerts
    python pipeline.py -f 102.7 --watch "emergency alerts, severe weather"

    # Monitor 2m ham calling frequency, single cycle
    python pipeline.py -f 146.52 -d 30 --watch "callsign W7ABC" --once
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from capture import is_wbfm
from llm import analyze_audio


# HuggingFace repo — weights auto-download on first run.
# Override with --model-path to point at a local checkpoint.
DEFAULT_MODEL_PATH = "wames123/airgrep-asr-gemma-4-e4b"


# ── ANSI colors ──────────────────────────────────────────────────────────────

BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
BRIGHT_YELLOW = "\033[1;93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

URGENCY_STYLE = {
    "low": YELLOW,
    "medium": BRIGHT_YELLOW,
    "high": RED + BOLD,
}

# ── Globals set at runtime ───────────────────────────────────────────────────

_log_path: Path = Path("alerts.log")
_alert_count: int = 0
_running: bool = True


# ── Alert callback (called by llm.py when the model fires alert_user) ──────


def alert_user(message: str, urgency: str) -> str:
    """Print an alert to the console and append to the log file."""
    global _alert_count
    _alert_count += 1

    urgency = urgency.lower().strip()
    style = URGENCY_STYLE.get(urgency, YELLOW)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Console
    label = f"[{urgency.upper()}]"
    print(f"\n{style}{'=' * 60}")
    print(f"  ALERT {label}  --  {ts}")
    print(f"  {message}")
    print(f"{'=' * 60}{RESET}\n")

    # Log file
    entry = {"time": ts, "urgency": urgency, "message": message}
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return f"Alert delivered to user ({urgency})."


def _on_status(text: str) -> None:
    """Print LLM status messages to the console."""
    print(f"{DIM}LLM: {text.strip()}{RESET}")


# ── SDR capture cycle ────────────────────────────────────────────────────────


def run_capture_cycle(
    freq_mhz: float,
    duration: float,
    gain: str,
    sample_rate: float,
    audio_rate: int,
) -> str:
    """Capture audio from SDR, return path to temporary WAV file."""
    from rtlsdr import RtlSdr
    from capture import capture_iq, process

    wbfm = is_wbfm(freq_mhz)
    freq_hz = freq_mhz * 1e6

    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate
        sdr.center_freq = freq_hz
        sdr.gain = "auto" if gain == "auto" else float(gain)

        num_samples = int(duration * sample_rate)
        print(f"{CYAN}Capturing {duration:.0f}s on {freq_mhz} MHz...{RESET}")
        iq = capture_iq(sdr, num_samples)
    finally:
        sdr.close()

    audio = process(iq, sample_rate, audio_rate, wbfm)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, audio, audio_rate, subtype="FLOAT")

    return tmp_path


# ── LLM cycle (wraps llm.analyze_audio) ─────────────────────────────────────


def run_llm_cycle(
    model_path: str,
    wav_path: str,
    freq_mhz: float,
    duration: float,
    watch: str,
    force_cpu: bool = False,
    max_gpu_memory: str | None = None,
    save_dir: Path | None = None,
) -> None:
    """Analyze one audio clip using the two-pass pipeline."""
    analyze_audio(
        model_path=model_path,
        wav_path=wav_path,
        freq_mhz=freq_mhz,
        duration=duration,
        watch=watch,
        alert_fn=alert_user,
        on_status=_on_status,
        save_dir=save_dir,
        force_cpu=force_cpu,
        max_gpu_memory=max_gpu_memory,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AirGrep headless monitor — grep for the airwaves. Powered by Gemma 4 E4B.",
    )
    parser.add_argument(
        "-f", "--freq", type=float, required=True,
        help="Center frequency in MHz (e.g. 102.7 or 146.52)",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=29.0,
        help="Capture duration per cycle in seconds (default: 29)",
    )
    parser.add_argument(
        "--watch", type=str, required=True,
        help="What to watch for -- natural language description",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle then exit (default: loop until Ctrl+C)",
    )
    parser.add_argument(
        "--gain", type=str, default="auto",
        help="SDR tuner gain in dB, or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        dest="model_path",
        help=f"HF repo id or local path to model checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--log", type=str, default="alerts.log",
        help="Alert log file path (default: alerts.log)",
    )
    parser.add_argument(
        "-s", "--sample-rate", type=float, default=960000,
        help="SDR sample rate in Hz (default: 960000)",
    )
    parser.add_argument(
        "--audio-rate", type=int, default=48000,
        help="Output audio sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--cpu", action="store_true", dest="force_cpu",
        help="Force CPU inference (no GPU). Slow — use only for testing.",
    )
    parser.add_argument(
        "--max-gpu-memory", type=str, default=None, dest="max_gpu_memory",
        help="Cap GPU VRAM usage (e.g. '7GiB'). Spills to CPU RAM for small GPUs.",
    )
    parser.add_argument(
        "--save-clips", type=str, default=None, dest="save_clips_dir",
        help="Save a copy of every analyzed audio clip to this directory "
             "(disabled by default — clips accumulate unbounded).",
    )
    return parser.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    global _log_path, _running

    args = parse_args(argv)
    _log_path = Path(args.log)

    freq_mhz = args.freq
    mode = "WBFM" if is_wbfm(freq_mhz) else "NBFM"

    # Validate sample rate / audio rate ratio
    dec_factor = args.sample_rate / args.audio_rate
    if abs(dec_factor - round(dec_factor)) > 0.001:
        print(f"ERROR: sample rate ({args.sample_rate:.0f}) is not an integer "
              f"multiple of audio rate ({args.audio_rate}).")
        sys.exit(1)

    # Graceful shutdown on Ctrl+C
    def handle_sigint(sig, frame):
        global _running
        print(f"\n{YELLOW}Ctrl+C received -- finishing current cycle...{RESET}")
        _running = False

    signal.signal(signal.SIGINT, handle_sigint)

    # Print startup banner
    print(f"{BOLD}{'=' * 60}")
    print(f"  AirGrep  |  grep for the airwaves  |  Gemma 4 E4B")
    print(f"{'=' * 60}{RESET}")
    print(f"  Frequency:   {freq_mhz} MHz ({mode})")
    print(f"  Cycle:       {args.duration:.0f}s capture")
    print(f"  Watching:    {args.watch}")
    print(f"  Model:       {Path(args.model_path).name}")
    print(f"  Alert log:   {_log_path}")
    print(f"  Mode:        {'single cycle' if args.once else 'continuous (Ctrl+C to stop)'}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")

    cycle_count = 0
    t_start = time.perf_counter()

    # Overlapped capture + inference: capture cycle N+1 while LLM analyzes N.
    pending_llm: Future | None = None
    pending_wav: str | None = None

    save_dir = Path(args.save_clips_dir) if args.save_clips_dir else None

    def _submit_llm(executor, wav_path):
        return executor.submit(
            run_llm_cycle, args.model_path, wav_path,
            freq_mhz, args.duration, args.watch,
            args.force_cpu, args.max_gpu_memory, save_dir,
        )

    def _collect_llm(future, wav_path):
        try:
            future.result()
        except Exception as e:
            print(f"{RED}LLM error: {e}{RESET}")
        Path(wav_path).unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        while _running:
            cycle_count += 1
            cycle_ts = datetime.now().strftime("%H:%M:%S")
            print(f"{BOLD}-- Cycle {cycle_count} [{cycle_ts}] --{RESET}")

            try:
                wav_path = run_capture_cycle(
                    freq_mhz=freq_mhz,
                    duration=args.duration,
                    gain=args.gain,
                    sample_rate=args.sample_rate,
                    audio_rate=args.audio_rate,
                )

                if pending_llm is not None:
                    print(f"{DIM}Waiting for analysis of previous cycle...{RESET}")
                    _collect_llm(pending_llm, pending_wav)
                    pending_llm = None
                    print()

                print(f"{CYAN}Sending audio to evaluator...{RESET}")

                if args.once:
                    run_llm_cycle(args.model_path, wav_path,
                                  freq_mhz, args.duration, args.watch,
                                  args.force_cpu, args.max_gpu_memory, save_dir)
                    Path(wav_path).unlink(missing_ok=True)
                else:
                    pending_llm = _submit_llm(executor, wav_path)
                    pending_wav = wav_path

            except Exception as e:
                print(f"{RED}Error in cycle {cycle_count}: {e}{RESET}")

            print()

            if args.once:
                break

        if pending_llm is not None:
            print(f"{DIM}Waiting for final analysis...{RESET}")
            _collect_llm(pending_llm, pending_wav)

    elapsed = time.perf_counter() - t_start
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  Monitor stopped")
    print(f"{'=' * 60}{RESET}")
    print(f"  Cycles:    {cycle_count}")
    print(f"  Alerts:    {_alert_count}")
    print(f"  Runtime:   {elapsed:.0f}s")
    print(f"  Log file:  {_log_path}")
    print(f"{BOLD}{'=' * 60}{RESET}")


if __name__ == "__main__":
    main()
