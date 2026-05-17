#!/usr/bin/env python3
"""AirGrep — grep for the airwaves.

Terminal UI powered by Textual + fine-tuned Gemma 4 E4B.

Launch modes:
    # Live SDR monitoring
    python app.py -f 102.7 --watch "emergency alerts, severe weather"

    # Demo mode with WAV files (no dongle needed)
    python app.py --demo path/to/clip.wav
    python app.py --demo path/to/clip1.wav --demo path/to/clip2.wav

    # Demo mode — prompted to browse for files interactively
    python app.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np
import soundfile as sf
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from capture import is_wbfm
from llm import analyze_audio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HuggingFace repo — weights auto-download on first run.
# Override with --model-path to point at a local checkpoint.
DEFAULT_MODEL_PATH = "wames123/airgrep-asr-gemma-4-e4b"
DEFAULT_WATCH = "emergency alerts, severe weather, breaking news, public safety threats"
DEFAULT_FREQ = 162.4
DEFAULT_DURATION = 29.0
DEFAULT_SAMPLE_RATE = 960_000
DEFAULT_AUDIO_RATE = 48_000

# Scanning
SCAN_FLUSH_SAMPLES = 4_096    # ~4ms at 960kHz — flush stale IQ after retune
SCAN_DWELL_SAMPLES = 4_096    # ~4ms at 960kHz — measure signal (~4k samples is plenty for RMS)
DEFAULT_SQUELCH = 0.4         # noise score threshold (lower = signal present)


def infer_step_khz(start_mhz: float, end_mhz: float) -> float:
    """Pick a channel step size based on the frequency band."""
    if start_mhz >= 88.0 and end_mhz <= 108.0:
        return 200.0    # FM broadcast
    return 5.0          # 5 kHz — hits every channel plan (5/10/12.5/15/25 kHz)


# ---------------------------------------------------------------------------
# Continuous SDR capture
# ---------------------------------------------------------------------------


class ContinuousCapture:
    """Signal-gated SDR capture — a 29s window begins the instant a signal
    arrives, not on a fixed cadence.

    The capture thread continuously reads small IQ chunks (~137 ms) and
    scores each one with the same HF-noise metric used elsewhere
    (``capture.detect_signal``).  While the channel is quiet the IQ is
    rolled into a short pre-buffer (~500 ms) and discarded; nothing is
    enqueued, no cycle is burned.  The moment a chunk scores below the
    squelch threshold, the pre-buffer becomes the head of a new capture
    window and we keep accumulating IQ for ``chunk_duration`` seconds.
    The completed window is demod/filter/decimate/normalize'd and pushed
    to the consumer queue.

    This means:
      - Transmissions that start near the end of a (would-be) fixed cycle
        no longer get truncated — the window starts at the onset.
      - No "No signal" cycles show up in the UI on a quiet channel.
      - A continuous stream of traffic produces back-to-back 29s windows.

    If the consumer (LLM) can't keep up and the queue hits ``max_chunks``,
    the oldest chunk is dropped and ``dropped_count`` increments.
    """

    CHUNK_IQ = 131_072  # ~137ms at 960 kHz
    PREBUFFER_SECS = 0.5  # roll-back window so we don't clip signal onset

    def __init__(
        self,
        freq_mhz: float,
        sample_rate: float,
        audio_rate: int,
        gain: str,
        chunk_duration: float = 29.0,
        max_chunks: int = 10,  # ~5 min at 29s/chunk
        squelch: float = DEFAULT_SQUELCH,
        on_state_change: Callable[[str], None] | None = None,
    ) -> None:
        self.freq_mhz = freq_mhz
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.gain = gain
        self.chunk_duration = chunk_duration
        self._max_chunks = max_chunks
        self.squelch = squelch
        self._on_state_change = on_state_change
        self._state = "listening"

        self._queue: deque[np.ndarray] = deque()
        self._dropped = 0
        self._lock = Lock()
        self._stop = Event()
        self._new_freq: float | None = None
        self._thread: Thread | None = None

    def _set_state(self, state: str) -> None:
        """Notify the app on quiet<->signal transitions. Called from
        the capture thread; the callback is responsible for thread safety.
        """
        if state == self._state:
            return
        self._state = state
        if self._on_state_change is not None:
            try:
                self._on_state_change(state)
            except Exception:
                pass

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True, name="sdr-capture")
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def change_freq(self, freq_mhz: float) -> None:
        """Request a retune — clears the queue and restarts accumulation."""
        self._new_freq = freq_mhz

    def has_chunk(self) -> bool:
        """True if at least one processed audio chunk is ready."""
        with self._lock:
            return len(self._queue) > 0

    def next_chunk(self) -> np.ndarray | None:
        """Pop the oldest unanalyzed audio chunk (FIFO). None if empty."""
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    # -- Internal capture loop (runs in daemon thread) ---------------------

    def _run(self) -> None:
        from rtlsdr import RtlSdr
        from capture import process, is_wbfm, detect_signal

        sdr = RtlSdr()
        sdr.sample_rate = self.sample_rate
        sdr.center_freq = self.freq_mhz * 1e6
        sdr.gain = "auto" if self.gain == "auto" else float(self.gain)

        chunk_samples = int(self.chunk_duration * self.sample_rate)
        prebuf_samples = int(self.PREBUFFER_SECS * self.sample_rate)

        # Rolling pre-buffer of recent IQ (only used while idle). When a
        # signal arrives we prepend this so we don't clip the onset.
        prebuf: deque[np.ndarray] = deque()
        prebuf_count = 0

        def _handle_retune() -> bool:
            """Apply a pending retune. Returns True if one was applied."""
            nonlocal prebuf, prebuf_count
            new_freq = self._new_freq
            if new_freq is None:
                return False
            self._new_freq = None
            sdr.center_freq = new_freq * 1e6
            self.freq_mhz = new_freq
            sdr.read_samples(16384)  # flush stale IQ
            prebuf = deque()
            prebuf_count = 0
            with self._lock:
                self._queue.clear()
            return True

        try:
            while not self._stop.is_set():
                _handle_retune()

                iq = sdr.read_samples(self.CHUNK_IQ).astype(np.complex64)

                # Noise score: lower means signal. squelch is the cutoff;
                # above = noise, below = signal.
                score = detect_signal(iq)

                if score >= self.squelch:
                    # Quiet. Keep the tail of recent IQ for onset replay
                    # then drop older frames. Nothing enters the queue.
                    self._set_state("listening")
                    prebuf.append(iq)
                    prebuf_count += len(iq)
                    while prebuf_count > prebuf_samples and len(prebuf) > 1:
                        prebuf_count -= len(prebuf.popleft())
                    continue

                # --- Signal onset ---------------------------------------
                self._set_state("capturing")
                accumulated: list[np.ndarray] = list(prebuf) + [iq]
                acc_count = sum(len(a) for a in accumulated)
                prebuf = deque()
                prebuf_count = 0

                aborted = False
                while acc_count < chunk_samples and not self._stop.is_set():
                    if self._new_freq is not None:
                        aborted = True
                        break
                    more = sdr.read_samples(self.CHUNK_IQ).astype(np.complex64)
                    accumulated.append(more)
                    acc_count += len(more)

                if aborted or self._stop.is_set():
                    self._set_state("listening")
                    continue

                full_iq = np.concatenate(accumulated)[:chunk_samples]
                wbfm = is_wbfm(self.freq_mhz)
                audio = process(
                    full_iq.astype(np.complex128),
                    self.sample_rate,
                    self.audio_rate,
                    wbfm,
                )

                with self._lock:
                    if len(self._queue) >= self._max_chunks:
                        self._queue.popleft()
                        self._dropped += 1
                    self._queue.append(audio)
                self._set_state("listening")
        finally:
            sdr.close()


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class StatusPanel(Static):
    """Left-hand panel showing live monitoring status."""

    # SDR / capture status (always active in live mode)
    sdr_phase = reactive("Idle")    # Idle | Listening | Capturing | Scanning | Stopped
    # LLM analysis status (independent of capture)
    llm_phase = reactive("Idle")    # Idle | Analyzing

    cycle = reactive(0)
    freq = reactive(0.0)
    mode_label = reactive("--")
    watch_text = reactive("")
    model_name = reactive("airgrep-asr-gemma-4-e4b")
    duration = reactive(DEFAULT_DURATION)
    last_llm = reactive("")
    is_demo = reactive(False)
    # Scan mode
    scan_start = reactive(0.0)
    scan_end = reactive(0.0)
    scan_step = reactive(0.0)
    scan_current = reactive(0.0)

    # Monotonic timestamp of the current capture's onset (0 = not capturing).
    capture_start_ts: float = 0.0

    def on_mount(self) -> None:
        # 5 Hz tick keeps the live capture timer smooth.
        self.set_interval(0.2, self.refresh)

    def render(self) -> str:
        # ── SDR indicator (always-on capture) ──
        sdr_icons = {
            "Idle": "[dim]\u25cb SDR idle[/]",
            "Listening": "[bold green]\u25cf LISTENING[/]",
            "Scanning": "[bold green]\u25cf SCANNING[/]",
            "Loading": "[bold yellow]\u25cf LOADING[/]",
            "Stopped": "[dim]\u25cb Stopped[/]",
        }
        if self.sdr_phase == "Capturing":
            if self.capture_start_ts > 0:
                elapsed = time.monotonic() - self.capture_start_ts
                # Clamp display so a stale timer can't show > duration.
                elapsed = min(elapsed, self.duration)
                sdr_str = (
                    f"[bold red]\u25cf CAPTURING[/]"
                    f"  [red]{elapsed:4.1f}s / {self.duration:.0f}s[/]"
                )
            else:
                sdr_str = "[bold red]\u25cf CAPTURING[/]"
        else:
            sdr_str = sdr_icons.get(self.sdr_phase, f"[dim]{self.sdr_phase}[/]")

        # ── LLM indicator ──
        if self.llm_phase == "Analyzing":
            llm_str = "[bold magenta]\u25cf ANALYZING[/]"
        else:
            llm_str = "[dim]\u25cb LLM idle[/]"

        demo_tag = "  [bold yellow]DEMO[/]" if self.is_demo else ""

        last = self.last_llm
        if len(last) > 200:
            last = last[:200] + "..."

        # Frequency display adapts to scan vs single mode
        if self.scan_start > 0:
            freq_lines = (
                f"  Scan       [bold]{self.scan_start}-{self.scan_end} MHz[/]\n"
                f"  Step       {self.scan_step} kHz  ({self.mode_label})"
            )
            if self.scan_current > 0 and self.sdr_phase == "Scanning":
                freq_lines += f"\n  Sweeping   [bold cyan]{self.scan_current:.3f} MHz[/]"
            elif self.scan_current > 0:
                freq_lines += f"\n  Locked     [bold cyan]{self.scan_current:.3f} MHz[/]"
            if self.freq > 0 and self.llm_phase == "Analyzing":
                freq_lines += f"\n  Locked     [bold green]{self.freq} MHz[/]"
        else:
            freq_lines = f"  Frequency  [bold]{self.freq} MHz[/]  ({self.mode_label})"

        return (
            f"  {sdr_str}{demo_tag}\n"
            f"  {llm_str}\n"
            f"\n"
            f"  Cycle      [bold]{self.cycle}[/]\n"
            f"{freq_lines}\n"
            f"  Duration   {self.duration:.0f}s chunks\n"
            f"  Model      {self.model_name}\n"
            f"\n"
            f"[bold cyan]WATCHING FOR[/]\n"
            f"  [italic]{self.watch_text}[/]\n"
            f"\n"
            f"[bold]Last result[/]\n"
            f"  [dim]{last or '\u2014'}[/]"
        )


class StatsBar(Static):
    """Bottom stats strip above the footer."""

    cycle_count = reactive(0)
    alert_count = reactive(0)
    start_time: float = 0.0

    def on_mount(self) -> None:
        self.start_time = time.monotonic()
        self.set_interval(1.0, self.refresh)

    def render(self) -> str:
        elapsed = int(time.monotonic() - self.start_time)
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        if h:
            uptime = f"{h}h {m:02d}m {s:02d}s"
        elif m:
            uptime = f"{m}m {s:02d}s"
        else:
            uptime = f"{s}s"
        return (
            f"  Cycles: [bold]{self.cycle_count}[/]  |  "
            f"Alerts: [bold]{self.alert_count}[/]  |  "
            f"Uptime: [bold]{uptime}[/]"
        )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class AirGrepApp(App):
    """AirGrep — grep for the airwaves."""

    TITLE = "AirGrep"
    SUB_TITLE = ""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-area {
        height: 1fr;
    }
    #left-panel {
        width: 44;
        min-width: 34;
        border-right: solid $accent;
    }
    #status-panel {
        padding: 1 2;
        height: 1fr;
    }
    #input-section {
        height: auto;
        padding: 0 2 1 2;
    }
    .input-label {
        margin-top: 1;
        color: $text-muted;
    }
    #freq-input, #watch-input {
        margin-bottom: 0;
    }
    #feed-panel {
        width: 1fr;
        padding: 1 2;
    }
    #feed-title {
        text-style: bold;
        margin-bottom: 1;
        color: $text-muted;
    }
    #feed-log {
        height: 1fr;
    }
    #log-overlay {
        display: none;
        width: 1fr;
        padding: 1 2;
        border-left: solid $warning;
    }
    #log-overlay.visible {
        display: block;
    }
    #log-title {
        text-style: bold underline;
        margin-bottom: 1;
        color: $warning;
    }
    #log-viewer {
        height: 1fr;
    }
    #stats-bar {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "clear_alerts", "Clear"),
        Binding("l", "toggle_log", "Log"),
        Binding("p", "play_alert", "Play"),
    ]

    def __init__(
        self,
        freq_mhz: float = DEFAULT_FREQ,
        duration: float = DEFAULT_DURATION,
        watch: str = DEFAULT_WATCH,
        model_path: str = DEFAULT_MODEL_PATH,
        gain: str = "auto",
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        audio_rate: int = DEFAULT_AUDIO_RATE,
        once: bool = False,
        log_path: str = "alerts.log",
        demo_files: list[str] | None = None,
        squelch: float = DEFAULT_SQUELCH,
        force_cpu: bool = False,
        max_gpu_memory: str | None = None,
    ) -> None:
        super().__init__()
        self.freq_mhz = freq_mhz
        self.duration = duration
        self.watch = watch
        self.model_path = model_path
        self.gain = gain
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.once = once
        self.log_path = Path(log_path)
        self.demo_files = demo_files or []
        self.is_demo = bool(self.demo_files)
        self.squelch = squelch
        self.force_cpu = force_cpu
        self.max_gpu_memory = max_gpu_memory

        self.scan_range: tuple[float, float] | None = None
        self._stop_event = Event()
        self._alert_count = 0
        self._cycle_count = 0

    # -- Layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            with Vertical(id="left-panel"):
                yield StatusPanel(id="status-panel")
                with Vertical(id="input-section"):
                    yield Label("Frequency (MHz)", classes="input-label")
                    yield Input(
                        id="freq-input",
                        value=str(self.freq_mhz),
                        placeholder="e.g. 102.7 or 144-148",
                    )
                    yield Label("Watch for", classes="input-label")
                    yield Input(
                        id="watch-input",
                        value=self.watch,
                        placeholder="e.g. emergency alerts, severe weather",
                    )
            with Vertical(id="feed-panel"):
                yield Label("Transcripts", id="feed-title")
                yield RichLog(id="feed-log", highlight=True, markup=True, wrap=True)
            with Vertical(id="log-overlay"):
                yield Label(
                    "Alert Log  (L to close, \u2191/\u2193 to navigate, P to play)",
                    id="log-title",
                )
                yield Label("", id="log-summary")
                yield ListView(id="log-viewer")
        yield StatsBar(id="stats-bar")
        yield Footer()

    def on_mount(self) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        status.freq = self.freq_mhz
        mode = "WBFM" if is_wbfm(self.freq_mhz) else "NBFM"
        status.mode_label = mode
        status.watch_text = self.watch
        status.model_name = "airgrep-asr-gemma-4-e4b"
        status.duration = self.duration
        status.is_demo = self.is_demo

        self.run_monitor()

    # -- Actions -----------------------------------------------------------

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Hide 'Play' from the footer unless the log overlay is visible."""
        if action == "play_alert":
            try:
                overlay = self.query_one("#log-overlay")
            except Exception:
                return None
            return True if overlay.has_class("visible") else None
        return True

    def action_clear_alerts(self) -> None:
        log = self.query_one("#feed-log", RichLog)
        log.clear()

    def action_toggle_log(self) -> None:
        overlay = self.query_one("#log-overlay")
        if overlay.has_class("visible"):
            overlay.remove_class("visible")
            self.refresh_bindings()
            return
        # Populate and show
        viewer = self.query_one("#log-viewer", ListView)
        summary = self.query_one("#log-summary", Label)
        viewer.clear()
        if self.log_path.exists():
            lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
            summary.update(f"[bold]{self.log_path} — {len(lines)} entries[/]")
            recent = lines[-50:]
            for line in recent:
                entry: dict | None = None
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    display = line
                else:
                    urg = entry.get("urgency", "?").upper()
                    freq = entry.get("freq_mhz")
                    freq_str = f" {freq} MHz" if freq else ""
                    audio_mark = " \U0001f509" if entry.get("audio") else ""
                    display = (
                        f"[{urg}] {entry.get('time', '?')}{freq_str}{audio_mark}  "
                        f"{entry.get('message', '')}"
                    )
                item = ListItem(Label(display, markup=True))
                # Attach the parsed entry so action_play_alert can find it.
                item.alert_entry = entry  # type: ignore[attr-defined]
                viewer.append(item)
            if len(lines) > 50:
                viewer.append(
                    ListItem(Label(
                        f"[dim]... and {len(lines) - 50} earlier entries[/]",
                        markup=True,
                    ))
                )
        else:
            summary.update("[dim]No log file yet.[/]")
        overlay.add_class("visible")
        self.refresh_bindings()

    def action_play_alert(self) -> None:
        """Play the audio clip associated with the highlighted log entry."""
        overlay = self.query_one("#log-overlay")
        if not overlay.has_class("visible"):
            return

        viewer = self.query_one("#log-viewer", ListView)
        item = viewer.highlighted_child
        entry = getattr(item, "alert_entry", None) if item else None

        if not entry:
            self._write_to_feed("[dim]\u2192 No alert selected.[/]")
            return

        audio_rel = entry.get("audio")
        if not audio_rel:
            self._write_to_feed("[dim]\u2192 No audio file recorded for this alert.[/]")
            return

        audio_path = self.log_path.parent / audio_rel
        if not audio_path.exists():
            self._write_to_feed(
                f"[yellow]\u2192 Audio file missing: {audio_rel}[/]"
            )
            return

        import os
        import subprocess
        try:
            if sys.platform == "win32":
                os.startfile(str(audio_path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(audio_path)])
            else:
                subprocess.Popen(["xdg-open", str(audio_path)])
            self._write_to_feed(f"[dim]\u2192 Playing {audio_path.name}[/]")
        except Exception as e:
            self._write_to_feed(f"[bold red]Play failed: {e}[/]")

    # -- Input handlers ----------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in input fields."""
        if event.input.id == "freq-input":
            self._apply_freq_change(event.value.strip())
        elif event.input.id == "watch-input":
            self._apply_watch_change(event.value.strip())
        # Unfocus so keybindings work again
        self.set_focus(None)

    def _apply_freq_change(self, raw: str) -> None:
        status = self.query_one("#status-panel", StatusPanel)

        # Check for range: "144-148" or "144.0 - 148.0"
        if "-" in raw and not raw.startswith("-"):
            parts = raw.split("-", 1)
            try:
                start = float(parts[0].strip())
                end = float(parts[1].strip())
            except ValueError:
                self._write_to_feed("[bold red]Invalid range: " + raw + "[/]")
                return
            if start >= end or start <= 0:
                self._write_to_feed("[bold red]Invalid range (start must be < end)[/]")
                return
            step = infer_step_khz(start, end)
            mode = "WBFM" if is_wbfm(start) else "NBFM"
            self.scan_range = (start, end)
            self.freq_mhz = start
            status.scan_start = start
            status.scan_end = end
            status.scan_step = step
            status.scan_current = 0.0
            status.freq = 0.0
            status.mode_label = mode
            n_steps = int((end - start) / (step / 1000.0)) + 1
            self._write_to_feed(
                f"[dim]\u2192 Scan mode: {start}-{end} MHz, "
                f"{step} kHz steps ({n_steps} ch)[/]"
            )
            return

        # Single frequency
        try:
            new_freq = float(raw)
        except ValueError:
            self._write_to_feed("[bold red]Invalid frequency: " + raw + "[/]")
            return
        if new_freq <= 0:
            self._write_to_feed("[bold red]Frequency must be positive[/]")
            return
        self.scan_range = None
        self.freq_mhz = new_freq
        mode = "WBFM" if is_wbfm(new_freq) else "NBFM"
        status.freq = new_freq
        status.mode_label = mode
        status.scan_start = 0.0
        status.scan_end = 0.0
        status.scan_current = 0.0
        self._write_to_feed(
            f"[dim]\u2192 Retuned to {new_freq} MHz ({mode})[/]"
        )

    def _apply_watch_change(self, new_watch: str) -> None:
        if not new_watch:
            self._write_to_feed("[bold red]Watch text cannot be empty[/]")
            return
        self.watch = new_watch
        status = self.query_one("#status-panel", StatusPanel)
        status.watch_text = new_watch
        self._write_to_feed(
            f"[dim]\u2192 Now watching for: {new_watch}[/]"
        )

    # -- Monitor loop (runs in a worker thread) ----------------------------

    @work(thread=True)
    def run_monitor(self) -> None:
        """Main monitoring loop — runs in a background thread."""
        if self.is_demo:
            self._run_demo_loop()
        else:
            self._run_live_loop()

    def _run_demo_loop(self) -> None:
        """Cycle through demo WAV files."""
        file_index = 0
        while not self._stop_event.is_set():
            wav_path = self.demo_files[file_index % len(self.demo_files)]
            file_index += 1

            self._cycle_count += 1
            self._update_cycle(self._cycle_count)
            self._update_sdr_phase("Loading")

            if not Path(wav_path).exists():
                self._log_feed(f"[bold red]File not found: {wav_path}[/]")
                continue

            self._update_llm_phase("Analyzing")
            self._run_llm_cycle(wav_path)
            self._update_llm_phase("Idle")

            self._update_sdr_phase("Idle")

            if self.once or file_index >= len(self.demo_files):
                break

            # Brief pause between demo cycles
            if not self._stop_event.wait(2.0):
                continue

        self._update_sdr_phase("Stopped")

    def _run_live_loop(self) -> None:
        """Live SDR capture loop — handles both single-freq and scan mode.

        Single-freq mode uses ContinuousCapture with a drain queue: IQ is
        captured continuously, processed into chunks, and queued.  The LLM
        consumes chunks FIFO — every chunk is analyzed before it is discarded.
        Queue is capped at ~5 min (10 chunks) to bound memory.

        Scan mode stops the continuous capture and does its own sweep + capture.
        """
        import tempfile

        capture: ContinuousCapture | None = None
        was_scanning = False

        try:
            while not self._stop_event.is_set():
                self._cycle_count += 1
                self._update_cycle(self._cycle_count)
                is_scanning = self.scan_range is not None

                if is_scanning:
                    # ── Scan mode ──────────────────────────────────────
                    if capture is not None:
                        capture.shutdown()
                        capture = None

                    start, end = self.scan_range
                    self._update_sdr_phase("Scanning")

                    try:
                        result = self._do_scan_cycle(start, end)
                    except Exception as e:
                        self._log_feed(f"[bold red]Scan error: {e}[/]")
                        self._update_sdr_phase("Idle")
                        if self.once:
                            break
                        self._stop_event.wait(5.0)
                        was_scanning = True
                        continue

                    if result is None:
                        self._update_sdr_phase("Idle")
                        if self.once:
                            break
                        was_scanning = True
                        continue

                    wav_path, found_freq = result
                    self.freq_mhz = found_freq
                    was_scanning = True

                else:
                    # ── Single frequency — continuous capture ──────────
                    if capture is None or was_scanning:
                        if capture is not None:
                            capture.shutdown()
                        capture = ContinuousCapture(
                            freq_mhz=self.freq_mhz,
                            sample_rate=self.sample_rate,
                            audio_rate=self.audio_rate,
                            gain=self.gain,
                            chunk_duration=self.duration,
                            squelch=self.squelch,
                            on_state_change=self._on_capture_state,
                        )
                        capture.start()
                        was_scanning = False
                        # Initial visible state — callback drives transitions thereafter.
                        self._update_sdr_phase("Listening")

                    # Retune if user changed frequency
                    if capture.freq_mhz != self.freq_mhz:
                        capture.change_freq(self.freq_mhz)

                    # Wait for next signal-gated chunk. Idle may be
                    # arbitrarily long, so poll for user-initiated mode
                    # or frequency changes so we don't strand the UI.
                    switched_to_scan = False
                    while not capture.has_chunk():
                        if self._stop_event.wait(0.5):
                            return
                        if self.scan_range is not None:
                            switched_to_scan = True
                            break
                        if capture.freq_mhz != self.freq_mhz:
                            capture.change_freq(self.freq_mhz)

                    if switched_to_scan:
                        # Loop back so the scan branch takes over.
                        continue

                    audio = capture.next_chunk()
                    depth = capture.queue_depth()
                    dropped = capture.dropped_count

                    # Write chunk to temp WAV
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    wav_path = tmp.name
                    tmp.close()
                    sf.write(wav_path, audio, self.audio_rate, subtype="FLOAT")

                    if dropped > 0:
                        self._log_feed(
                            f"[yellow]  \u26a0 {dropped} chunk(s) dropped (LLM can't keep up)[/]"
                        )

                # ── Analyze with LLM (capture continues in background) ──
                self._update_llm_phase("Analyzing")
                self._run_llm_cycle(wav_path)
                self._update_llm_phase("Idle")
                Path(wav_path).unlink(missing_ok=True)

                if self.once:
                    break
        finally:
            if capture is not None:
                capture.shutdown()

        self._update_sdr_phase("Stopped")

    def _do_scan_cycle(
        self, start_mhz: float, end_mhz: float
    ) -> tuple[str, float] | None:
        """Wideband FFT sweep + capture on strongest signal.

        Uses fft_scan() to detect active channels across the entire range
        in a single pass (~75 ms for 4 MHz) instead of per-channel
        retuning (~10 s).  Then captures 29s of audio on the strongest hit.

        Returns (wav_path, found_freq_mhz) or None if no signal detected.
        """
        import tempfile
        from rtlsdr import RtlSdr
        from capture import fft_scan, capture_iq, process

        step_khz = infer_step_khz(start_mhz, end_mhz)
        wbfm = is_wbfm(start_mhz)

        sdr = RtlSdr()
        try:
            sdr.sample_rate = self.sample_rate
            sdr.gain = "auto" if self.gain == "auto" else float(self.gain)

            # ── Wideband FFT sweep ────────────────────────────────
            def _on_chunk(lo_mhz: float, hi_mhz: float) -> None:
                self._update_scan_current(lo_mhz)

            hits = fft_scan(
                sdr,
                start_mhz, end_mhz,
                sample_rate=self.sample_rate,
                channel_khz=step_khz,
                on_chunk=_on_chunk,
            )

            if not hits:
                return None

            # Lock onto strongest signal
            found_freq = hits[0][0]
            self._update_scan_current(found_freq)

            # ── Full capture on found frequency ───────────────────
            sdr.center_freq = found_freq * 1e6
            sdr.read_samples(4096)  # flush after retune
            num_samples = int(self.duration * self.sample_rate)
            self.call_from_thread(self._begin_capture_indicator)
            iq = capture_iq(sdr, num_samples)
        finally:
            sdr.close()

        audio = process(iq, self.sample_rate, self.audio_rate, wbfm)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        sf.write(tmp_path, audio, self.audio_rate, subtype="FLOAT")

        return tmp_path, found_freq

    def _run_llm_cycle(self, wav_path: str) -> None:
        """Two-pass audio analysis: transcribe then evaluate."""

        def _alert_fn(message: str, urgency: str) -> str:
            return self._fire_alert(message, urgency, wav_path=wav_path)

        def _on_status(text: str) -> None:
            self._update_last_llm(text)
            ts = datetime.now().strftime("%H:%M:%S")
            # Stream transcripts into the feed panel
            if text.startswith("[Transcript]"):
                transcript = text[len("[Transcript]"):].strip()
                self._log_feed(
                    f"[dim]{ts}[/]  {transcript}"
                )
            elif text == "[no signal]":
                self._log_feed(f"[dim]{ts}  No signal[/]")
            elif "no speech" in text.lower():
                self._log_feed(f"[dim]{ts}  (silence)[/]")

        analyze_audio(
            model_path=self.model_path,
            wav_path=wav_path,
            freq_mhz=self.freq_mhz,
            duration=self.duration,
            watch=self.watch,
            alert_fn=_alert_fn,
            on_status=_on_status,
            force_cpu=self.force_cpu,
            max_gpu_memory=self.max_gpu_memory,
        )

    # -- Helpers to talk to the UI from worker thread ----------------------

    def _fire_alert(self, message: str, urgency: str, wav_path: str | None = None) -> str:
        """Record an alert — called from the worker thread."""
        self._alert_count += 1
        urgency = urgency.lower().strip()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

        urgency_styles = {
            "low": "yellow",
            "medium": "bold yellow",
            "high": "bold red",
        }
        style = urgency_styles.get(urgency, "yellow")

        freq_str = f"{self.freq_mhz:.3f} MHz" if self.freq_mhz else ""
        short_ts = datetime.now().strftime("%H:%M:%S")
        self._log_feed(
            f"[{style}][{urgency.upper()}] {short_ts} | {freq_str}[/]\n"
            f"[{style}]  {message}[/]"
        )

        # Beep
        self.call_from_thread(self.bell)

        # Update stats bar
        self.call_from_thread(self._sync_stats)

        # Save audio clip associated with this alert
        audio_rel: str | None = None
        if wav_path and Path(wav_path).exists():
            audio_dir = self.log_path.parent / "alerts_audio"
            audio_dir.mkdir(exist_ok=True)
            dest = audio_dir / f"alert_{ts_file}_{urgency}.wav"
            import shutil
            shutil.copy2(wav_path, dest)
            audio_rel = str(dest.relative_to(self.log_path.parent))
            self._log_feed(f"[dim]  Audio saved: {dest.name}[/]")

        # Write to log file
        entry = {
            "time": ts,
            "urgency": urgency,
            "freq_mhz": self.freq_mhz,
            "message": message,
            "audio": audio_rel,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return f"Alert delivered to user ({urgency})."

    def _log_feed(self, markup: str) -> None:
        """Write a line to the transcript feed panel (thread-safe)."""
        self.call_from_thread(self._write_to_feed, markup)

    def _write_to_feed(self, markup: str) -> None:
        log = self.query_one("#feed-log", RichLog)
        log.write(markup)

    def _update_cycle(self, cycle: int) -> None:
        self.call_from_thread(self._sync_cycle, cycle)

    def _update_sdr_phase(self, phase: str) -> None:
        """Update the SDR/capture indicator (Idle|Listening|Scanning|Stopped)."""
        self.call_from_thread(self._sync_sdr_phase, phase)

    def _update_llm_phase(self, phase: str) -> None:
        """Update the LLM indicator (Idle|Analyzing)."""
        self.call_from_thread(self._sync_llm_phase, phase)

    def _update_last_llm(self, text: str) -> None:
        self.call_from_thread(self._sync_last_llm, text)

    def _sync_cycle(self, cycle: int) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        status.cycle = cycle
        stats = self.query_one("#stats-bar", StatsBar)
        stats.cycle_count = cycle

    def _sync_sdr_phase(self, phase: str) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        status.sdr_phase = phase
        # Reset the capture timer when leaving Capturing.
        if phase != "Capturing":
            status.capture_start_ts = 0.0

    def _on_capture_state(self, state: str) -> None:
        """Bridge ContinuousCapture state changes (capture thread) to the UI."""
        if state == "capturing":
            self.call_from_thread(self._begin_capture_indicator)
        elif state == "listening":
            self._update_sdr_phase("Listening")

    def _begin_capture_indicator(self) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        status.capture_start_ts = time.monotonic()
        status.sdr_phase = "Capturing"
        ts = datetime.now().strftime("%H:%M:%S")
        self._write_to_feed(f"[dim]{ts}[/]  [red]\u25cf signal detected[/]")

    def _sync_llm_phase(self, phase: str) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        status.llm_phase = phase

    def _sync_last_llm(self, text: str) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        status.last_llm = text

    def _update_scan_current(self, freq: float) -> None:
        self.call_from_thread(self._sync_scan_current, freq)

    def _sync_scan_current(self, freq: float) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        status.scan_current = freq

    def _sync_stats(self) -> None:
        stats = self.query_one("#stats-bar", StatsBar)
        stats.alert_count = self._alert_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AirGrep — grep for the airwaves. Powered by Gemma 4 E4B.",
    )
    parser.add_argument(
        "-f", "--freq", type=float, default=DEFAULT_FREQ,
        help=f"Center frequency in MHz (default: {DEFAULT_FREQ})",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=DEFAULT_DURATION,
        help=f"Capture duration per cycle in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--watch", type=str, default=DEFAULT_WATCH,
        help="What to watch for — natural language description",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle then exit",
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
        "-s", "--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE,
        help=f"SDR sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--audio-rate", type=int, default=DEFAULT_AUDIO_RATE,
        help=f"Output audio sample rate in Hz (default: {DEFAULT_AUDIO_RATE})",
    )
    parser.add_argument(
        "--demo", type=str, action="append", default=None, dest="demo_files",
        help="WAV file(s) for demo mode — no SDR dongle needed (can repeat)",
    )
    parser.add_argument(
        "--squelch", type=float, default=DEFAULT_SQUELCH,
        help=f"Scan squelch threshold — lower is more sensitive (default: {DEFAULT_SQUELCH})",
    )
    parser.add_argument(
        "--cpu", action="store_true", dest="force_cpu",
        help="Force CPU inference (no GPU). Slow — use only for testing.",
    )
    parser.add_argument(
        "--max-gpu-memory", type=str, default=None, dest="max_gpu_memory",
        help="Cap GPU VRAM usage (e.g. '7GiB'). Spills layers to CPU RAM "
             "to fit Gemma 4 E4B on small GPUs. See INSTALL.md.",
    )
    return parser.parse_args(argv)


def _prompt_for_demo_files() -> list[str]:
    """Interactive prompt when --demo is passed with no files."""
    print("\n  AirGrep — Demo Mode Setup")
    print("  " + "=" * 40)
    print("  No WAV files specified. Enter paths to WAV files for demo mode.")
    print("  (One per line. Empty line to finish.)\n")
    files: list[str] = []
    while True:
        try:
            path = input("  WAV file path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            break
        if not path:
            break
        p = Path(path)
        if not p.exists():
            print(f"  [!] File not found: {path}")
            continue
        if not p.suffix.lower() == ".wav":
            print(f"  [!] Not a .wav file: {path}")
            continue
        files.append(str(p.resolve()))
        print(f"  [+] Added: {p.name}")
    return files


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Handle demo mode
    demo_files = args.demo_files
    if demo_files is not None:
        valid = []
        for f in demo_files:
            p = Path(f)
            if not p.exists():
                print(f"  [!] File not found: {f}")
            else:
                valid.append(str(p.resolve()))
        if not valid:
            print("  No valid files. Exiting.")
            sys.exit(1)
        demo_files = valid

    # If no --demo and no --freq override, try to detect SDR; offer demo mode
    if demo_files is None:
        try:
            from rtlsdr import RtlSdr
            sdr = RtlSdr()
            sdr.close()
        except Exception:
            print("\n  No RTL-SDR dongle detected.")
            print("  Would you like to run in demo mode with WAV files? (y/n)")
            try:
                answer = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            if answer in ("y", "yes"):
                demo_files = _prompt_for_demo_files()
                if not demo_files:
                    print("  No files provided. Exiting.")
                    sys.exit(0)
            else:
                print("  Exiting. Connect an RTL-SDR dongle and try again.")
                sys.exit(0)

    app = AirGrepApp(
        freq_mhz=args.freq,
        duration=args.duration,
        watch=args.watch,
        model_path=args.model_path,
        gain=args.gain,
        sample_rate=args.sample_rate,
        audio_rate=args.audio_rate,
        once=args.once,
        log_path=args.log,
        demo_files=demo_files,
        squelch=args.squelch,
        force_cpu=args.force_cpu,
        max_gpu_memory=args.max_gpu_memory,
    )
    app.run()


if __name__ == "__main__":
    main()
