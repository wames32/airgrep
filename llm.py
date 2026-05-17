"""Two-pass LLM analysis using a single fine-tuned Gemma 4 E4B model.

Pass 1 — Transcription (Gemma 4 E4B, audio-LoRA fine-tuned):
    The model transcribes the audio directly.  The fine-tune was trained
    specifically on NBFM radio-degraded speech, so it handles callsigns
    and weak signals that vanilla Gemma 4 misses.

Pass 2 — Evaluation (same model, text-only, tool-calling):
    A fresh conversation receives the transcript and the watch criteria.
    If the transcript matches, the model calls ``alert_user``.

Both passes share a single loaded model instance to minimize memory
footprint.  Both are stateless (fresh conversation each time) with
retry logic for transient inference failures.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


# ── Debug logging (persists to disk so it survives the TUI closing) ───────

_LOG_PATH = Path(__file__).parent / "llm_debug.log"
_log = logging.getLogger("sdr.llm")
if not _log.handlers:
    _log.setLevel(logging.DEBUG)
    _fh = logging.FileHandler(str(_LOG_PATH), mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_fh)
    _log.propagate = False
    _log.info("=" * 60)
    _log.info("llm.py loaded — session start")


# ── Shared model (lazy-loaded singleton, used for both passes) ─────────────

_model = None
_processor = None


# ── Evaluator conversation history ─────────────────────────────────────────
# The evaluation pass (pass 2) carries a rolling window of prior transcript/
# response pairs so the model has conversational context across cycles on
# the same frequency. Capped at EVAL_HISTORY_MAX messages (= 4 cycles of
# user+assistant). Any frequency change clears it. The ASR pass (pass 1)
# is intentionally stateless — we want each transcription judged only on
# the audio in front of it.

EVAL_HISTORY_MAX = 8
_eval_history: "deque[dict]" = deque(maxlen=EVAL_HISTORY_MAX)
_eval_history_freq: float | None = None


def reset_eval_history(reason: str = "") -> None:
    """Clear the evaluator's rolling conversation history."""
    global _eval_history_freq
    _eval_history.clear()
    _eval_history_freq = None
    if reason:
        _log.info("Eval history cleared: %s", reason)


def _get_model(
    model_path: str,
    force_cpu: bool = False,
    max_gpu_memory: str | None = None,
):
    """Load the fine-tuned Gemma 4 E4B model + processor on first call, then cache.

    Parameters
    ----------
    model_path : str
        HF repo id or local path.
    force_cpu : bool
        If True, load entirely on CPU in float32.  Inference will be
        minutes per clip — use only for environments with no GPU.
    max_gpu_memory : str, optional
        Cap on VRAM (e.g. ``"7GiB"``) for accelerate's device_map.  When
        set, layers that don't fit spill to CPU RAM (slower, but lets
        the ~8B-param E4B fit on small GPUs; see INSTALL.md).  Ignored
        when ``force_cpu=True``.
    """
    global _model, _processor
    if _model is None:
        # Decide device + dtype.  bf16 on CPU is unreliable on older cores
        # (silent upcast to fp32, or kernel crashes) — safest default is fp32.
        cuda_available = torch.cuda.is_available() and not force_cpu
        if cuda_available:
            dtype = torch.bfloat16
            if max_gpu_memory:
                device_map = "auto"
                max_memory = {0: max_gpu_memory, "cpu": "64GiB"}
                _log.info("Loading with max_gpu_memory=%s (CPU offload enabled)",
                          max_gpu_memory)
            else:
                device_map = "auto"
                max_memory = None
        else:
            dtype = torch.float32
            device_map = {"": "cpu"}
            max_memory = None
            if force_cpu:
                _log.warning("force_cpu=True: loading on CPU in float32. Expect minutes/clip.")
            else:
                _log.warning("CUDA unavailable: falling back to CPU + float32. Expect minutes/clip.")

        _log.info("Loading model from %s (dtype=%s, device_map=%s)",
                  model_path, dtype, device_map)
        load_kwargs = dict(dtype=dtype, device_map=device_map)
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

        # Fallback path: if the HF repo is incomplete (e.g. still uploading)
        # or lacks processor configs, fall back to a local merged bf16
        # checkpoint. Set AIRGREP_LOCAL_MODEL_PATH to override the default.
        import os
        local_fallback = os.environ.get(
            "AIRGREP_LOCAL_MODEL_PATH",
            r"C:/Users/wafor/OneDrive/Documentos/sdr_py/gemma4/finetune/runs/merged_bf16",
        )
        local_fallback_exists = Path(local_fallback).is_dir()

        def _try_load(path: str):
            proc = AutoProcessor.from_pretrained(path)
            mdl = AutoModelForImageTextToText.from_pretrained(path, **load_kwargs)
            return proc, mdl

        try:
            _processor, _model = _try_load(model_path)
        except (ValueError, OSError, EnvironmentError) as e:
            if local_fallback_exists and str(Path(local_fallback).resolve()) != str(Path(model_path).resolve() if Path(model_path).exists() else Path(model_path)):
                _log.warning(
                    "Failed to load from %s (%s). Falling back to local "
                    "checkpoint at %s.", model_path, e, local_fallback,
                )
                _processor, _model = _try_load(local_fallback)
            else:
                _log.warning(
                    "Failed to load processor from %s (%s) and no local "
                    "fallback available. Retrying with base "
                    "google/gemma-4-E4B-it processor.", model_path, e,
                )
                _processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
                _model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        _model.eval()
        _log.info("Model loaded. Device=%s dtype=%s", _model.device, _model.dtype)
    return _model, _processor


# ── Constants ───────────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
ASR_AUDIO_MAX_S = 29  # Gemma 4 audio cap
ASR_SAMPLE_RATE = 16_000


# ── Tool schema ─────────────────────────────────────────────────────────────

ALERT_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "alert_user",
        "description": "Alert the user about detected content in the radio transmission.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Description of what was detected and why it is relevant.",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Importance level.",
                },
            },
            "required": ["message", "urgency"],
        },
    },
}

_ALERT_PARAM_NAMES = list(
    ALERT_USER_TOOL["function"]["parameters"]["properties"].keys()
)


# ── Prompt builders ─────────────────────────────────────────────────────────


def build_asr_prompt() -> str:
    """System prompt for pass 1: ASR transcription."""
    return (
        "You are transcribing audio from a radio receiver. "
        "The audio may contain amateur (ham) radio transmissions with callsigns. "
        "Transcribe exactly what you hear. Output only the transcription."
    )


def build_evaluation_prompt(freq_mhz: float, mode: str, watch: str) -> str:
    """System prompt for pass 2: evaluate transcript against watch criteria.

    Tool specification is embedded directly in the prompt rather than being
    passed via ``apply_chat_template(tools=...)`` — Gemma 4's chat template
    handling of the ``tools=`` parameter is unstable, and the model natively
    emits ``call:name{...}`` format which we parse ourselves.
    """
    return (
        f"You are an automated radio monitor evaluating a transcription "
        f"from {freq_mhz} MHz ({mode} FM).\n\n"
        f"You will receive a verbatim transcription of a radio capture. "
        f"Evaluate whether it matches the following watch criteria:\n"
        f"{watch}\n\n"
        f"If the transcription contains content matching the watch criteria, "
        f"emit EXACTLY this format (and nothing else):\n"
        f"    call:alert_user{{message: <short description of what matched>, urgency: <low|medium|high>}}\n\n"
        f"If nothing matches, respond with a brief 1-sentence summary and no call.\n\n"
        f"IMPORTANT: Base your judgment ONLY on what appears in the "
        f"transcription text. Do not infer or assume content that is not "
        f"explicitly stated. If the transcription only says [unintelligible] "
        f"or is empty, that is NOT a match for any watch criteria."
    )


# ── Audio loading ───────────────────────────────────────────────────────────


def _load_audio_16k(path: str) -> np.ndarray:
    """Load a WAV, downmix to mono, resample to 16 kHz, cap at 29s."""
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != ASR_SAMPLE_RATE:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)
    x = x[: ASR_AUDIO_MAX_S * ASR_SAMPLE_RATE]
    return x


# ── Tool call parsing ────────────────────────────────────────────────────────


def _parse_tool_calls(
    text: str,
    known_params: list[str] | None = None,
) -> list[dict]:
    """Parse tool calls from Gemma 4 output.

    Handles two observed formats:
    1. Strict JSON: ``<tool_call>{...}</tool_call>``
    2. Loose k/v:   ``call:func_name{key: value, key: value}``
    """
    calls: list[dict] = []

    # Format 1 — strict JSON inside <tool_call> tags
    for m in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            parsed = json.loads(m.strip())
            if isinstance(parsed, dict):
                calls.append(parsed)
        except json.JSONDecodeError:
            pass
    if calls:
        return calls

    # Format 2 — Gemma's loose "call:name{...}" format
    for m in re.finditer(r"call\s*:\s*(\w+)\s*\{(.*?)\}", text, re.DOTALL):
        name = m.group(1)
        body = m.group(2)
        args = _parse_loose_kv(body, known_params or [])
        if name and args:
            calls.append({"name": name, "arguments": args})

    return calls


def _parse_loose_kv(body: str, known_keys: list[str]) -> dict:
    """Parse ``key: value, key: value`` where values may contain commas.

    Uses ``known_keys`` as split anchors so that a value containing commas
    or colons (like a natural-language message) isn't misparsed.
    """
    if not known_keys:
        # Best-effort fallback: naive comma split
        args: dict = {}
        for part in body.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                args[k.strip()] = v.strip().strip('"\'')
        return args

    positions: list[tuple[int, int, str]] = []
    for key in known_keys:
        for m in re.finditer(rf"\b{re.escape(key)}\s*:\s*", body):
            positions.append((m.start(), m.end(), key))
    positions.sort()

    args = {}
    for i, (_start, end, key) in enumerate(positions):
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        value = body[end:next_start].strip().rstrip(",").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        args[key] = value
    return args


# ── Core analysis ───────────────────────────────────────────────────────────


def analyze_audio(
    model_path: str,
    wav_path: str,
    freq_mhz: float,
    duration: float,
    watch: str,
    alert_fn: Callable[[str, str], str],
    on_status: Callable[[str], None] | None = None,
    save_dir: Path | str | None = None,
    force_cpu: bool = False,
    max_gpu_memory: str | None = None,
) -> str:
    """Two-pass audio analysis: transcribe then evaluate.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned Gemma 4 E4B HuggingFace checkpoint.
    wav_path : str
        Path to the WAV file to analyze.
    freq_mhz : float
        Center frequency in MHz.
    duration : float
        Capture duration in seconds.
    watch : str
        Natural-language watch criteria.
    alert_fn : (message, urgency) -> str
        Called when the model triggers an alert.
    on_status : (text) -> None, optional
        Called with status messages.
    save_dir : Path or str, optional
        If set, save a copy of every analyzed clip to this directory.

    Returns
    -------
    str
        The transcript from pass 1.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = save_dir / f"cycle_{ts}_{freq_mhz}MHz.wav"
        shutil.copy2(wav_path, dest)
        if on_status:
            on_status(f"Saved debug clip: {dest.name}")

    # ── Pre-LLM energy gate ─────────────────────────────────────────────
    # The squelch gate in capture.py zeros out noise-only frames, so
    # silence arrives as all-zero audio.  Check RMS *before* touching the
    # GPU — skipping the entire 5-10s LLM pipeline on empty clips.
    SILENCE_RMS_THRESHOLD = 0.005
    try:
        audio_check = _load_audio_16k(wav_path)
    except Exception as e:
        # If we can't read the WAV here, _transcribe will fail on the same
        # file — fail fast instead of spinning up the GPU for nothing.
        _log.error("Energy gate: audio load failed for %s: %s", wav_path, e)
        if on_status:
            on_status(f"Audio load failed: {e}")
        return f"[error: audio load failed: {e}]"
    rms = float(np.sqrt(np.mean(audio_check ** 2)))
    _log.info("Energy gate: rms=%.4f (threshold=%.4f)", rms, SILENCE_RMS_THRESHOLD)
    if rms < SILENCE_RMS_THRESHOLD:
        _log.info("Energy gate: SKIP — no signal detected")
        if on_status:
            on_status("[no signal]")
        return "[no signal]"

    from capture import is_wbfm
    mode = "WBFM" if is_wbfm(freq_mhz) else "NBFM"

    if on_status:
        on_status("Loading model (first call may take a while)...")
    model, processor = _get_model(
        model_path, force_cpu=force_cpu, max_gpu_memory=max_gpu_memory,
    )

    transcript = _transcribe(model, processor, wav_path, on_status)

    # ── Post-ASR safety net ───────────────────────────────────────────
    # If the model echoed the system prompt, treat as silence.
    asr_prompt = build_asr_prompt()
    if transcript.strip() == asr_prompt.strip():
        _log.warning("Post-ASR safety net: transcript matches system prompt — treating as silence")
        if on_status:
            on_status("[no signal]")
        return "[no signal]"

    _evaluate(model, processor, transcript, freq_mhz, mode, watch, alert_fn, on_status)

    return transcript


def _transcribe(
    model,
    processor,
    wav_path: str,
    on_status: Callable[[str], None] | None,
) -> str:
    """Pass 1: transcribe audio using fine-tuned Gemma 4 E4B."""
    try:
        audio = _load_audio_16k(wav_path)
        _log.info("ASR audio loaded: shape=%s rms=%.4f max=%.4f",
                  audio.shape, float(np.sqrt(np.mean(audio ** 2))),
                  float(np.max(np.abs(audio))))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": build_asr_prompt()},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {
            k: (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        _log.debug("ASR input keys=%s input_ids shape=%s",
                   list(inputs.keys()), tuple(inputs["input_ids"].shape))

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        transcript = processor.tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()
        _log.info("ASR output: %r", transcript)

        if not transcript:
            transcript = "[no speech]"
        if on_status:
            on_status(f"[Transcript] {transcript}")
        return transcript

    except Exception as e:
        _log.error("ASR failed: %s\n%s", e, traceback.format_exc())
        msg = f"ASR transcription failed: {e}"
        if on_status:
            on_status(msg)
        return f"[error: {msg}]"


def _evaluate(
    model,
    processor,
    transcript: str,
    freq_mhz: float,
    mode: str,
    watch: str,
    alert_fn: Callable[[str, str], str],
    on_status: Callable[[str], None] | None,
) -> None:
    """Pass 2: evaluate transcript against watch criteria (text-only, tool-calling)."""
    if transcript.startswith("[error:"):
        return
    no_speech_markers = {"[no speech]", "[inaudible]", "[silence]"}
    if transcript.strip().lower() in no_speech_markers or not transcript.strip():
        if on_status:
            on_status("Skipped evaluation — no speech detected.")
        return

    # Reset the rolling history whenever the tuned frequency changes — cross-
    # frequency conversations would just confuse the evaluator.
    global _eval_history_freq
    if _eval_history_freq is None or _eval_history_freq != freq_mhz:
        if _eval_history_freq is not None:
            reset_eval_history(f"freq changed {_eval_history_freq} → {freq_mhz}")
        _eval_history_freq = freq_mhz

    system_prompt = build_evaluation_prompt(freq_mhz, mode, watch)
    # Content must be a list of typed parts — the multimodal processor iterates
    # over it looking for {"type": ...} entries. A bare string breaks that.
    new_user_msg = {
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"Here is the transcription of the radio capture:\n\n{transcript}",
        }],
    }
    initial_messages: list[dict] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        *list(_eval_history),
        new_user_msg,
    ]

    _log.info(
        "Eval: transcript=%r freq=%s mode=%s history=%d msg(s)",
        transcript, freq_mhz, mode, len(_eval_history),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _log.debug("Eval attempt %d: applying chat template", attempt)
            inputs = processor.apply_chat_template(
                initial_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {
                k: (v.to(model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            _log.debug("Eval attempt %d: input_ids shape=%s", attempt,
                       tuple(inputs["input_ids"].shape))

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_text = processor.tokenizer.decode(new_ids, skip_special_tokens=False)
            clean_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            _log.info("Eval raw output: %r", raw_text)
            _log.info("Eval clean output: %r", clean_text)

            tool_calls = _parse_tool_calls(raw_text, _ALERT_PARAM_NAMES)
            if not tool_calls:
                tool_calls = _parse_tool_calls(clean_text, _ALERT_PARAM_NAMES)
            _log.info("Eval parsed %d tool call(s): %s", len(tool_calls), tool_calls)

            # Record this turn in the rolling history. deque(maxlen=8) evicts
            # the oldest pair automatically once we go past 4 cycles.
            assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": clean_text or ""}],
            }
            _eval_history.append(new_user_msg)
            _eval_history.append(assistant_msg)

            if not tool_calls:
                if clean_text and on_status:
                    on_status(clean_text)
                return

            seen_alerts: set[tuple] = set()
            for tc in tool_calls:
                fn_name = tc.get("name", "")
                fn_args = tc.get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}

                if fn_name == "alert_user":
                    alert_key = (
                        fn_args.get("message", ""),
                        fn_args.get("urgency", ""),
                    )
                    if alert_key in seen_alerts:
                        continue
                    seen_alerts.add(alert_key)
                    alert_fn(
                        fn_args.get("message", ""),
                        fn_args.get("urgency", "low"),
                    )
            return

        except Exception as e:
            _log.error(
                "Eval attempt %d failed: %s\n%s",
                attempt, e, traceback.format_exc(),
            )
            if attempt < MAX_RETRIES:
                if on_status:
                    on_status(
                        f"Evaluation failed (attempt {attempt}/{MAX_RETRIES}), "
                        f"retrying in {RETRY_DELAY}s..."
                    )
                time.sleep(RETRY_DELAY)
            else:
                if on_status:
                    on_status(
                        f"Evaluation failed after {MAX_RETRIES} attempts: {e} "
                        f"(see {_LOG_PATH.name} for traceback)"
                    )
