"""Plot train + val loss from runs/train_log.jsonl.

The log contains entries from every run (smoke + full), distinguished by
monotonic step progression. We pick the last contiguous ascending sequence
so "latest full run wins" even if older smoke runs are still in the file.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path(__file__).parent / "runs" / "train_log.jsonl"
OUT = Path(__file__).parent / "runs" / "loss_curve.png"


def main():
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]

    # Split runs: a new run starts when step resets or decreases.
    runs: list[list[dict]] = []
    cur: list[dict] = []
    last_step = -1
    for r in rows:
        if r["step"] < last_step:  # val entries share step with preceding train entry; only RESET marks a new run
            if cur:
                runs.append(cur)
            cur = []
        cur.append(r)
        last_step = r["step"]
    if cur:
        runs.append(cur)

    # Pick the run with the most steps (i.e. the 1000-step full run)
    run = max(runs, key=len)
    print(f"Plotting run with {len(run)} log entries "
          f"(steps {run[0]['step']}..{run[-1]['step']})")

    train = [(r["step"], r["loss"]) for r in run if "loss" in r]
    val = [(r["step"], r["val_loss"]) for r in run if "val_loss" in r]

    fig, ax = plt.subplots(figsize=(9, 5))
    if train:
        xs, ys = zip(*train)
        ax.plot(xs, ys, color="#4a90e2", alpha=0.45, linewidth=1.0,
                label="train loss (per log-interval mean)")
        # Running mean for readability
        w = 10
        if len(ys) >= w:
            import numpy as np
            ys_arr = np.array(ys)
            smooth = np.convolve(ys_arr, np.ones(w) / w, mode="valid")
            xs_s = xs[w - 1:]
            ax.plot(xs_s, smooth, color="#1a4f8c", linewidth=2.0,
                    label=f"train loss (running mean, w={w})")
    if val:
        xs, ys = zip(*val)
        ax.plot(xs, ys, "o-", color="#d9534f", linewidth=2.0, markersize=7,
                label="val loss (held-out speakers)")
        for x, y in val:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="#8a2a27")

    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Gemma 4 E4B audio-encoder LoRA — 1000-step fine-tune")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
