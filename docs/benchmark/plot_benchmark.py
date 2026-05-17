#!/usr/bin/env python3
"""Generate a bar chart comparing base vs fine-tuned ASR Word Error Rate."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    base_path = Path(__file__).parent / "benchmark_base.json"
    ft_path = Path(__file__).parent / "benchmark_finetuned.json"

    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(ft_path, encoding="utf-8") as f:
        ft = json.load(f)

    # Collect conditions in a nice order
    condition_order = [
        "clean", "clean_link", "nominal_ham",
        "overmodulated", "weak_signal", "ducked", "squelched",
    ]
    # Pretty labels for display
    labels = {
        "clean": "Clean",
        "clean_link": "Clean\n(link sim)",
        "nominal_ham": "Nominal\nHam",
        "overmodulated": "Over-\nmodulated",
        "weak_signal": "Weak\nSignal",
        "ducked": "Ducked",
        "squelched": "Squelched",
    }

    conditions = [c for c in condition_order if c in base["by_condition"]]
    base_wers = [base["by_condition"][c]["avg_wer"] * 100 for c in conditions]
    ft_wers = [ft["by_condition"][c]["avg_wer"] * 100 for c in conditions]
    display_labels = [labels.get(c, c) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_base = ax.bar(x - width / 2, base_wers, width,
                       label=f'Base Gemma 4 E4B ({base["overall_wer"]*100:.1f}% overall)',
                       color="#E57373", edgecolor="white", linewidth=0.5)
    bars_ft = ax.bar(x + width / 2, ft_wers, width,
                     label=f'Fine-tuned ({ft["overall_wer"]*100:.1f}% overall)',
                     color="#4FC3F7", edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8, color="#B71C1C")
    for bar in bars_ft:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8, color="#0277BD")

    ax.set_xlabel("Radio Condition", fontsize=12)
    ax.set_ylabel("Word Error Rate (%)", fontsize=12)
    ax.set_title("AirGrep ASR Benchmark: Base vs Fine-tuned Gemma 4 E4B",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, max(max(base_wers), max(ft_wers)) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out = Path(__file__).parent / "benchmark_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
