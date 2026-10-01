"""Parse a TRL/GRPO training log and generate metric plots.

Run from the project root or this directory:
    python Smol_fine_tuning/plot_training_logs.py

The script reads dictionary-style metric lines emitted by GRPOTrainer and
writes PNG files plus a CSV copy of the parsed metrics.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "outputs" / "Qwen2.5-0.5-Instruct-GRPO" / "Traning_logs.txt"
DEFAULT_OUTPUT = DEFAULT_LOG.parent / "plots"


def parse_metrics(log_path: Path) -> list[dict[str, float]]:
    """Extract trainer metric dictionaries from a text log."""
    records: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith("{") or not text.endswith("}"):
                continue
            try:
                value: Any = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(value, dict) or "epoch" not in value:
                continue
            record: dict[str, float] = {}
            for key, item in value.items():
                try:
                    record[str(key)] = float(item)
                except (TypeError, ValueError):
                    pass
            if "epoch" in record:
                records.append(record)
    if not records:
        raise ValueError(f"No trainer metric records found in {log_path}")
    return records


def save_csv(records: list[dict[str, float]], output_path: Path) -> None:
    fields = sorted({key for record in records for key in record})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def values(records: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    points = [(record["epoch"], record[key]) for record in records if key in record]
    return [point[0] for point in points], [point[1] for point in points]


def plot_series(
    records: list[dict[str, float]],
    series: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for key, label, color in series:
        x_axis, y_axis = values(records, key)
        if x_axis:
            axis.plot(x_axis, y_axis, linewidth=1.5, label=label, color=color)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(records: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_series(
        records,
        [("loss", "Loss", "#2878b5"), ("reward", "Reward", "#c43d3d")],
        "GRPO loss and reward",
        "Value",
        output_dir / "loss_reward.png",
    )
    plot_series(
        records,
        [
            ("rewards/reward_function/mean", "Mean reward", "#2878b5"),
            ("rewards/reward_function/std", "Reward std", "#c43d3d"),
        ],
        "Reward statistics",
        "Reward",
        output_dir / "reward_statistics.png",
    )
    plot_series(
        records,
        [
            ("completions/mean_length", "Mean completion length", "#2878b5"),
            ("completions/mean_terminated_length", "Mean terminated length", "#5a9e54"),
            ("completions/max_length", "Maximum completion length", "#c43d3d"),
        ],
        "Generated completion lengths",
        "Tokens",
        output_dir / "completion_lengths.png",
    )
    plot_series(
        records,
        [("learning_rate", "Learning rate", "#2878b5")],
        "Learning-rate schedule",
        "Learning rate",
        output_dir / "learning_rate.png",
    )
    plot_series(
        records,
        [("entropy", "Entropy", "#7b4ab5"), ("grad_norm", "Gradient norm", "#e07b39")],
        "Entropy and gradient norm",
        "Value",
        output_dir / "entropy_gradient_norm.png",
    )
    plot_series(
        records,
        [("completions/clipped_ratio", "Clipped ratio", "#c43d3d")],
        "Completion clipping ratio",
        "Ratio",
        output_dir / "clipped_ratio.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Training log path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for PNG plots and parsed_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log.resolve()
    output_dir = args.output_dir.resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    records = parse_metrics(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(records, output_dir / "parsed_metrics.csv")
    make_plots(records, output_dir)
    print(f"Parsed {len(records):,} metric records from {log_path}")
    print(f"Wrote plots and CSV to {output_dir}")


if __name__ == "__main__":
    main()