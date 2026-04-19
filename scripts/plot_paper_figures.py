"""Generate the paper figures from JSON summaries and optional raw prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-quality figures from model summaries.")
    parser.add_argument(
        "--summary-json",
        nargs="+",
        default=["outputs/final_paper_run.json", "outputs/ablation_run.json"],
        help="JSON summary files from compare_models.py or similar runs.",
    )
    parser.add_argument(
        "--raw-predictions",
        nargs="*",
        default=[],
        help="Optional raw prediction CSV files containing p_up and cumulative-return columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save generated figure PNGs.",
    )
    return parser.parse_args()


def get_run_name(path: Path, payload: dict[str, Any]) -> str:
    lower = path.stem.lower()
    if "ablation" in lower:
        return "Ablation"
    if "final" in lower or "paper" in lower or "full" in lower:
        return "Full"
    if payload.get("paper_run"):
        return "Full"
    return path.stem


def flatten_json_summary(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_name = get_run_name(path, payload)
    rows: list[dict[str, Any]] = []
    cross_model: dict[str, Any] = {}
    for model_key in ("baseline_lstm", "causal_fusion_lstm"):
        if model_key not in payload:
            continue
        model_payload = payload[model_key]
        diagnostics = model_payload.get("probability_diagnostics", {})
        row: dict[str, Any] = {
            "run": run_name,
            "model": model_key,
            "accuracy": model_payload.get("accuracy"),
            "f1": model_payload.get("f1"),
            "final_cum_strategy_return": model_payload.get("final_cum_strategy_return"),
            "final_cum_buy_hold_return": model_payload.get("final_cum_buy_hold_return"),
            "mean_p_up": diagnostics.get("mean_p_up"),
            "std_p_up": diagnostics.get("std_p_up"),
            "min_p_up": diagnostics.get("min_p_up"),
            "max_p_up": diagnostics.get("max_p_up"),
            "frac_p_up_above_threshold": diagnostics.get("frac_p_up_above_threshold"),
        }
        rows.append(row)
    if "cross_model" in payload:
        cross_model = {"run": run_name, **payload["cross_model"]}
    return rows, cross_model


def build_summary_dataframe(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Summary JSON missing: {path}")
        file_rows, cross_row = flatten_json_summary(path)
        rows.extend(file_rows)
        if cross_row:
            cross_rows.append(cross_row)
    cross_df = pd.DataFrame(cross_rows) if cross_rows else None
    return pd.DataFrame(rows), cross_df


def plot_accuracy_f1(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{run}\n{model}" for run, model in zip(df["run"], df["model"])]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy", color="#4c72b0")
    ax.bar(x + width / 2, df["f1"], width, label="F1", color="#dd8452")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and F1 by Run and Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_f1_by_model.png", dpi=300)
    plt.close(fig)


def plot_strategy_vs_buy_hold(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{run}\n{model}" for run, model in zip(df["run"], df["model"])]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["final_cum_strategy_return"], width, label="Strategy", color="#2ca02c")
    ax.bar(x + width / 2, df["final_cum_buy_hold_return"], width, label="Buy-and-Hold", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Final Strategy Return vs Buy-and-Hold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_vs_buy_hold.png", dpi=300)
    plt.close(fig)


def plot_probability_diagnostics(df: pd.DataFrame, cross_df: pd.DataFrame | None, output_dir: Path) -> None:
    metrics = ["mean_p_up", "std_p_up", "min_p_up", "max_p_up"]
    figs = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, metric in zip(axes.flat, metrics):
        pivot = df.pivot(index="run", columns="model", values=metric)
        pivot.plot.bar(ax=ax, rot=0)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Probability Diagnostics by Run and Model", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "probability_diagnostics.png", dpi=300)
    plt.close(fig)

    if cross_df is not None and not cross_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        cross_df = cross_df.set_index("run")
        cross_df.plot.bar(ax=ax, rot=0, colormap="tab10")
        ax.set_title("Cross-Model p_up Diagnostics")
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "cross_model_probability_stats.png", dpi=300)
        plt.close(fig)


def plot_raw_probabilities(raw_paths: list[str], output_dir: Path) -> None:
    if not raw_paths:
        return

    all_frames: list[pd.DataFrame] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw prediction CSV missing: {path}")
        df = pd.read_csv(path)
        source = path.stem
        if "model" not in df.columns:
            if "baseline" in source.lower():
                df["model"] = "baseline_lstm"
            elif "causal" in source.lower():
                df["model"] = "causal_fusion_lstm"
            else:
                df["model"] = source
        df["source"] = df["model"]
        all_frames.append(df)
    predictions = pd.concat(all_frames, ignore_index=True)
    if "p_up" not in predictions.columns:
        raise ValueError("Raw prediction CSV files must contain a p_up column.")

    fig, ax = plt.subplots(figsize=(10, 5))
    predictions.boxplot(column="p_up", by="source", ax=ax)
    ax.set_title("Distribution of p_up by Model")
    ax.set_xlabel("")
    ax.set_ylabel("p_up")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "p_up_distribution_boxplot.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for source, group in predictions.groupby("source"):
        ax.hist(group["p_up"].dropna(), bins=30, alpha=0.5, label=source)
    ax.set_title("p_up Distribution by Model")
    ax.set_xlabel("p_up")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "p_up_distribution_histogram.png", dpi=300)
    plt.close(fig)

    if {"cum_strategy_return", "cum_buy_hold_return"}.issubset(predictions.columns):
        plot_cumulative_returns_from_predictions(predictions, output_dir)


def plot_cumulative_returns_from_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    if "timestamp" in predictions.columns:
        index = pd.to_datetime(predictions["timestamp"], errors="coerce")
        if index.isna().all():
            index = np.arange(len(predictions))
    elif "date" in predictions.columns:
        index = pd.to_datetime(predictions["date"], errors="coerce")
        if index.isna().all():
            index = np.arange(len(predictions))
    elif "eval_sample_index" in predictions.columns:
        index = predictions["eval_sample_index"]
    else:
        index = np.arange(len(predictions))

    fig, ax = plt.subplots(figsize=(10, 5))
    for source, group in predictions.groupby("source"):
        ax.plot(index, group["cum_strategy_return"], label=f"{source} Strategy", linewidth=2)
        ax.plot(index, group["cum_buy_hold_return"], label=f"{source} Buy-and-Hold", linewidth=2, linestyle="--")
    ax.set_title("Cumulative Returns Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns_over_time.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, cross_df = build_summary_dataframe(args.summary_json)
    if summary_df.empty:
        raise ValueError("No summary data found to plot.")

    plot_accuracy_f1(summary_df, output_dir)
    plot_strategy_vs_buy_hold(summary_df, output_dir)
    plot_probability_diagnostics(summary_df, cross_df, output_dir)
    if args.raw_predictions:
        plot_raw_probabilities(args.raw_predictions, output_dir)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
