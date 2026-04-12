"""
Compare baseline LSTM vs CausalFusionLSTM using identical walk-forward retraining
on the same fused dataset and feature columns.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from causal_lstm_stock.config import load_config
from causal_lstm_stock.evaluate.probability_metrics import cross_model_p_up_stats, probability_diagnostics
from causal_lstm_stock.evaluate.walk_forward import (
    build_backtest_arrays,
    prepare_fused_from_raw,
    run_walk_forward,
)
from causal_lstm_stock.features.modalities import select_feature_columns


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metrics_from_pred_df(pred_df: pd.DataFrame) -> dict[str, Any]:
    y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "final_cum_strategy_return": float(pred_df["cum_strategy_return"].iloc[-1]),
        "final_cum_buy_hold_return": float(pred_df["cum_buy_hold_return"].iloc[-1]),
        "num_predictions": int(len(pred_df)),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "final_strategy_cum",
        "final_buy_hold_cum",
        "n_preds",
    ]
    col_widths = [max(len(h), 12) for h in headers]
    for row in rows:
        for i, h in enumerate(headers):
            col_widths[i] = max(col_widths[i], len(str(row.get(h, ""))) + 2)

    def line(cells: list[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    print(line(headers))
    print(line(["-" * col_widths[i] for i in range(len(headers))]))
    for row in rows:
        print(
            line(
                [
                    str(row.get("model", "")),
                    f"{row.get('accuracy', 0):.4f}",
                    f"{row.get('precision', 0):.4f}",
                    f"{row.get('recall', 0):.4f}",
                    f"{row.get('f1', 0):.4f}",
                    f"{row.get('final_cum_strategy_return', 0):.6f}",
                    f"{row.get('final_cum_buy_hold_return', 0):.6f}",
                    str(row.get("num_predictions", "")),
                ]
            )
        )


def _print_prob_table(rows: list[tuple[str, dict[str, Any]]]) -> None:
    headers = ["model", "mean_p_up", "std_p_up", "min_p_up", "max_p_up", "frac_pred_up", "frac_p_ge_thr"]
    col_widths = [14, 12, 12, 10, 10, 14, 12]
    for name, d in rows:
        col_widths[0] = max(col_widths[0], len(name) + 2)

    def line(cells: list[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    print()
    print("Probability diagnostics (can differ when discrete metrics match):")
    print(line(headers))
    print(line(["-" * col_widths[i] for i in range(len(headers))]))
    for name, d in rows:
        print(
            line(
                [
                    name,
                    f"{d['mean_p_up']:.4f}",
                    f"{d['std_p_up']:.4f}",
                    f"{d['min_p_up']:.4f}",
                    f"{d['max_p_up']:.4f}",
                    f"{d['frac_predict_up']:.4f}",
                    f"{d['frac_p_up_above_threshold']:.4f}",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A/B walk-forward: baseline_lstm vs causal_fusion_lstm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", type=str, default=None, help="Ticker. Default: config.")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for UP class; also used for frac_p_up_above_threshold.",
    )
    p.add_argument(
        "--min-train-samples",
        type=int,
        default=None,
        help="First walk-forward index to start evaluation. If omitted: 60 normally, or 150 with --paper-run. "
        "Too small + few epochs often yields majority-class collapse (e.g. always predict UP).",
    )
    p.add_argument("--retrain-frequency", type=int, default=5, help="Retrain every N eval steps.")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs per retrain. If omitted: train.yaml value normally, or 30 with --paper-run. "
        "Increase (e.g. 30–50) for meaningful separation vs underfitting.",
    )
    p.add_argument("--max-steps", type=int, default=None, help="Cap evaluation steps (debug).")
    p.add_argument(
        "--paper-run",
        action="store_true",
        help="Preset for heavier runs: min_train_samples=150 and epochs=30 when those flags are omitted. "
        "Keeps CI/smoke defaults fast when this flag is not used.",
    )

    p.add_argument("--use-raw-data", action="store_true")
    p.add_argument("--prices-csv", type=str, default=None)
    p.add_argument("--news-csv", type=str, default=None)
    p.add_argument("--causal-csv", type=str, default=None)
    p.add_argument("--macro-csv", type=str, default=None)
    p.add_argument("--finbert-csv", type=str, default=None)
    p.add_argument("--save-fused", action="store_true")

    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save summaries and probability diagnostics as JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    min_train = args.min_train_samples
    epochs = args.epochs
    if args.paper_run:
        if min_train is None:
            min_train = 150
        if epochs is None:
            epochs = 30
        print(
            f"--paper-run: using min_train_samples={min_train}, epochs={epochs} "
            f"(where not overridden on CLI)\n"
        )
    if min_train is None:
        min_train = 60
    if epochs is None:
        epochs = cfg.train.epochs

    fused_path = root / cfg.data.paths["fused_output_csv"]
    if args.use_raw_data:
        fused_df = prepare_fused_from_raw(
            root,
            cfg,
            args.prices_csv,
            args.news_csv,
            args.causal_csv,
            args.finbert_csv,
            args.macro_csv,
        )
        if args.save_fused:
            fused_path.parent.mkdir(parents=True, exist_ok=True)
            fused_df.to_csv(fused_path, index=False)
    else:
        if not fused_path.exists():
            raise FileNotFoundError(f"Fused dataset not found: {fused_path}. Run prepare_data.py or --use-raw-data.")
        fused_df = pd.read_csv(fused_path)

    ticker = args.ticker or cfg.data.ticker
    feature_cols = select_feature_columns(fused_df, modalities=cfg.data.modalities)
    X, y, meta_df = build_backtest_arrays(
        fused_df=fused_df,
        ticker=ticker,
        lookback_window=cfg.data.lookback_window,
        feature_cols=feature_cols,
    )

    common_kw = dict(
        X=X,
        y=y,
        meta_df=meta_df,
        cfg=cfg,
        feature_cols=feature_cols,
        min_train_samples=min_train,
        retrain_frequency=args.retrain_frequency,
        threshold=args.threshold,
        epochs=epochs,
        max_steps=args.max_steps,
    )

    _set_seed(cfg.train.seed)
    pred_baseline = run_walk_forward(
        **common_kw,
        architecture="baseline_lstm",
        fusion_cfg=None,
    )
    metrics_baseline = _metrics_from_pred_df(pred_baseline)
    metrics_baseline["model"] = "baseline_lstm"
    prob_baseline = probability_diagnostics(pred_baseline, args.threshold)

    _set_seed(cfg.train.seed)
    pred_causal = run_walk_forward(
        **common_kw,
        architecture="causal_fusion_lstm",
        fusion_cfg=cfg.model.fusion,
    )
    metrics_causal = _metrics_from_pred_df(pred_causal)
    metrics_causal["model"] = "causal_fusion_lstm"
    prob_causal = probability_diagnostics(pred_causal, args.threshold)

    cross = cross_model_p_up_stats(pred_baseline, pred_causal)

    print("Walk-forward A/B (identical data, retrain each run from seed)")
    print(f"  ticker={ticker}  features={len(feature_cols)}  epochs={epochs}  min_train_samples={min_train}")
    print()
    _print_table([metrics_baseline, metrics_causal])
    _print_prob_table(
        [
            ("baseline_lstm", prob_baseline),
            ("causal_fusion_lstm", prob_causal),
        ]
    )
    print()
    print("Cross-model p_up (aligned by eval_sample_index):")
    print(f"  mean_abs_delta_p_up: {cross.get('mean_abs_delta_p_up')}")
    print(f"  corr_p_up: {cross.get('corr_p_up')}")

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ticker": ticker,
            "feature_columns": feature_cols,
            "epochs": epochs,
            "min_train_samples": min_train,
            "threshold": args.threshold,
            "paper_run": bool(args.paper_run),
            "baseline_lstm": {**metrics_baseline, "probability_diagnostics": prob_baseline},
            "causal_fusion_lstm": {**metrics_causal, "probability_diagnostics": prob_causal},
            "cross_model": cross,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON to {out_path}")


if __name__ == "__main__":
    main()
