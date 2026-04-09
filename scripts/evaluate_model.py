from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from causal_lstm_stock.config import load_config
from causal_lstm_stock.data.dataset_builder import build_sequences
from causal_lstm_stock.evaluate import evaluate_classifier
from causal_lstm_stock.features.modalities import select_feature_columns
from causal_lstm_stock.models.baseline_lstm import BaselineLSTM
from causal_lstm_stock.models.causal_fusion_lstm import CausalFusionLSTM


def _build_model(arch: str, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int):
    if arch == "baseline_lstm":
        return BaselineLSTM(input_dim, hidden_dim, num_layers, dropout, num_classes)
    if arch == "causal_fusion_lstm":
        return CausalFusionLSTM(input_dim, hidden_dim, num_layers, dropout, num_classes)
    raise ValueError(f"Unknown architecture: {arch}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    fused_path = root / cfg.data.paths["fused_output_csv"]
    ckpt_path = root / cfg.train.checkpoint_dir / "latest_model.pt"

    if not fused_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError("Missing fused dataset or checkpoint. Run prepare_data.py and train_model.py first.")

    fused_df = pd.read_csv(fused_path)
    fused_df["date"] = pd.to_datetime(fused_df["date"])
    feature_cols = select_feature_columns(fused_df, modalities=cfg.data.modalities)
    if not feature_cols:
        raise ValueError("No feature columns selected. Check data.modalities in configs/data.yaml.")
    ds = build_sequences(
        fused_df,
        lookback_window=cfg.data.lookback_window,
        feature_columns=feature_cols,
    )

    model_input_dim = ds.X.shape[-1]
    model = _build_model(
        arch=cfg.model.architecture,
        input_dim=model_input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    )
    try:
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
    except RuntimeError as err:
        raise RuntimeError(
            "Checkpoint is incompatible with the current feature set. "
            "If you changed modality toggles or FinBERT settings, retrain first with scripts/train_model.py."
        ) from err

    metrics = evaluate_classifier(model, ds.X, ds.y)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Feature columns used ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()
