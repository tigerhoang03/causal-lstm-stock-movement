from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from causal_lstm_stock.config import load_config
from causal_lstm_stock.data.dataset_builder import build_sequences
from causal_lstm_stock.models.baseline_lstm import BaselineLSTM
from causal_lstm_stock.models.causal_fusion_lstm import CausalFusionLSTM
from causal_lstm_stock.train import train_model


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
    if not fused_path.exists():
        raise FileNotFoundError(f"Fused dataset not found: {fused_path}. Run scripts/prepare_data.py first.")

    fused_df = pd.read_csv(fused_path)
    fused_df["date"] = pd.to_datetime(fused_df["date"])

    ds = build_sequences(fused_df, lookback_window=cfg.data.lookback_window)
    if ds.X.size == 0:
        raise ValueError("No sequences built. Check your data coverage and lookback window.")

    model_input_dim = ds.X.shape[-1]
    model = _build_model(
        arch=cfg.model.architecture,
        input_dim=model_input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    )

    result = train_model(
        model=model,
        X=ds.X,
        y=ds.y,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        seed=cfg.train.seed,
    )

    ckpt_dir = root / cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"Checkpoint saved to: {ckpt_path}")
    print(f"Final train loss: {result.train_loss_history[-1]:.4f}")
    print(f"Final val loss: {result.val_loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
