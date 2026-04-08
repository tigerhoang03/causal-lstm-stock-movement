#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[smoke] Preparing fused dataset"
python scripts/prepare_data.py
if [[ ! -f "$ROOT_DIR/data/processed/fused_dataset.csv" ]]; then
  echo "[smoke][error] Missing fused dataset: data/processed/fused_dataset.csv"
  exit 1
fi

echo "[smoke] Training model"
python scripts/train_model.py
if [[ ! -f "$ROOT_DIR/checkpoints/latest_model.pt" ]]; then
  echo "[smoke][error] Missing checkpoint: checkpoints/latest_model.pt"
  exit 1
fi

echo "[smoke] Evaluating model"
python scripts/evaluate_model.py

echo "[smoke] End-to-end smoke test passed"
