# Causal LSTM Stock Movement

Predict next-day stock direction (**UP/DOWN**) using three optional feature groups:
- **Price**
- **News**
- **Causal/Macro**

The repo supports two model architectures:
- `baseline_lstm`
- `causal_fusion_lstm`

## What you need to run the project

Minimum required artifacts:
1. Python dependencies installed (`requirements.txt`)
2. Raw CSV inputs for price/news/causal (sample files are included)
3. Fused dataset: `data/processed/fused_dataset.csv`
4. Trained checkpoint: `checkpoints/latest_model.pt`

Default data and modality toggles are in `configs/data.yaml`.

## Quick start (essential commands)

```bash
PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/train_model.py
PYTHONPATH=src python scripts/evaluate_model.py
```

Run single-step inference:

```bash
PYTHONPATH=src python scripts/run_inference.py --output-json outputs/inference_latest.json
```

## Compare pipeline levels (3-level scoring)

Use `configs/data.yaml` → `data.modalities` to run these three levels:

1. **Level 1 (price only)**
   - `price: true`, `news: false`, `causal: false`
2. **Level 2 (price + news)**
   - `price: true`, `news: true`, `causal: false`
3. **Level 3 (full model)**
   - `price: true`, `news: true`, `causal: true`

For each level, run:

```bash
PYTHONPATH=src python scripts/compare_models.py \
  --paper-run --max-steps 50 \
  --output-json outputs/compare_<level>.json
```

This outputs A/B metrics for baseline vs causal-fusion, plus probability diagnostics.

## Visualize outputs

Generate walk-forward files:

```bash
PYTHONPATH=src python scripts/walk_forward_backtest.py \
  --output-csv outputs/walk_forward_predictions.csv \
  --summary-json outputs/walk_forward_summary.json
```

Plot from `outputs/walk_forward_predictions.csv`:
- `cum_strategy_return`
- `cum_buy_hold_return`

## Optional scripts

- `scripts/build_finbert_features.py` (build FinBERT daily features)
- `scripts/fetch_real_data.py` (pull/assemble real-style raw data)
- `scripts/live_predict_job.py` (live-style fetch + inference job)

## Run tests

```bash
PYTHONPATH=src pytest -q
```
