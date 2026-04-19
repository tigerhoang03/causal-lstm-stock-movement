# Causal LSTM Stock Movement

Predict next-day stock direction (**UP/DOWN**) from:
- price data
- news data
- causal/macro signals

Supported models:
- `baseline_lstm`
- `causal_fusion_lstm`

---

## 0) Start from scratch (VS Code + Python)

These steps assume you already have **VS Code** and **Python 3.10+** installed.

### A. Open the project
1. Open VS Code.
2. `File` → `Open Folder...` → select this repo folder.
3. Open a terminal in VS Code (`Terminal` → `New Terminal`).

### B. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### C. Install dependencies
```bash
pip install -r requirements.txt
```

### D. Set `PYTHONPATH`

**macOS / Linux**
```bash
export PYTHONPATH=src
```

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH="src"
```

> You must set `PYTHONPATH` in each new terminal session.

---

## 1) Run the project end-to-end (sample data)

From the repo root, run:

```bash
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

If these complete, you now have:
- fused dataset: `data/processed/fused_dataset.csv`
- model checkpoint: `checkpoints/latest_model.pt`

---

## 2) Run inference (single prediction)

```bash
python scripts/run_inference.py --output-json outputs/inference_latest.json
```

This prints prediction info in terminal and saves JSON output.

---

## 3) Compare performance across the 3 pipeline levels

Edit `configs/data.yaml` under `data.modalities` and run comparison after each change.

### Level 1: Price only
```yaml
price: true
news: false
causal: false
```

### Level 2: Price + News
```yaml
price: true
news: true
causal: false
```

### Level 3: Full model
```yaml
price: true
news: true
causal: true
```

For each level:

```bash
python scripts/compare_models.py \
  --paper-run --max-steps 50 \
  --output-json outputs/compare_<level>.json
```

This gives baseline-vs-causal model metrics and probability diagnostics.

---

## 4) Visualize strategy performance

Generate walk-forward outputs:

```bash
python scripts/walk_forward_backtest.py \
  --output-csv outputs/walk_forward_predictions.csv \
  --summary-json outputs/walk_forward_summary.json
```

Plot these columns from `outputs/walk_forward_predictions.csv`:
- `cum_strategy_return`
- `cum_buy_hold_return`

---

## 5) Run tests

```bash
pytest -q
```

---

## Common issues

1. **`ModuleNotFoundError: causal_lstm_stock`**
   - You forgot to set `PYTHONPATH=src` in the current terminal.

2. **Checkpoint mismatch after changing modalities**
   - Re-run:
     - `python scripts/train_model.py`

3. **Not enough rows / lookback error**
   - Use more data or reduce `lookback_window` in `configs/data.yaml`.

---

## Optional scripts

- `scripts/build_finbert_features.py` (build FinBERT daily features)
- `scripts/fetch_real_data.py` (fetch/assemble real-style raw data)
- `scripts/live_predict_job.py` (live-style fetch + inference pipeline)
