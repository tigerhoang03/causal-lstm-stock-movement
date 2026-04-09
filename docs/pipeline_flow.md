# Pipeline Flow (Training + Live Prediction)

This document explains how data flows through the project and how each model signal is used.

## 1) Training Pipeline (Offline)

1. Collect raw files into:
   - `data/raw/prices/*.csv`
   - `data/raw/news/*.csv`
   - `data/raw/causal/*.csv`
2. (Optional but recommended) Run `scripts/build_finbert_features.py`
   - Runs true FinBERT article inference
   - Builds daily aggregated FinBERT features
   - Saves to `data/interim/finbert_daily_features.csv`
   - Requires `torch>=2.6` in the runtime environment
3. Run `scripts/prepare_data.py`
   - Builds price/news/causal features
   - Merges daily FinBERT features when enabled in `configs/data.yaml`
   - Fuses them into `data/processed/fused_dataset.csv`
4. Run `scripts/train_model.py`
   - Creates sliding windows of length `lookback_window`
   - Selects features based on modality flags in config
   - Trains selected architecture (`baseline_lstm` or `causal_fusion_lstm`)
   - Saves checkpoint to `checkpoints/latest_model.pt`
5. Run `scripts/evaluate_model.py`
   - Reports classification metrics (accuracy/precision/recall/f1)

## 2) Live Prediction Pipeline (Free Data)

Run `scripts/live_predict_job.py`.

What it does:
1. Fetches free market data from Yahoo Finance chart API (daily OHLCV), with Stooq fallback
2. Fetches free news headlines from Google News RSS
3. Fetches free macro series from FRED (`VIXCLS`, `FEDFUNDS`, `DGS10`)
4. Writes live raw data files:
   - `data/raw/prices/live_<ticker>_prices.csv`
   - `data/raw/news/live_<ticker>_news.csv`
   - `data/raw/causal/live_<ticker>_causal.csv`
5. If FinBERT is enabled, runs true FinBERT inference on fetched headlines and writes:
   - `data/interim/live_<ticker>_finbert_daily.csv`
6. Optionally retrains checkpoint on fetched live data (`--retrain-on-live-data`)
7. Runs `scripts/run_inference.py` with `--use-raw-data`
8. Writes prediction JSON (default: `outputs/live/latest_prediction.json`)

## 3) Walk-Forward Validation (Realistic Historical Testing)

Run `scripts/walk_forward_backtest.py`.

What it does:
1. Builds sequence samples in time order
2. Trains on past samples only
3. Predicts on the next unseen sample
4. Retrains periodically (`--retrain-frequency`)
5. Tracks:
   - Classification metrics
   - Strategy return vs buy-and-hold return

## 4) How Causal Signals Are Used

Current implementation:
- There is **no separately trained causal neural model yet**.
- Causal signals are exogenous numeric features in `causal_csv`:
  - `intervention_score`
  - `confounder_proxy`
  - `macro_shock_signal`
- After feature fusion, these columns become part of each LSTM input timestep.
- In `causal_fusion_lstm`, hidden states pass through a learned gate before classification.

What this means practically:
- The LSTM learns how to combine price + news + causal channels jointly.
- “Causal modeling” currently lives in signal engineering, not a dedicated SCM learner.

## 5) How FinBERT/Sentiment Is Used

Current implementation:
- True FinBERT inference is implemented through `scripts/build_finbert_features.py`
  and the shared `nlp/finbert_inference.py` module.
- Per-article FinBERT probabilities are computed (`positive`, `neutral`, `negative`).
- Daily `(date, ticker)` features are aggregated and merged into news modality:
  - `finbert_pos_mean`
  - `finbert_neu_mean`
  - `finbert_neg_mean`
  - `finbert_sentiment_mean`
  - `finbert_confidence_mean`
  - `finbert_article_count`

In the live job:
- Live headlines are scored with the same FinBERT module when FinBERT is enabled.
- A lightweight `sentiment_score` heuristic is still retained as a fallback feature.

## 6) Modality Toggle System

Modality controls live in `configs/data.yaml`:

```yaml
modalities:
  price: true
  news: true
  causal: true
  include_other_features: true
```

These toggles are respected by:
- `train_model.py`
- `evaluate_model.py`
- `run_inference.py`
- `walk_forward_backtest.py`
- live retraining inside `live_predict_job.py`

Example ablations:
- Price-only: `price=true, news=false, causal=false`
- Price+News: `price=true, news=true, causal=false`
- Full model: `price=true, news=true, causal=true`

Note:
- For raw-data inference/backtest runs, passing `--finbert-csv` explicitly will include FinBERT daily features even if `finbert.enabled` is false in config.
- If feature dimensions change (for example from modality toggles or FinBERT columns), retrain before loading checkpoints for evaluation/inference.

## 7) Current Limitations (Important)

- Live free feeds can be noisy/delayed.
- Causal signal generation is still proxy-based.
- FinBERT quality depends on article coverage/cleanliness and inference latency.
- Default checkpoint may be stale unless retrained.

## 8) Recommended Operational Routine

Daily (after market close):
1. Run `live_predict_job.py --retrain-on-live-data --archive-run`
2. Save prediction JSON and raw snapshots
3. Next day, compare prediction vs realized move
4. Track rolling live metrics over time
