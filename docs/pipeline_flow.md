# Pipeline Flow (Training + Live Prediction)

This document explains how data flows through the project and how each model signal is used.

## 1) Training Pipeline (Offline)

1. Collect raw files into:
   - `data/raw/prices/*.csv`
   - `data/raw/news/*.csv`
   - `data/raw/causal/*.csv`
2. Run `scripts/prepare_data.py`
   - Builds price/news/causal features
   - Fuses them into `data/processed/fused_dataset.csv`
3. Run `scripts/train_model.py`
   - Creates sliding windows of length `lookback_window`
   - Trains selected architecture (`baseline_lstm` or `causal_fusion_lstm`)
   - Saves checkpoint to `checkpoints/latest_model.pt`
4. Run `scripts/evaluate_model.py`
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
5. Optionally retrains checkpoint on fetched live data (`--retrain-on-live-data`)
6. Runs `scripts/run_inference.py` with `--use-raw-data`
7. Writes prediction JSON (default: `outputs/live/latest_prediction.json`)

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
- A true FinBERT model is **not yet plugged in**.
- News CSV includes `sentiment_score`; loader aggregates daily mean sentiment + article count.
- Those daily values are fused into the LSTM input windows.

In the live job:
- `sentiment_score` is currently a lightweight keyword heuristic from headlines.

Target implementation (recommended):
1. Run FinBERT on each article headline/body
2. Store article-level logits/probabilities
3. Aggregate by `(date, ticker)` (mean, std, max, count)
4. Feed aggregated FinBERT features into fused dataset

## 6) Current Limitations (Important)

- Live free feeds can be noisy/delayed.
- Causal signal generation is still proxy-based.
- News sentiment is heuristic unless FinBERT is integrated.
- Default checkpoint may be stale unless retrained.

## 7) Recommended Operational Routine

Daily (after market close):
1. Run `live_predict_job.py --retrain-on-live-data --archive-run`
2. Save prediction JSON and raw snapshots
3. Next day, compare prediction vs realized move
4. Track rolling live metrics over time
