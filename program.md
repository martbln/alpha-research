# Research Goal

Maximize `val_acc` — accuracy at predicting whether the S&P 500 closes **up or down**
the next trading day. The val split covers 2020–2024 (COVID crash, 2022 bear, 2023–24 bull).

## Rules

- Modify **only** `train.py`. Do not touch `data/` or any other file.
- `BUDGET_SECONDS = 300` must not change.
- The final printed line must be `val_acc: X.XXXX`.
- No lookahead: features for day *t* may only use data available at the close of day *t*.
  (e.g. tomorrow's open is not a valid feature.)
- The val split must stay time-ordered and must not be shuffled.

## Why even 54 % matters

The baseline sits around 50–52 % — barely above a coin flip. In live trading,
a consistent 54–55 % directional edge compounded over hundreds of trades per year
is highly profitable. The agent is trying to find that edge autonomously.

## Promising directions

### Better features (highest leverage — start here)
- **Multi-day returns** — add 2d, 5d, 10d, 20d, 60d lookback windows
- **Volatility** — rolling standard deviation of daily returns (5d, 20d)
- **Momentum** — price relative to N-day moving average
- **RSI** (Relative Strength Index) — classic 14-day overbought/oversold signal
- **MACD** — trend-following momentum indicator
- **Bollinger Band width** — measures volatility regime
- **Volume features** — volume change, On-Balance Volume
- **Multi-asset features** — add VIX (^VIX), 10-year yield (^TNX), gold (GC=F)
  as additional tickers via yfinance; these are all free
- **Day-of-week / month-of-year** — calendar effects are real in equity markets

### Better architecture
- **LSTM or GRU** — model the full return sequence instead of a flat feature vector
- **Transformer / attention** — attend to different historical regimes
- **Wider/deeper MLP** — current baseline is tiny (32 hidden units)
- **Batch normalisation** or **layer norm**

### Better training
- **Learning rate schedule** — cosine decay or OneCycleLR
- **Dropout** — market data is noisy; regularisation helps generalisation
- **Class weighting** — up/down days are not always balanced
- **Larger batch size** — data is small enough to fit in memory

## Expected improvement trajectory

| Approach | Typical val_acc |
|---|---|
| Baseline (this file) | ~50–52 % |
| + multi-day returns + volatility | ~52–54 % |
| + RSI / MACD / multi-asset | ~54–56 % |
| + LSTM + full feature set | ~55–58 % |
