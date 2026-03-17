# Alpha Research

Can AI autonomously discover features that predict the S&P 500's next-day direction?

This repo applies the NightResearch autoresearch loop to **quantitative finance** —
a domain where even a 3–4 % edge over random has real trading value.

## The setup

| | |
|---|---|
| **Task** | Binary classification: will S&P 500 close higher or lower tomorrow? |
| **Data** | 24 years of daily OHLCV (2000–2024, ~6 000 trading days) |
| **Val set** | Last 20 % — covers COVID crash, 2022 bear, 2023–24 bull run |
| **Metric** | `val_acc` (direction accuracy) — maximize |
| **Budget** | 5 minutes of GPU training per attempt |

## Why this is interesting

The baseline sits at ~50–52 % — barely above a coin flip.
In live trading, a consistent **54–55 % directional edge** is worth a lot
when compounded across hundreds of trades per year.

The agent has to discover that edge autonomously by finding better features
(RSI, MACD, multi-asset inputs) and better architectures (LSTM, Transformer).

## Expected improvement

| Approach | Typical val_acc |
|---|---|
| Baseline (naive MLP, 1 feature) | ~50–52 % |
| + technical indicators | ~52–54 % |
| + multi-asset + LSTM | ~55–58 % |

## NightResearch config

| Field | Value |
|---|---|
| Setup command | `pip install yfinance pandas --quiet && python setup.py` |
| Run command | `python train.py` |
| Metric regex | `val_acc: ([0-9.]+)` |
| Direction | maximize |
| Editable | `train.py` |
| Blocked | `data/**`, `setup.py` |
| Engine | `autoresearch_agent` |
| Timeout | `420` |
