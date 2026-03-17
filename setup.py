"""
One-time data download — run via setup_command before training starts.
Downloads and caches S&P 500 OHLCV data so train.py doesn't need network access.
"""

import os
import yfinance as yf

os.makedirs("data", exist_ok=True)
out = "data/sp500.csv"

if os.path.exists(out):
    print(f"Data already cached at {out}")
else:
    print("Downloading S&P 500 data (2000–2024)...")
    df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31",
                     progress=False, auto_adjust=True)
    df.to_csv(out)
    print(f"Saved {len(df)} trading days → {out}")
