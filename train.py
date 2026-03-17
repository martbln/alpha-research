"""
S&P 500 daily direction predictor — the ONLY file the agent should modify.

Task   : predict whether S&P 500 closes HIGHER or LOWER than the previous day.
Metric : val_acc  (accuracy on the last 20 % of data, time-ordered — never shuffled)
Budget : BUDGET_SECONDS of wall-clock training time.

Baseline: 2-layer MLP trained on a single feature — yesterday's daily return.
Expected baseline accuracy: ~50-52 % (barely above a coin flip).

The val split covers 2020-2024, so the model is evaluated on recent, unseen market
conditions including COVID crash, 2022 bear market, and 2023-24 bull run.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BUDGET_SECONDS = 300          # 5-minute training budget — do not change
DATA_FILE      = "data/sp500.csv"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_ohlcv() -> "pd.DataFrame":
    """Load cached S&P 500 OHLCV data (downloaded by setup command)."""
    import pandas as pd
    if not os.path.exists(DATA_FILE):
        # Fallback: download on the fly (slower)
        import yfinance as yf
        os.makedirs("data", exist_ok=True)
        df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31",
                         progress=False, auto_adjust=True)
        df.to_csv(DATA_FILE)
        print(f"Downloaded {len(df)} trading days")
    return pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)


def build_dataset(df) -> tuple:
    """
    Build (X_train, y_train, X_val, y_val) from raw OHLCV.

    Baseline feature set: just yesterday's daily return (1 feature).
    Target: 1 if close[t+1] > close[t], else 0.
    Split : first 80 % for training, last 20 % for validation (time-ordered).
    """
    close = df["Close"].squeeze().values.astype(np.float64)

    # Daily returns
    ret = np.diff(close) / (close[:-1] + 1e-10)      # shape (T-1,)

    # Targets: direction of tomorrow's move
    y = (ret > 0).astype(np.float32)                 # shape (T-1,)

    # ---- Baseline: single feature = yesterday's return ----
    # X[i] predicts y[i+1]  (i.e., tomorrow's direction)
    X = ret[:-1].reshape(-1, 1).astype(np.float32)   # shape (T-2, 1)
    y = y[1:]                                         # shape (T-2,)

    # Standardise
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X     = (X - mu) / sigma

    # Time-ordered split — do NOT shuffle
    n     = len(X)
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df                        = load_ohlcv()
    X_tr, y_tr, X_val, y_val = build_dataset(df)
    print(f"train: {len(X_tr)} days  |  val: {len(X_val)} days  |  features: {X_tr.shape[1]}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=64, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=256, shuffle=False,
    )

    model     = Model(X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    epoch = step = 0
    t0       = time.perf_counter()
    deadline = t0 + BUDGET_SECONDS

    while time.perf_counter() < deadline:
        model.train()
        for X_b, y_b in train_loader:
            if time.perf_counter() >= deadline:
                break
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(X_b), y_b).backward()
            optimizer.step()
            step += 1
        epoch += 1

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds    = (model(X_b) > 0).float()
            correct += (preds == y_b).sum().item()
            total   += y_b.size(0)

    val_acc = correct / total
    print(f"epochs: {epoch}")
    print(f"steps: {step}")
    print(f"elapsed_seconds: {time.perf_counter() - t0:.1f}")
    print(f"val_acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
