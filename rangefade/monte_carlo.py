"""Monte Carlo resampling of trade sequences.

Resampling unit is the trading DAY (preserves intra-day correlation and the
trades-per-day cadence the prop rules care about). Two modes:
  shuffle_days   — permute observed days (no reuse): answers "order risk"
  bootstrap_days — sample days with replacement: answers "distribution risk"
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def days_from_trades(tdf: pd.DataFrame) -> list[np.ndarray]:
    """Group per-contract $PnLs by trading day, chronological within day."""
    if tdf.empty:
        return []
    out = []
    for _, sub in tdf.sort_values("entry_time").groupby("trading_day", sort=True):
        out.append(sub["pnl_usd"].to_numpy(float))
    return out


def resample_day_paths(
    days: list[np.ndarray],
    n_paths: int,
    path_days: int,
    mode: str = "shuffle_days",
    seed: int = 7,
):
    """Yield n_paths sequences of day-arrays of length path_days."""
    rng = np.random.default_rng(seed)
    n = len(days)
    if n == 0:
        return
    for _ in range(n_paths):
        if mode == "bootstrap_days":
            picks = rng.integers(0, n, size=path_days)
        else:  # shuffle, cycling if the path is longer than history
            reps = int(np.ceil(path_days / n))
            order = np.concatenate([rng.permutation(n) for _ in range(reps)])[:path_days]
            picks = order
        yield [days[i] for i in picks]


def flat_day_rate(tdf: pd.DataFrame, n_days_total: int) -> float:
    """Fraction of calendar trading days WITHOUT a setup (needed to model waiting time)."""
    if n_days_total <= 0:
        return 0.0
    days_with = tdf["trading_day"].nunique() if not tdf.empty else 0
    return max(0.0, 1.0 - days_with / n_days_total)
