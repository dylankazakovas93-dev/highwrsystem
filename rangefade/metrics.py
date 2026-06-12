"""Performance metrics and segment breakdowns.

Ranking philosophy (pre-registered, see THESIS.md): win rate and its stability come
first, then losing-streak/downside boundedness, then slippage robustness. Net profit
is reported but never used as the primary ranking key.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .strategy import Trade


def trade_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["win"] = df["pnl_usd"] > 0
    df["year"] = df["entry_time"].dt.year
    df["month"] = df["entry_time"].dt.strftime("%Y-%m")
    df["hour"] = df["entry_time"].dt.hour
    return df


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def max_losing_streak(pnl: pd.Series) -> int:
    streak = best = 0
    for x in pnl:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    return best


def max_consecutive_losing_days(daily_pnl: pd.Series) -> int:
    return max_losing_streak(daily_pnl)


def max_drawdown(pnl: pd.Series) -> float:
    eq = pnl.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def summarize(tdf: pd.DataFrame) -> dict:
    """Full stats block for one trade set (single slippage scenario)."""
    if tdf.empty:
        return {"trades": 0}
    pnl = tdf["pnl_usd"]
    wins = tdf[pnl > 0]
    losses = tdf[pnl <= 0]
    daily = pnl.groupby(tdf["trading_day"]).sum()
    gross_win = wins["pnl_usd"].sum()
    gross_loss = -losses["pnl_usd"].sum()
    n = len(tdf)
    wr = len(wins) / n
    ci = wilson_ci(len(wins), n)
    top2_share = float("nan")
    if gross_win > 0:
        top2_share = float(wins["pnl_usd"].nlargest(2).sum() / max(gross_win, 1e-9))
    return {
        "trades": int(n),
        "win_rate": round(wr, 4),
        "win_rate_ci_low": round(ci[0], 4),
        "win_rate_ci_high": round(ci[1], 4),
        "profit_factor": round(float(gross_win / gross_loss), 3) if gross_loss > 0 else float("inf"),
        "net_usd": round(float(pnl.sum()), 2),
        "expectancy_usd": round(float(pnl.mean()), 2),
        "expectancy_points": round(float(tdf["pnl_points"].mean()), 3),
        "avg_win_usd": round(float(wins["pnl_usd"].mean()), 2) if len(wins) else 0.0,
        "avg_loss_usd": round(float(losses["pnl_usd"].mean()), 2) if len(losses) else 0.0,
        "win_loss_ratio": round(float(wins["pnl_usd"].mean() / abs(losses["pnl_usd"].mean())), 3)
        if len(wins) and len(losses) and losses["pnl_usd"].mean() != 0 else float("inf"),
        "max_drawdown_usd": round(max_drawdown(pnl), 2),
        "max_losing_streak": int(max_losing_streak(pnl)),
        "max_consec_losing_days": int(max_consecutive_losing_days(daily)),
        "best_day_usd": round(float(daily.max()), 2),
        "worst_day_usd": round(float(daily.min()), 2),
        "avg_duration_min": round(float(tdf["duration_min"].mean()), 1),
        "largest_loss_usd": round(float(pnl.min()), 2),
        "top2_win_share": round(top2_share, 3),
        "pct_target_inside": round(float(tdf["target_inside"].mean()), 4),
        "trading_days_with_trades": int(daily.size),
    }


def summarize_by(tdf: pd.DataFrame, key: str) -> pd.DataFrame:
    if tdf.empty:
        return pd.DataFrame()
    rows = []
    for val, sub in tdf.groupby(key, observed=True):
        row = {"segment": str(val)}
        row.update(summarize(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def regime_buckets(tdf: pd.DataFrame, col: str = "prior_day_atr") -> pd.DataFrame:
    """Terciles of prior-day ATR observed at entry: low / mid / high volatility regime."""
    if tdf.empty or tdf[col].dropna().empty:
        return pd.DataFrame()
    tdf = tdf.copy()
    try:
        tdf["regime"] = pd.qcut(tdf[col], 3, labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    return summarize_by(tdf, "regime")


def slippage_table(results: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """results: {slippage_ticks: trade_frame}. The kill test: does the edge survive 2 ticks?"""
    rows = []
    for ticks, tdf in sorted(results.items()):
        row = {"slippage_ticks": ticks}
        row.update(summarize(tdf))
        rows.append(row)
    return pd.DataFrame(rows)
