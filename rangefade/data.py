"""OHLCV loading, timezone normalization, trading-day assignment, quality reporting.

Accepts flexible vendor CSV/parquet layouts and produces a canonical frame:
tz-aware DatetimeIndex in market tz (ET), columns open/high/low/close/volume,
plus a CME-convention `trading_day` (18:00 ET roll).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataCfg

_TS_ALIASES = ["timestamp", "datetime", "date_time", "time", "ts", "date"]
_COL_ALIASES = {
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "last", "settle"],
    "volume": ["volume", "vol", "v", "totalvolume", "up_volume"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _build_timestamp(df: pd.DataFrame, cfg: DataCfg) -> pd.Series:
    cols = set(df.columns)
    # split date + time columns (common vendor export)
    if "date" in cols and "time" in cols and not any(a in cols for a in ("timestamp", "datetime", "date_time")):
        raw = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        return pd.to_datetime(raw, format=cfg.timestamp_format)
    for alias in _TS_ALIASES:
        if alias in cols:
            s = df[alias]
            if pd.api.types.is_numeric_dtype(s):  # epoch stamps
                unit = "ms" if float(s.iloc[0]) > 1e11 else "s"
                return pd.to_datetime(s, unit=unit, utc=True)
            return pd.to_datetime(s, format=cfg.timestamp_format)
    raise ValueError(f"no timestamp column found; columns = {list(df.columns)}")


def load_ohlcv(path: str | Path, cfg: DataCfg | None = None) -> pd.DataFrame:
    """Load 1-min OHLCV into the canonical ET-indexed frame."""
    cfg = cfg or DataCfg()
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
        if not isinstance(raw.index, pd.RangeIndex):
            raw = raw.reset_index()
    else:
        raw = pd.read_csv(path)
    raw = _normalize_columns(raw)

    ts = _build_timestamp(raw, cfg)
    out = pd.DataFrame(index=pd.DatetimeIndex(ts))
    for canon, aliases in _COL_ALIASES.items():
        col = next((a for a in aliases if a in raw.columns), None)
        if col is None:
            if canon == "volume":
                out["volume"] = 0.0
                continue
            raise ValueError(f"missing required column '{canon}'; columns = {list(raw.columns)}")
        out[canon] = pd.to_numeric(raw[col].values, errors="coerce")

    # timezone normalization
    if out.index.tz is None:
        src = cfg.tz_input or cfg.tz_market
        out.index = out.index.tz_localize(src, ambiguous="NaT", nonexistent="NaT")
        n_bad = int(out.index.isna().sum())
        if n_bad:
            print(f"[data] dropped {n_bad} bars with ambiguous/nonexistent local times (DST)")
            out = out[~out.index.isna()]
    out.index = out.index.tz_convert(cfg.tz_market)
    out.index.name = "ts"

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_index()
    dup = out.index.duplicated(keep="last")
    if dup.any():
        print(f"[data] dropped {int(dup.sum())} duplicate timestamps (kept last)")
        out = out[~dup]

    bad = (out["high"] < out[["open", "close", "low"]].max(axis=1)) | (
        out["low"] > out[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        print(f"[data] dropped {int(bad.sum())} bars with inconsistent OHLC")
        out = out[~bad]
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    return out


def add_trading_day(df: pd.DataFrame, day_roll_hour: int = 18) -> pd.DataFrame:
    """CME convention: bars at/after the roll hour belong to the NEXT calendar trading day."""
    df = df.copy()
    shift_hours = 24 - day_roll_hour
    df["trading_day"] = (df.index + pd.Timedelta(hours=shift_hours)).date
    return df


def data_quality_report(df: pd.DataFrame, bar_minutes: int = 1) -> dict:
    idx = df.index
    days = pd.Series(df["trading_day"]) if "trading_day" in df else pd.Series((idx + pd.Timedelta(hours=6)).date, index=idx)
    per_day = df.groupby(days.values).size()
    expected = 23 * 60 // bar_minutes  # full futures day (Globex 23h)
    diffs = idx.to_series().diff().dropna().dt.total_seconds() / 60.0
    report = {
        "rows": int(len(df)),
        "start": str(idx[0]) if len(df) else None,
        "end": str(idx[-1]) if len(df) else None,
        "trading_days": int(per_day.size),
        "median_bars_per_day": float(per_day.median()) if per_day.size else 0.0,
        "expected_bars_per_full_day": expected,
        "days_below_50pct_coverage": int((per_day < 0.5 * expected).sum()),
        "max_intraday_gap_minutes": float(diffs[diffs < 60 * 30].max()) if len(diffs) else None,
        "zero_volume_share": float((df["volume"] == 0).mean()) if len(df) else 0.0,
        "tz": str(idx.tz),
    }
    return report


def filter_period(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Inclusive date filtering on trading_day."""
    out = df
    if start:
        out = out[out["trading_day"] >= pd.Timestamp(start).date()]
    if end:
        out = out[out["trading_day"] <= pd.Timestamp(end).date()]
    return out
