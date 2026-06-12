"""Build a continuous front-month NQ 1-min series from Databento GLBX parent-symbology files.

Inputs: data/raw/*.ohlcv-1m.csv.zst (CSV) and *.ohlcv-1m.dbn.zst (DBN), NQ.FUT parent
(all outrights + spreads). Output: data/NQ_continuous.parquet with ts (UTC), OHLCV,
plus a roll calendar at data/NQ_rolls.csv.

Method: per CME trading day (18:00 ET roll), keep the outright contract with the
highest total volume; contract order is enforced forward-only (no rolling back).
Prices are NOT back-adjusted — intraday features never span the roll, and the few
roll-gap days are handled by the day-move filter rather than by rewriting history.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data")
OUTRIGHT = re.compile(r"^NQ[HMUZ]\d$")
MONTH = {"H": 3, "M": 6, "U": 9, "Z": 12}


def contract_key(symbol: str, ref_year: int) -> tuple[int, int]:
    m = MONTH[symbol[2]]
    d = int(symbol[3])
    y = ref_year - 1 + ((d - (ref_year - 1) % 10) % 10)
    return (y, m)


def load_csv_zst(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
    )
    return df


def load_dbn_zst(path: Path) -> pd.DataFrame:
    from databento import DBNStore

    store = DBNStore.from_file(path)
    df = store.to_df(map_symbols=True)  # index ts_event UTC, prices floats
    df = df.reset_index()[["ts_event", "open", "high", "low", "close", "volume", "symbol"]]
    return df


def to_front_month(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["symbol"].astype(str).str.fullmatch(OUTRIGHT)].copy()
    ts = pd.to_datetime(df["ts_event"], utc=True)
    et = ts.dt.tz_convert("America/New_York")
    df["ts"] = ts
    df["td"] = (et + pd.Timedelta(hours=6)).dt.date

    vol = df.groupby(["td", "symbol"], observed=True)["volume"].sum().reset_index()
    picks = []
    prev_key = None
    prev_sym = None
    for td, grp in vol.groupby("td", sort=True):
        grp = grp.sort_values("volume", ascending=False)
        chosen = None
        for _, row in grp.iterrows():
            key = contract_key(row["symbol"], pd.Timestamp(td).year)
            if prev_key is None or key >= prev_key:
                chosen = (row["symbol"], key)
                break
        if chosen is None:  # all candidates behind the current front: keep previous
            chosen = (prev_sym, prev_key)
        picks.append({"td": td, "symbol": chosen[0]})
        prev_sym, prev_key = chosen
    front = pd.DataFrame(picks)

    rolls = front[front["symbol"] != front["symbol"].shift(1)].copy()
    merged = df.merge(front, on=["td", "symbol"], how="inner")
    out = merged[["ts", "open", "high", "low", "close", "volume", "symbol"]].sort_values("ts")
    return out, rolls


def main() -> int:
    frames = []
    for path in sorted(RAW.glob("*.ohlcv-1m.csv.zst")):
        print(f"[build] reading {path.name}")
        frames.append(load_csv_zst(path))
    for path in sorted(RAW.glob("*.ohlcv-1m.dbn.zst")):
        print(f"[build] decoding {path.name}")
        frames.append(load_dbn_zst(path))
    if not frames:
        print("no input files under data/raw/", file=sys.stderr)
        return 1
    raw = pd.concat(frames, ignore_index=True)
    print(f"[build] {len(raw):,} raw rows (all instruments)")
    cont, rolls = to_front_month(raw)
    cont = cont.drop_duplicates(subset="ts", keep="last")
    print(f"[build] {len(cont):,} continuous front-month bars "
          f"({cont['ts'].iloc[0]} .. {cont['ts'].iloc[-1]})")
    print(f"[build] {len(rolls)} contract segments (rolls):")
    print(rolls.to_string(index=False))
    OUT.mkdir(exist_ok=True)
    cont.drop(columns=["symbol"]).to_parquet(OUT / "NQ_continuous.parquet", index=False)
    rolls.to_csv(OUT / "NQ_rolls.csv", index=False)
    print(f"[build] wrote {OUT/'NQ_continuous.parquet'} and {OUT/'NQ_rolls.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
