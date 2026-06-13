#!/usr/bin/env python3
"""
Verification backtest for the Camarilla 9AM strategy described in message.txt.

Data: Databento GLBX.MDP3 NQ.FUT, ohlcv-1m, 2020-01 .. 2026-06 (CSV, zstd).
We reconstruct a daily front-month (highest-volume outright contract per day),
convert UTC->America/New_York, build 4h candles for the filters, and simulate
the mechanical rules exactly as written.
"""
import io, re, glob
import zstandard
import numpy as np
import pandas as pd

FILES = sorted(glob.glob("data/**/*.ohlcv-1m.csv.zst", recursive=True))
OUTRIGHT = re.compile(r"^NQ[FGHJKMNQUVXZ]\d$")  # exclude spreads like NQZ0-NQH1

BLACKOUT = {
    "2025-02-01","2025-02-03",
    "2025-04-02","2025-04-03","2025-04-04",
    "2025-04-07","2025-04-08","2025-04-09",
}

SM = 0.25   # stop multiplier
TM = 3.00   # target multiplier
CAM = 1.1

def load():
    frames = []
    dctx = zstandard.ZstdDecompressor()
    for f in FILES:
        with open(f, "rb") as fh:
            txt = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
            df = pd.read_csv(txt, usecols=["ts_event","open","high","low","close","volume","symbol"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["symbol"].str.match(OUTRIGHT)].copy()
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.sort_values("ts")
    # convert to ET
    df["et"] = df["ts"].dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.normalize().dt.tz_localize(None).dt.date
    return df

def pick_front_month(df):
    # For each calendar (ET) day pick the single outright contract with the
    # highest total volume that day; keep only its bars.
    vol = df.groupby(["date","symbol"])["volume"].sum().reset_index()
    front = vol.sort_values("volume").groupby("date").tail(1)[["date","symbol"]]
    front = front.rename(columns={"symbol":"front"})
    m = df.merge(front, on="date")
    m = m[m["symbol"] == m["front"]].copy()
    return m

def four_hour_candles(df):
    # Build a continuous stream of 4h candles on the LOCAL ET clock
    # (blocks starting 00,04,08,12,16,20) by grouping on (date, hour//4).
    # Doing this explicitly avoids DST drift that a fixed-freq resample suffers.
    d = df.copy()
    d["edate"] = d["et"].dt.tz_localize(None).dt.normalize()
    d["block"] = (d["et"].dt.hour // 4) * 4
    d = d.sort_values("et")
    grp = d.groupby(["edate","block"])
    cand = grp.agg(open=("open","first"), high=("high","max"),
                   low=("low","min"), close=("close","last"),
                   volume=("volume","sum")).reset_index()
    cand = cand.sort_values(["edate","block"]).reset_index(drop=True)
    cand["range"] = cand["high"] - cand["low"]
    # 20-period rolling median over the continuous 4h stream, ending at the
    # PRIOR candle -> shift(1).
    cand["med_vol"] = cand["volume"].shift(1).rolling(20).median()
    cand["med_rng"] = cand["range"].shift(1).rolling(20).median()
    cand["hour"] = cand["block"]
    cand = cand.set_index("edate")
    return cand

_CACHE = {}
def _data():
    if not _CACHE:
        df = load(); df = pick_front_month(df)
        _CACHE["df"] = df
        _CACHE["cand"] = four_hour_candles(df)
    return _CACHE["df"], _CACHE["cand"]

def run(intrabar="pessimistic", manage_from_next=False, use_filters=True, diag=None):
    df, cand = _data()

    # the 04:00-08:00 ET candle is the one whose start hour == 4
    london = cand[cand["hour"] == 4].copy()

    # index 1m bars by date for fast slicing of the entry/management window
    df = df.sort_values("et")
    df["d"] = df["et"].dt.tz_localize(None).dt.date
    df["hm"] = df["et"].dt.hour*60 + df["et"].dt.minute
    by_day = {d: g for d, g in df.groupby("d")}

    cnt = dict(days=0, after_blackout=0, after_warmup=0, after_vol=0,
               after_spike=0, has_window=0, triggered=0, ambiguous=0)
    trades = []
    for ts, row in london.iterrows():
        cnt["days"] += 1
        d = ts.date()
        if str(d) in BLACKOUT:
            continue
        cnt["after_blackout"] += 1
        if pd.isna(row["med_vol"]) or pd.isna(row["med_rng"]):
            continue
        cnt["after_warmup"] += 1
        # filters
        if use_filters and not (row["volume"] > row["med_vol"]):    # vol filter
            continue
        cnt["after_vol"] += 1
        if use_filters and row["range"] > 1.8 * row["med_rng"]:      # spike filter
            continue
        cnt["after_spike"] += 1

        close = row["close"]; rng = row["range"]
        if rng <= 0:
            continue
        R3 = close + rng*CAM/4
        S3 = close - rng*CAM/4
        Base = rng*CAM/4

        g = by_day.get(d)
        if g is None:
            continue
        win = g[(g["hm"] >= 9*60+30) & (g["hm"] <= 9*60+59)]
        if win.empty:
            continue
        cnt["has_window"] += 1

        # find first trigger
        side = None; entry_idx = None
        for i, b in win.iterrows():
            hi_long = b["high"] >= R3
            lo_short = b["low"] <= S3
            if hi_long and lo_short:
                side = "ambiguous"; entry_idx = i; break
            if hi_long:
                side = "long"; entry_idx = i; break
            if lo_short:
                side = "short"; entry_idx = i; break
        if side == "ambiguous":
            cnt["ambiguous"] += 1
            continue
        if side is None:
            continue
        cnt["triggered"] += 1

        if side == "long":
            entry = R3; stop = R3 - SM*Base; target = R3 + TM*Base
        else:
            entry = S3; stop = S3 + SM*Base; target = S3 - TM*Base

        # manage from entry bar through 11:59 ET (force flat at 12:00)
        mgmt = g[(g["et"] >= win.loc[entry_idx,"et"]) & (g["hm"] <= 11*60+59)]
        outcome = None; exit_px = None
        for j,(i,b) in enumerate(mgmt.iterrows()):
            if j == 0 and manage_from_next:
                # entry bar only establishes the position; manage from next bar
                continue
            hit_stop = (b["low"] <= stop) if side=="long" else (b["high"] >= stop)
            hit_tgt  = (b["high"] >= target) if side=="long" else (b["low"] <= target)
            if hit_stop and hit_tgt:
                # same-bar both: tie-break
                if intrabar == "pessimistic":
                    outcome = "stop"; exit_px = stop
                else:
                    outcome = "target"; exit_px = target
                break
            if hit_stop:
                outcome="stop"; exit_px=stop; break
            if hit_tgt:
                outcome="target"; exit_px=target; break
        if outcome is None:
            # forced flat at 12:00 -> exit at last managed bar close
            exit_px = mgmt.iloc[-1]["close"]
            outcome = "flat"

        if side == "long":
            pnl_pts = exit_px - entry
        else:
            pnl_pts = entry - exit_px
        risk_pts = SM*Base
        r_mult = pnl_pts / risk_pts
        trades.append(dict(date=d, dow=ts.day_name() if hasattr(ts,'day_name') else pd.Timestamp(d).day_name(),
                           side=side, outcome=outcome, base=Base, entry=entry,
                           pnl_pts=pnl_pts, r=r_mult))
    if diag is not None:
        diag.update(cnt)
    return pd.DataFrame(trades)

def report(t, label):
    print(f"\n===== {label} =====")
    n=len(t)
    if n==0:
        print("no trades"); return
    wins = (t["r"]>0).sum()
    wr = wins/n
    gross_win = t.loc[t["r"]>0,"r"].sum()
    gross_loss = -t.loc[t["r"]<0,"r"].sum()
    pf = gross_win/gross_loss if gross_loss>0 else float('inf')
    mean_r=t["r"].mean(); std_r=t["r"].std()
    sharpe_trade = mean_r/std_r if std_r>0 else float('nan')
    # annualize per-trade sharpe by trades/year
    years = (max(t["date"])-min(t["date"])).days/365.25
    tpy = n/years if years>0 else float('nan')
    sharpe_ann = sharpe_trade*np.sqrt(tpy) if tpy==tpy else float('nan')
    # equity / max drawdown in R
    eq = t["r"].cumsum()
    dd = (eq.cummax()-eq).max()
    print(f"trades={n}  span={min(t['date'])}..{max(t['date'])} ({years:.1f}y, {tpy:.0f}/yr)")
    print(f"win rate      = {wr*100:.1f}%  ({wins}/{n})")
    print(f"outcomes      = {t['outcome'].value_counts().to_dict()}")
    print(f"profit factor = {pf:.2f}")
    print(f"total R       = {t['r'].sum():.1f}   mean R/trade = {mean_r:.3f}")
    print(f"sharpe/trade  = {sharpe_trade:.3f}   annualized = {sharpe_ann:.2f}")
    print(f"max drawdown  = {dd:.1f} R")
    by = t.groupby("dow")["r"].agg(["count", lambda x:(x>0).mean()])
    by.columns=["n","wr"]
    order=["Monday","Tuesday","Wednesday","Thursday","Friday"]
    by=by.reindex([d for d in order if d in by.index])
    print("by day of week:")
    for d,r in by.iterrows():
        print(f"   {d:9s} {r['wr']*100:5.1f}%  (n={int(r['n'])})")

if __name__ == "__main__":
    d = {}
    t1 = run("pessimistic", manage_from_next=False, diag=d)
    report(t1, "A: manage-from-entry-bar, pessimistic ties (strictest)")
    print("FUNNEL:", d)

    t2 = run("pessimistic", manage_from_next=True)
    report(t2, "B: manage-from-NEXT-bar, pessimistic ties (fair entry)")

    t3 = run("optimistic", manage_from_next=True)
    report(t3, "C: manage-from-NEXT-bar, optimistic ties (lenient)")

    dnf = {}
    t4 = run("pessimistic", manage_from_next=True, use_filters=False, diag=dnf)
    report(t4, "D: NO filters, manage-from-NEXT-bar (trade-count check)")
    print("FUNNEL(no filters):", dnf)

    t2.to_csv("work/trades_main.csv", index=False)
    print("\nsaved work/trades_main.csv")
