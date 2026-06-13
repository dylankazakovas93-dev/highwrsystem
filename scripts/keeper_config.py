"""CANONICAL REFERENCE — the locked "open-drive continuation" config.
This single file fully specifies the strategy that the handoff asks Codex to verify.
Run: python scripts/keeper_config.py   (expects data/NQ_continuous.parquet)

EXACT RULES
-----------
Instrument : NQ continuous front-month, 1-min OHLCV, tz America/New_York,
             CME trading day rolls at 18:00 ET.
ATR        : SIMPLE moving average (NOT Wilder) of True Range over 14 bars,
             TR = max(H-L, |H-prevC|, |L-prevC|). Computed on resampled bars
             (label='right', closed='right'). Take each trading day's LAST value,
             then use the PRIOR trading day's value (shift 1) => no lookahead.
             Gate uses 30-min ATR; target uses 15-min ATR.
Entry      : at 11:00 ET, move = close(11:00 bar) - open(09:30 RTH bar).
             Trade only if |move| >= 1.5 * ATR30_prior.
             Direction = sign(move)  (CONTINUATION: long if up, short if down).
             Fill on the NEXT bar's open (11:01) + 1 tick slippage against you.
Target     : TP_dist = 0.40 * ATR15_prior. TP fills at the limit price.
Stop       : stop_dist = TP_dist / 0.2  (== 5*TP_dist == 2.0*ATR15), capped at 200 pts.
             Fills at stop price, or worse (at the bar open) on a gap, +1 tick slippage.
Timed exit : if neither TP nor stop hit within 90 minutes of entry, exit at market
             (bar close +-1 tick). Hard backstop flat at 15:55 ET.
Both-touch : a bar that hits BOTH stop and TP counts as the STOP (loss). [conservative]
Frequency  : exactly one trade per trading day.
Costs      : $5.00 round-turn commission; NQ point value $20; slippage 1 tick = 0.25 pt/side.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill

# ---- locked parameters ----
TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP_TICKS = 1
SLIP = SLIP_TICKS * TICK
SNAPSHOT = "11:00"
GATE_ATR_FREQ, GATE_MULT = "30min", 1.5
TP_ATR_FREQ, TP_MULT = "15min", 0.40
RR = 0.2                 # stop_dist = TP_dist / RR
STOP_CAP_PTS = 200
MAX_HOLD_MIN = 90
FLAT = "15:55"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def atr_prior(df, freq, period=14):
    g = df.resample(freq, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    pc = g["close"].shift(1)
    tr = pd.concat([g["high"] - g["low"], (g["high"] - pc).abs(), (g["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=max(8, period // 2)).mean()   # SMA of TR, not Wilder
    td = add_trading_day(g.assign(_tr=tr), 18)["trading_day"]
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()  # prior-day value


def backtest(df):
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    idx = df.index; mins = (idx.hour * 60 + idx.minute).to_numpy()
    bd = np.array([d.date() for d in idx])
    dcs, _ = pd.factorize(df["trading_day"])
    gate_atr = atr_prior(df, GATE_ATR_FREQ); tp_atr = atr_prior(df, TP_ATR_FREQ)
    sh, sm = map(int, SNAPSHOT.split(":")); snm = sh * 60 + sm
    fh, fm = map(int, FLAT.split(":")); fl = fh * 60 + fm
    rth = 9 * 60 + 30
    rows = []
    for d in np.unique(dcs):
        pos = np.nonzero(dcs == d)[0]
        td = df["trading_day"].iloc[pos[0]]
        ga, ta = gate_atr.get(td, np.nan), tp_atr.get(td, np.nan)
        if not (np.isfinite(ga) and ga > 0 and np.isfinite(ta) and ta > 0):
            continue
        dm = mins[pos]; dd = bd[pos]
        op = pos[(dd == td) & (dm >= rth) & (dm < 16 * 60)]
        if len(op) == 0:
            continue
        rth_open = o[op[0]]
        sp = pos[(dd == td) & (dm >= snm) & (dm < 16 * 60)]
        if len(sp) == 0:
            continue
        si = sp[0]; move = c[si] - rth_open
        if abs(move) < GATE_MULT * ga:
            continue
        s = int(np.sign(move)); ei = si + 1
        fp = pos[(dd == td) & (dm >= fl)]
        eod_i = fp[0] if len(fp) else pos[-1]
        flat_i = min(eod_i, ei + MAX_HOLD_MIN)
        if ei >= flat_i:
            continue
        entry = market_fill(s, o[ei], SLIP)
        tp_d = TP_MULT * ta
        sl_d = min(tp_d / RR, STOP_CAP_PTS)
        tp = entry + s * tp_d; stop = entry - s * sl_d
        bx = run_bracket(s, ei, entry, stop, tp, o, h, l, c, flat_i,
                         TICK, SLIP, tp_through_ticks=1, both_touch="loss")
        rows.append({"y": td.year, "pnl": s * (bx.exit_px - entry) * PV - COMM,
                     "tp": tp_d, "sl": sl_d, "reason": bx.reason})
    return pd.DataFrame(rows)


def pf(p):
    gw, gl = p[p > 0].sum(), -p[p < 0].sum()
    return gw / gl if gl > 0 else float("inf")


if __name__ == "__main__":
    df = add_trading_day(load_ohlcv("data/NQ_continuous.parquet"), 18)
    t = backtest(filter_period(df, "2020-01-01", "2026-12-31"))
    print(__doc__.split("EXACT RULES")[0].strip())
    print("VERIFICATION TARGETS (whichever engine you build, expect to land near these):")
    print(f"  {'Year':<5}{'Trades':>7}{'WinRate':>9}{'PF':>7}{'TPmed':>7}{'SLmed':>7}{'net$':>9}")
    for y in YEARS:
        s = t[t.y == y]
        if len(s) == 0:
            continue
        print(f"  {y:<5}{len(s):>7}{s.pnl.gt(0).mean()*100:>8.1f}%{pf(s.pnl):>7.2f}"
              f"{s.tp.median():>7.1f}{s.sl.median():>7.1f}{s.pnl.sum():>9.0f}")
    print(f"  {'ALL':<5}{len(t):>7}{t.pnl.gt(0).mean()*100:>8.1f}%{pf(t.pnl):>7.2f}"
          f"{t.tp.median():>7.1f}{t.sl.median():>7.1f}{t.pnl.sum():>9.0f}")
    print(f"\n  {len(t)} trades | {len(t)/7:.0f}/yr | {t.pnl.gt(0).mean()*100:.1f}% WR | "
          f"PF {pf(t.pnl):.2f} | exp ${t.pnl.mean():.0f}/trade | net ${t.pnl.sum():.0f}")
    print(f"  exit-reason mix: {t['reason'].value_counts().to_dict()}")
