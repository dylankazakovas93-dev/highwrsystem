"""LOCKED open-drive continuation config — clean year-by-year readout.

Rule: at the snapshot time, measure move = close(snapshot) - open(09:30 RTH).
If |move| >= gate * ATR30(prior day), enter IN THAT DIRECTION on the next bar.
TP = tp_mult * ATR30. Stop = min(10*TP, max_sl_pts). Flat at 15:55. One trade/day.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def atr30_prior(df):
    g = df.resample('30min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1)


def run(df, snapshot, gate=2.0, tp_mult=0.5, max_sl_pts=200, direction='cont', flat='15:55'):
    o = df['open'].to_numpy(float); h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float); c = df['close'].to_numpy(float)
    idx = df.index; mins = (idx.hour * 60 + idx.minute).to_numpy()
    bd = np.array([d.date() for d in idx])
    dcs, _ = pd.factorize(df['trading_day'])
    am = atr30_prior(df).to_dict()
    sh, sm = map(int, snapshot.split(':')); snm = sh * 60 + sm
    fh, fm = map(int, flat.split(':')); fl = fh * 60 + fm
    rth = 570
    rows = []
    for d in np.unique(dcs):
        pos = np.nonzero(dcs == d)[0]
        td = df['trading_day'].iloc[pos[0]]
        atr = am.get(td, np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue
        dm = mins[pos]; dd = bd[pos]
        op = pos[(dd == td) & (dm >= rth) & (dm < 960)]
        if len(op) == 0:
            continue
        o930 = o[op[0]]
        sp = pos[(dd == td) & (dm >= snm) & (dm < 960)]
        if len(sp) == 0:
            continue
        si = sp[0]; mv = c[si] - o930
        if abs(mv) < gate * atr:
            continue
        s = int(np.sign(mv)) if direction == 'cont' else int(-np.sign(mv))
        ei = si + 1
        fp = pos[(dd == td) & (dm >= fl)]
        fi = fp[0] if len(fp) else pos[-1]
        if ei >= fi:
            continue
        entry = market_fill(s, o[ei], SLIP)
        tp_d = tp_mult * atr
        sl_d = min(tp_d / 0.1, max_sl_pts)
        tp = entry + s * tp_d; stop = entry - s * sl_d
        bx = run_bracket(s, ei, entry, stop, tp, o, h, l, c, fi, TICK, SLIP, 1, 'loss')
        rows.append({'y': td.year, 'pnl': s * (bx.exit_px - entry) * PV - COMM})
    return pd.DataFrame(rows)


def pf(p):
    gw = p[p > 0].sum(); gl = -p[p < 0].sum()
    return gw / gl if gl > 0 else float('inf')


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
dall = filter_period(df, '2020-01-01', '2026-12-31')

for snap in ['11:00', '12:00', '13:00']:
    t = run(dall, snap, gate=2.0, tp_mult=0.5, max_sl_pts=200)
    print("\n" + "=" * 64)
    print(f"  OPEN-DRIVE CONTINUATION  |  snapshot {snap}  |  gate 2.0xATR")
    print(f"  TP 0.5xATR  |  stop 200 pts  |  flat 15:55  |  1 trade/day  |  1t slip")
    print("=" * 64)
    print(f"  {'Year':<6}{'Trades':>8}{'WinRate':>10}{'PF':>8}{'Exp/trd':>10}{'Net$':>11}")
    print("  " + "-" * 51)
    for y in YEARS:
        s = t[t.y == y]
        if len(s) == 0:
            continue
        print(f"  {y:<6}{len(s):>8}{s.pnl.gt(0).mean()*100:>9.1f}%{pf(s.pnl):>8.2f}"
              f"{s.pnl.mean():>10.0f}{s.pnl.sum():>11.0f}")
    print("  " + "-" * 51)
    print(f"  {'ALL':<6}{len(t):>8}{t.pnl.gt(0).mean()*100:>9.1f}%{pf(t.pnl):>8.2f}"
          f"{t.pnl.mean():>10.0f}{t.pnl.sum():>11.0f}")
