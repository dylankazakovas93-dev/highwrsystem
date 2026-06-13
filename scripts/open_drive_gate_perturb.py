"""Gate perturbation test — 11:00 snapshot locked, 200pt stop cap locked.
Vary entry-gate ATR timeframe (15m/30m/60m) and multiple. TP = 0.5*ATR30 (locked).
Reports overall WR/PF/freq AND the worst single-year WR (robustness under perturbation)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def atr_prior(df, freq):
    g = df.resample(freq, label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()


def run(df, gate_map, gate_mult, atr30_map, snapshot='11:00', tp_mult=0.5, max_sl=200, flat='15:55'):
    o = df['open'].to_numpy(float); h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float); c = df['close'].to_numpy(float)
    idx = df.index; mins = (idx.hour * 60 + idx.minute).to_numpy()
    bd = np.array([d.date() for d in idx])
    dcs, _ = pd.factorize(df['trading_day'])
    sh, sm = map(int, snapshot.split(':')); snm = sh * 60 + sm
    fh, fm = map(int, flat.split(':')); fl = fh * 60 + fm
    rth = 570
    rows = []
    for d in np.unique(dcs):
        pos = np.nonzero(dcs == d)[0]
        td = df['trading_day'].iloc[pos[0]]
        g_atr = gate_map.get(td, np.nan); t_atr = atr30_map.get(td, np.nan)
        if not (np.isfinite(g_atr) and g_atr > 0 and np.isfinite(t_atr) and t_atr > 0):
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
        if abs(mv) < gate_mult * g_atr:
            continue
        s = int(np.sign(mv)); ei = si + 1
        fp = pos[(dd == td) & (dm >= fl)]
        fi = fp[0] if len(fp) else pos[-1]
        if ei >= fi:
            continue
        entry = market_fill(s, o[ei], SLIP)
        tp_d = tp_mult * t_atr; sl_d = min(tp_d / 0.1, max_sl)
        tp = entry + s * tp_d; stop = entry - s * sl_d
        bx = run_bracket(s, ei, entry, stop, tp, o, h, l, c, fi, TICK, SLIP, 1, 'loss')
        rows.append({'y': td.year, 'pnl': s * (bx.exit_px - entry) * PV - COMM})
    return pd.DataFrame(rows)


def pf(p):
    gw = p[p > 0].sum(); gl = -p[p < 0].sum()
    return gw / gl if gl > 0 else float('inf')


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
dall = filter_period(df, '2020-01-01', '2026-12-31')
maps = {'15m': atr_prior(dall, '15min'), '30m': atr_prior(dall, '30min'), '60m': atr_prior(dall, '60min')}
atr30 = maps['30m']

for tf in ['15m', '30m', '60m']:
    print("\n" + "=" * 78)
    print(f"  GATE PERTURBATION — 11:00 snapshot, gate on {tf} ATR, TP 0.5xATR30, stop cap 200pt")
    print("=" * 78)
    print(f"  {'gate×':>6}{'Trades':>8}{'/yr':>5}{'WinRate':>10}{'PF':>7}{'Exp$':>7}{'minYrWR':>9}{'#yrs<80%':>9}")
    print("  " + "-" * 61)
    for gm in [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
        t = run(dall, maps[tf], gm, atr30)
        if len(t) < 20:
            print(f"  {gm:>6}{len(t):>8}  (thin)"); continue
        yr_wr = {y: t[t.y == y].pnl.gt(0).mean() for y in YEARS if len(t[t.y == y]) >= 8}
        minwr = min(yr_wr.values()) * 100
        below = sum(1 for v in yr_wr.values() if v < 0.80)
        print(f"  {gm:>6}{len(t):>8}{len(t)/7:>5.0f}{t.pnl.gt(0).mean()*100:>9.1f}%"
              f"{pf(t.pnl):>7.2f}{t.pnl.mean():>7.0f}{minwr:>8.1f}%{below:>9}")
