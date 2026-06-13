"""TP-basis + time-exit study. Locked: 11:00, gate 1.5x30m ATR, RR 0.2 (stop=min(TP/0.2,200)).
Part A: find a TP basis giving ~15pt in 2025/26 (smaller TP -> higher WR).
Part B: hard time-exit after N minutes (exit at market) vs hold-to-15:55."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
GATE, RR, CAP = 1.5, 0.2, 200


def atr_prior(df, freq):
    g = df.resample(freq, label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()


def run(df, gate_atr, tp_atr, tp_mult, max_hold_min=None, snapshot='11:00', flat='15:55'):
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
        ga = gate_atr.get(td, np.nan); ta = tp_atr.get(td, np.nan)
        if not (np.isfinite(ga) and ga > 0 and np.isfinite(ta) and ta > 0):
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
        if abs(mv) < GATE * ga:
            continue
        s = int(np.sign(mv)); ei = si + 1
        fp = pos[(dd == td) & (dm >= fl)]
        eod_i = fp[0] if len(fp) else pos[-1]
        flat_i = eod_i if max_hold_min is None else min(eod_i, ei + max_hold_min)
        if ei >= flat_i:
            continue
        entry = market_fill(s, o[ei], SLIP)
        tp_d = tp_mult * ta; sl_d = min(tp_d / RR, CAP)
        tp = entry + s * tp_d; stop = entry - s * sl_d
        bx = run_bracket(s, ei, entry, stop, tp, o, h, l, c, flat_i, TICK, SLIP, 1, 'loss')
        rows.append({'y': td.year, 'pnl': s * (bx.exit_px - entry) * PV - COMM,
                     'sl': sl_d, 'tp': tp_d, 'reason': bx.reason})
    return pd.DataFrame(rows)


def pf(p):
    gw = p[p > 0].sum(); gl = -p[p < 0].sum()
    return gw / gl if gl > 0 else float('inf')


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
dall = filter_period(df, '2020-01-01', '2026-12-31')
A = {'15m': atr_prior(dall, '15min'), '30m': atr_prior(dall, '30min'), '60m': atr_prior(dall, '60min')}
gate_atr = A['30m']

print("PART A — TP basis (hold to 15:55). Goal: ~15pt TP in 2025/26, watch WR.")
print("=" * 92)
print(f"  {'TP basis':<16}{'TP25med':>8}{'TP26med':>8} | " + " ".join(f"{y:>5}" for y in YEARS) + f" | {'ALLwr':>6}{'PF':>6}")
print("  " + "-" * 84)
tp_defs = [('0.50x ATR30', '30m', 0.50), ('0.30x ATR30', '30m', 0.30), ('0.25x ATR30', '30m', 0.25),
           ('0.50x ATR15', '15m', 0.50), ('0.40x ATR15', '15m', 0.40)]
for name, tf, m in tp_defs:
    t = run(dall, gate_atr, A[tf], m)
    wr = {y: t[t.y == y].pnl.gt(0).mean() * 100 for y in YEARS if len(t[t.y == y])}
    tp25 = t[t.y == 2025].tp.median(); tp26 = t[t.y == 2026].tp.median()
    cells = " ".join(f"{wr.get(y, float('nan')):>5.1f}" for y in YEARS)
    print(f"  {name:<16}{tp25:>8.1f}{tp26:>8.1f} | {cells} | {t.pnl.gt(0).mean()*100:>5.1f}%{pf(t.pnl):>6.2f}")

print("\nPART B — time-exit sweep on the ~15pt TP (0.25x ATR30). Exit at market after N min.")
print("=" * 92)
print(f"  {'max_hold':<10}{'Trades':>7}{'WR':>7}{'PF':>6}{'exp$':>7}{'net$':>9} | " + " ".join(f"{y:>5}" for y in YEARS))
print("  " + "-" * 84)
for mh in [30, 45, 60, 90, 120, None]:
    t = run(dall, gate_atr, A['30m'], 0.25, max_hold_min=mh)
    wr = {y: t[t.y == y].pnl.gt(0).mean() * 100 for y in YEARS if len(t[t.y == y])}
    cells = " ".join(f"{wr.get(y, float('nan')):>5.1f}" for y in YEARS)
    lab = 'EOD(15:55)' if mh is None else f'{mh}min'
    print(f"  {lab:<10}{len(t):>7}{t.pnl.gt(0).mean()*100:>6.1f}%{pf(t.pnl):>6.2f}{t.pnl.mean():>7.0f}{t.pnl.sum():>9.0f} | {cells}")
