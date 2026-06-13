"""RR=0.2 fixed. (1) Gate sweep incl 1.2x to test 'lower gate just adds frequency, WR flat'.
(2) Frequency + avg/median SL & TP for 2025 and 2026. Locked: 11:00, TP 0.5xATR30,
stop = min(TP/0.2, 200) = min(2.5xATR30, 200)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
RR = 0.2


def atr_prior(df, freq):
    g = df.resample(freq, label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()


def run(df, atr30, gate_mult, rr=RR, tp_mult=0.5, max_sl=200, snapshot='11:00', flat='15:55'):
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
        atr = atr30.get(td, np.nan)
        if not (np.isfinite(atr) and atr > 0):
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
        if abs(mv) < gate_mult * atr:
            continue
        s = int(np.sign(mv)); ei = si + 1
        fp = pos[(dd == td) & (dm >= fl)]
        fi = fp[0] if len(fp) else pos[-1]
        if ei >= fi:
            continue
        entry = market_fill(s, o[ei], SLIP)
        tp_d = tp_mult * atr; sl_d = min(tp_d / rr, max_sl)
        tp = entry + s * tp_d; stop = entry - s * sl_d
        bx = run_bracket(s, ei, entry, stop, tp, o, h, l, c, fi, TICK, SLIP, 1, 'loss')
        rows.append({'y': td.year, 'pnl': s * (bx.exit_px - entry) * PV - COMM, 'sl': sl_d, 'tp': tp_d})
    return pd.DataFrame(rows)


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
dall = filter_period(df, '2020-01-01', '2026-12-31')
atr30 = atr_prior(dall, '30min')

print("RR=0.2 | 11:00 | TP 0.5xATR30 | stop min(2.5xATR30, 200)")
print("\n(1) GATE SWEEP — does a lower gate just add frequency without moving WR?")
print("=" * 84)
print(f"  {'gate×':>6}{'Trades':>8}{'/yr':>5}  | " + " ".join(f"{y:>5}" for y in YEARS) + f" | {'ALLwr':>6}{'minYr':>7}")
print("  " + "-" * 74)
for gm in [1.0, 1.2, 1.5, 1.75, 2.0, 2.5]:
    t = run(dall, atr30, gm)
    wr = {y: t[t.y == y].pnl.gt(0).mean() * 100 for y in YEARS if len(t[t.y == y]) >= 8}
    cells = " ".join(f"{wr.get(y, float('nan')):>5.1f}" for y in YEARS)
    print(f"  {gm:>6}{len(t):>8}{len(t)/7:>5.0f}  | {cells} | {t.pnl.gt(0).mean()*100:>5.1f}%{min(wr.values()):>6.1f}%")

print("\n(2) FREQUENCY + SL/TP distances (points) for 2025 & 2026, at gate 1.2 and 2.0")
print("=" * 84)
print(f"  {'gate×':>6}{'year':>6}{'trades':>8} | {'TP_avg':>7}{'TP_med':>7} | {'SL_avg':>7}{'SL_med':>7} | {'WR':>6}")
print("  " + "-" * 66)
for gm in [1.2, 2.0]:
    t = run(dall, atr30, gm)
    for y in [2025, 2026]:
        s = t[t.y == y]
        print(f"  {gm:>6}{y:>6}{len(s):>8} | {s.tp.mean():>7.1f}{s.tp.median():>7.1f} | "
              f"{s.sl.mean():>7.1f}{s.sl.median():>7.1f} | {s.pnl.gt(0).mean()*100:>5.1f}%")
