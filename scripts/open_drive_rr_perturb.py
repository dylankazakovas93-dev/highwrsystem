"""RR perturbation, WIN RATE only, year-by-year. Locked: 11:00 snapshot, gate 2.0x30m ATR,
TP 0.5xATR30, stop = min(TP/RR, 200pt cap). Vary RR (reward:risk rel. to TP).
Trade COUNT is identical across RR (RR only moves the stop), shown once."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
RRS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def atr_prior(df, freq):
    g = df.resample(freq, label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()


def run(df, atr30, rr, gate_mult=2.0, tp_mult=0.5, max_sl=200, snapshot='11:00', flat='15:55'):
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

base = run(dall, atr30, 0.1)
nyr = {y: len(base[base.y == y]) for y in YEARS}
print("LOCKED: 11:00 snapshot | gate 2.0x30m ATR | TP 0.5xATR30 | stop = min(TP/RR, 200pt)")
print("Trades/year (same for every RR): " + "  ".join(f"{y}:{nyr[y]}" for y in YEARS) + f"  | ALL:{len(base)}")
print("\nWIN RATE (%) by RR multiple, year-by-year")
print("=" * 86)
print(f"  {'RR':>5} | " + " ".join(f"{y:>5}" for y in YEARS) + f" | {'ALL':>6} {'minYr':>6}")
print("  " + "-" * 78)
for rr in RRS:
    t = run(dall, atr30, rr)
    wr = {y: t[t.y == y].pnl.gt(0).mean() * 100 for y in YEARS if len(t[t.y == y])}
    cells = " ".join(f"{wr.get(y, float('nan')):>5.1f}" for y in YEARS)
    allwr = t.pnl.gt(0).mean() * 100
    minwr = min(wr.values())
    # show effective stop (avg) to see where the 200 cap stops binding
    print(f"  {rr:>5} | {cells} | {allwr:>5.1f}% {minwr:>5.1f}%")
print("\n(avg stop distance in pts by RR — shows where the 200 cap stops binding)")
for rr in [0.1, 0.2, 0.3, 0.5, 0.8]:
    t = run(dall, atr30, rr)
    print(f"  RR {rr}: avg stop {t.sl.mean():>5.0f}pt  avg TP {t.tp.mean():>4.0f}pt")
