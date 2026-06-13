"""LOCKED CONFIG CARD: 11:00 snapshot, gate 1.5x30m ATR, TP 0.5xATR30, RR 0.2
(stop = min(2.5xATR30, 200pt)), flat 15:55, 1 trade/day, 1-tick slippage.
Year-by-year: frequency, win rate, PF, and avg/median SL & TP (points)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
GATE, RR, TPM, CAP = 1.5, 0.2, 0.5, 200


def atr_prior(df, freq):
    g = df.resample(freq, label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()


def run(df, atr30, snapshot='11:00', flat='15:55'):
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
        if abs(mv) < GATE * atr:
            continue
        s = int(np.sign(mv)); ei = si + 1
        fp = pos[(dd == td) & (dm >= fl)]
        fi = fp[0] if len(fp) else pos[-1]
        if ei >= fi:
            continue
        entry = market_fill(s, o[ei], SLIP)
        tp_d = TPM * atr; sl_d = min(tp_d / RR, CAP)
        tp = entry + s * tp_d; stop = entry - s * sl_d
        bx = run_bracket(s, ei, entry, stop, tp, o, h, l, c, fi, TICK, SLIP, 1, 'loss')
        rows.append({'y': td.year, 'pnl': s * (bx.exit_px - entry) * PV - COMM, 'sl': sl_d, 'tp': tp_d})
    return pd.DataFrame(rows)


def pf(p):
    gw = p[p > 0].sum(); gl = -p[p < 0].sum()
    return gw / gl if gl > 0 else float('inf')


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
t = run(filter_period(df, '2020-01-01', '2026-12-31'), atr_prior(filter_period(df, '2020-01-01', '2026-12-31'), '30min'))

print("=" * 86)
print("  OPEN-DRIVE CONTINUATION — LOCKED CARD")
print("  11:00 | gate 1.5x30m ATR | TP 0.5xATR30 | RR 0.2 stop=min(2.5xATR30,200) | flat 15:55")
print("=" * 86)
print(f"  {'Year':<5}{'Trades':>7}{'WinRate':>9}{'PF':>6} | {'TPavg':>6}{'TPmed':>6}{'SLavg':>6}{'SLmed':>6}"
      f" | {'TP$avg':>7}{'SL$avg':>7}")
print("  " + "-" * 78)
for y in YEARS:
    s = t[t.y == y]
    if len(s) == 0:
        continue
    print(f"  {y:<5}{len(s):>7}{s.pnl.gt(0).mean()*100:>8.1f}%{pf(s.pnl):>6.2f} | "
          f"{s.tp.mean():>6.1f}{s.tp.median():>6.1f}{s.sl.mean():>6.1f}{s.sl.median():>6.1f} | "
          f"{s.tp.mean()*PV:>7.0f}{s.sl.mean()*PV:>7.0f}")
print("  " + "-" * 78)
s = t
print(f"  {'ALL':<5}{len(s):>7}{s.pnl.gt(0).mean()*100:>8.1f}%{pf(s.pnl):>6.2f} | "
      f"{s.tp.mean():>6.1f}{s.tp.median():>6.1f}{s.sl.mean():>6.1f}{s.sl.median():>6.1f} | "
      f"{s.tp.mean()*PV:>7.0f}{s.sl.mean()*PV:>7.0f}")
print(f"\n  Overall: {len(t)} trades | {len(t)/7:.0f}/yr | {t.pnl.gt(0).mean()*100:.1f}% WR | "
      f"PF {pf(t.pnl):.2f} | exp ${t.pnl.mean():.0f}/trade | net ${t.pnl.sum():.0f}")
