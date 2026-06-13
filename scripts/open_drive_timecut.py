"""Time-based loss-cut perturbation.
Base (keeper): 11:00 | gate 1.5x30m ATR | TP 0.40x ATR15 | RR 0.2 (stop=min(TP/0.2,200))
  | final flat 90min/15:55 | 1 trade/day.
Rule under test: at T_check min after entry, if the position is UNDERWATER (close worse
than entry), flatten at market (small loss). Winners keep running to TP / 90min flat.
Compared against 'none' (no early cut = let the structural stop handle losers)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
GATE, RR, CAP, TPM, FINAL_HOLD = 1.5, 0.2, 200, 0.40, 90


def atr_prior(df, freq):
    g = df.resample(freq, label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()


def walk(s, ei, entry, stop, tp, o, h, l, c, flat_i, t_check, use_stop=True):
    """t_check: bar offset for the underwater check (None=off). use_stop: structural stop on/off."""
    j = ei
    while j <= flat_i:
        hi, lo = h[j], l[j]
        stop_hit = use_stop and ((lo <= stop) if s > 0 else (hi >= stop))
        tp_hit = (hi >= tp) if s > 0 else (lo <= tp)
        if stop_hit and tp_hit:
            return (min(o[j], stop) - SLIP) if s > 0 else (max(o[j], stop) + SLIP), 'stop'
        if stop_hit:
            return (min(o[j], stop) - SLIP) if s > 0 else (max(o[j], stop) + SLIP), 'stop'
        if tp_hit:
            return tp, 'tp'
        if t_check is not None and j == ei + t_check:
            under = (c[j] < entry) if s > 0 else (c[j] > entry)
            if under:
                return (c[j] - s * SLIP), 'timecut'
        if j == flat_i:
            return (c[j] - s * SLIP), 'time'
        j += 1
    return c[flat_i], 'time'


def run(df, g30, a15, t_check, use_stop=True, snapshot='11:00', flat='15:55'):
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
        ga = g30.get(td, np.nan); ta = a15.get(td, np.nan)
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
        flat_i = min(eod_i, ei + FINAL_HOLD)
        if ei >= flat_i:
            continue
        entry = o[ei] + s * SLIP
        tp_d = TPM * ta; sl_d = min(tp_d / RR, CAP)
        tp = entry + s * tp_d; stop = entry - s * sl_d
        px, reason = walk(s, ei, entry, stop, tp, o, h, l, c, flat_i, t_check, use_stop)
        rows.append({'y': td.year, 'pnl': s * (px - entry) * PV - COMM, 'reason': reason})
    return pd.DataFrame(rows)


def pf(p):
    gw = p[p > 0].sum(); gl = -p[p < 0].sum()
    return gw / gl if gl > 0 else float('inf')


def card(label, t_check, use_stop, g30, a15, dall):
    t = run(dall, g30, a15, t_check, use_stop)
    wr = {y: t[t.y == y].pnl.gt(0).mean() * 100 for y in YEARS if len(t[t.y == y])}
    cells = " ".join(f"{wr.get(y, float('nan')):>5.1f}" for y in YEARS)
    avgL = t[t.pnl < 0].pnl.mean()
    print(f"  {label:<16}{t.pnl.gt(0).mean()*100:>6.1f}%{pf(t.pnl):>6.2f}{t.pnl.mean():>6.0f}"
          f"{t.pnl.sum():>8.0f}{avgL:>7.0f} | {cells}")
    return t


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
dall = filter_period(df, '2020-01-01', '2026-12-31')
g30 = atr_prior(dall, '30min'); a15 = atr_prior(dall, '15min')

hdr = f"  {'config':<16}{'WR':>7}{'PF':>6}{'exp$':>6}{'net$':>8}{'avgLoss':>7} | " + " ".join(f"{y:>5}" for y in YEARS)
print("VARIATION 1 — cut underwater trades at T_check (structural stop still active)")
print("=" * 96); print(hdr); print("  " + "-" * 88)
card("none (baseline)", None, True, g30, a15, dall)
for tc in [10, 20, 30, 45, 60]:
    card(f"cut@{tc}min", tc, True, g30, a15, dall)

print("\nVARIATION 2 — NO structural stop; only exit = underwater cut at T_check (or 90min flat)")
print("=" * 96); print(hdr); print("  " + "-" * 88)
for tc in [10, 20, 30, 45, 60]:
    card(f"cut@{tc}min,noSL", tc, False, g30, a15, dall)
