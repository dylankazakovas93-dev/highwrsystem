"""Breakeven-stop perturbation on the combined config.
Base (locked): 11:00 | gate 1.5x30m ATR | TP 0.40x ATR15 (~15pt in 2025/26) | RR 0.2
  (stop=min(TP/0.2,200)) | exit at market after 90 min or 15:55.
BE rule: once favorable excursion reaches be_trig * TP_dist, move stop to entry.
Conservative: BE arms at end of the triggering bar (protection active next bar);
both-touch counts against the trader (loss pre-arm, BE-scratch post-arm)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.data import load_ohlcv, add_trading_day, filter_period

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
GATE, RR, CAP, TPM, HOLD = 1.5, 0.2, 200, 0.40, 90


def atr_prior(df, freq):
    g = df.resample(freq, label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high'] - g['low'], (g['high'] - pc).abs(), (g['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_t=tr), 18)['trading_day']
    return pd.Series(atr.groupby(td.values).last()).shift(1).to_dict()


def walk_be(s, ei, entry, stop0, tp, be_level, o, h, l, c, flat_i):
    """be_level=None disables BE. Returns (exit_px, reason)."""
    armed = False
    stop = stop0
    j = ei
    while j <= flat_i:
        hi, lo = h[j], l[j]
        if not armed:
            stop_hit = (lo <= stop) if s > 0 else (hi >= stop)
            tp_hit = (hi >= tp) if s > 0 else (lo <= tp)
            if stop_hit and tp_hit:
                return (min(o[j], stop) - SLIP) if s > 0 else (max(o[j], stop) + SLIP), 'stop'  # both-touch=loss
            if stop_hit:
                return (min(o[j], stop) - SLIP) if s > 0 else (max(o[j], stop) + SLIP), 'stop'
            if tp_hit:
                return tp, 'tp'
            if be_level is not None:
                reach = (hi >= be_level) if s > 0 else (lo <= be_level)
                if reach:
                    armed = True
                    stop = entry  # move to breakeven
        else:
            be_hit = (lo <= stop) if s > 0 else (hi >= stop)
            tp_hit = (hi >= tp) if s > 0 else (lo <= tp)
            if be_hit and tp_hit:
                return entry, 'be'   # both-touch post-arm -> scratch (conservative)
            if be_hit:
                return entry, 'be'
            if tp_hit:
                return tp, 'tp'
        if j == flat_i:
            return (c[j] - s * SLIP) if s > 0 else (c[j] + s * SLIP), 'time'
        j += 1
    return c[flat_i], 'time'


def run(df, gate_atr, tp_atr, be_trig, snapshot='11:00', flat='15:55'):
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
        flat_i = min(eod_i, ei + HOLD)
        if ei >= flat_i:
            continue
        entry = o[ei] + s * SLIP
        tp_d = TPM * ta; sl_d = min(tp_d / RR, CAP)
        tp = entry + s * tp_d; stop0 = entry - s * sl_d
        be_level = None if be_trig is None else entry + s * be_trig * tp_d
        px, reason = walk_be(s, ei, entry, stop0, tp, be_level, o, h, l, c, flat_i)
        rows.append({'y': td.year, 'pnl': s * (px - entry) * PV - COMM, 'reason': reason})
    return pd.DataFrame(rows)


def pf(p):
    gw = p[p > 0].sum(); gl = -p[p < 0].sum()
    return gw / gl if gl > 0 else float('inf')


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
dall = filter_period(df, '2020-01-01', '2026-12-31')
g30 = atr_prior(dall, '30min'); a15 = atr_prior(dall, '15min')

print("BREAKEVEN-STOP PERTURBATION")
print("Base: 11:00 | gate 1.5x30m | TP 0.40xATR15 | RR 0.2 cap200 | exit 90min/15:55")
print("BE: move stop to entry once price reaches be_trig x TP in favor")
print("=" * 92)
print(f"  {'be_trig':<8}{'Trades':>7}{'WR':>7}{'PF':>6}{'exp$':>6}{'net$':>8} | " + " ".join(f"{y:>5}" for y in YEARS))
print("  " + "-" * 84)
for bt in [None, 0.5, 0.75, 1.0, 1.5, 2.0]:
    t = run(dall, g30, a15, bt)
    wr = {y: t[t.y == y].pnl.gt(0).mean() * 100 for y in YEARS if len(t[t.y == y])}
    cells = " ".join(f"{wr.get(y, float('nan')):>5.1f}" for y in YEARS)
    lab = 'none' if bt is None else f'{bt}xTP'
    print(f"  {lab:<8}{len(t):>7}{t.pnl.gt(0).mean()*100:>6.1f}%{pf(t.pnl):>6.2f}{t.pnl.mean():>6.0f}{t.pnl.sum():>8.0f} | {cells}")

# detail card for a representative BE level (1.0xTP) — exit reason mix
print("\nDetail: BE @1.0xTP — exit reason mix (be = scratched at breakeven)")
t = run(dall, g30, a15, 1.0)
print("  reasons:", t['reason'].value_counts().to_dict())
