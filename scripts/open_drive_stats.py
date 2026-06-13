"""Open-drive continuation: per-year TP/stop point distances + entry-gate & snapshot sweep.
TP = 1xATR30(prior day); stop = TP/RR. The set of trades depends only on (gate, snapshot,
direction), NOT on RR — RR only moves the stop. So TP stats are RR-independent; stop = TP/RR."""
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
    td = add_trading_day(g.assign(_tr=tr), 18)['trading_day']
    daily = atr.groupby(td.values).last()
    return pd.Series(daily).shift(1)


def run(df, snapshot, gate, rr, direction='cont', tp_mult=1.0, flat='15:55'):
    o = df['open'].to_numpy(float); h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float); c = df['close'].to_numpy(float)
    idx = df.index; mins = (idx.hour * 60 + idx.minute).to_numpy()
    bar_date = np.array([d.date() for d in idx])
    day_codes, _ = pd.factorize(df['trading_day'])
    atr_map = atr30_prior(df).to_dict()
    sh, sm = map(int, snapshot.split(':')); snap_min = sh * 60 + sm
    fh, fm = map(int, flat.split(':')); flat_min = fh * 60 + fm
    rth = 9 * 60 + 30
    rows = []
    for dc in np.unique(day_codes):
        pos = np.nonzero(day_codes == dc)[0]
        td = df['trading_day'].iloc[pos[0]]
        atr = atr_map.get(td, np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue
        dmin = mins[pos]; ddate = bar_date[pos]
        op = pos[(ddate == td) & (dmin >= rth) & (dmin < 16 * 60)]
        if len(op) == 0:
            continue
        open930 = o[op[0]]
        sp = pos[(ddate == td) & (dmin >= snap_min) & (dmin < 16 * 60)]
        if len(sp) == 0:
            continue
        snap_i = sp[0]; move = c[snap_i] - open930
        if abs(move) < gate * atr:
            continue
        sgn = np.sign(move); s = int(sgn) if direction == 'cont' else int(-sgn)
        ent_i = snap_i + 1
        fp = pos[(ddate == td) & (dmin >= flat_min)]
        flat_i = fp[0] if len(fp) else pos[-1]
        if ent_i >= flat_i:
            continue
        entry = market_fill(s, o[ent_i], SLIP)
        tp_dist = tp_mult * atr; sl_dist = tp_dist / rr
        tp = entry + s * tp_dist; stop = entry - s * sl_dist
        bx = run_bracket(s, ent_i, entry, stop, tp, o, h, l, c, flat_i, TICK, SLIP, 1, 'loss')
        pnl = s * (bx.exit_px - entry) * PV - COMM
        rows.append({'td': td, 'y': td.year, 'pnl': pnl, 'tp_dist': tp_dist,
                     'sl_dist': sl_dist, 'reason': bx.reason})
    return pd.DataFrame(rows)


df = load_ohlcv('data/NQ_continuous.parquet'); df = add_trading_day(df, 18)
dall = filter_period(df, '2020-01-01', '2026-12-31')

# ---- Table 1: per-year TP & stop point distances (continuation, gate 1.0) ----
print("=" * 96)
print("PER-YEAR TP & STOP DISTANCES (points) — continuation, 11:00 snapshot, gate=1.0xATR30, TP=1xATR30")
print("  TP = ATR30(prior day); stop = TP/RR.  TP is the same for all RR; stop shown at RR 0.1/0.2/0.3")
print("=" * 96)
t = run(dall, '11:00', 1.0, 0.3)  # trade set independent of RR
print(f"{'year':>5} {'n':>4} | {'TP_avg':>7} {'TP_med':>7} | {'SL_avg@.1':>9} {'@.2':>7} {'@.3':>7} | {'SL_med@.1':>9} {'@.2':>7} {'@.3':>7}")
for y in YEARS:
    s = t[t.y == y]
    if len(s) < 5:
        print(f"{y:>5} {len(s):>4} | (thin)"); continue
    tpa, tpm = s.tp_dist.mean(), s.tp_dist.median()
    print(f"{y:>5} {len(s):>4} | {tpa:>7.1f} {tpm:>7.1f} | "
          f"{tpa/0.1:>9.0f} {tpa/0.2:>7.0f} {tpa/0.3:>7.0f} | "
          f"{tpm/0.1:>9.0f} {tpm/0.2:>7.0f} {tpm/0.3:>7.0f}")

# ---- Table 2: entry-gate & snapshot sweep on WIN RATE (RR=0.1, the high-WR end) ----
for snap in ['11:00', '12:00']:
    print("\n" + "=" * 96)
    print(f"WIN RATE by entry gate — continuation, snapshot={snap}, RR=0.1 (TP=1xATR30, stop=10xATR30)")
    print("=" * 96)
    print(f"{'gate':>5} {'n':>5} {'/yr':>4} | " + " ".join(f"{y:>5}" for y in YEARS) + f" | {'ALLwr':>6} {'exp$':>7}")
    for gate in [1.0, 1.5, 2.0, 2.5]:
        t = run(dall, snap, gate, 0.1)
        if t.empty:
            print(f"{gate:>5} (none)"); continue
        cells = []
        for y in YEARS:
            s = t[t.y == y]
            cells.append(f"{s.pnl.gt(0).mean()*100:>4.0f}%" if len(s) >= 10 else f"{'n'+str(len(s)):>5}")
        print(f"{gate:>5} {len(t):>5} {len(t)/7:>4.0f} | " + " ".join(cells) +
              f" | {t.pnl.gt(0).mean()*100:>5.1f}% {t.pnl.mean():>7.0f}")
