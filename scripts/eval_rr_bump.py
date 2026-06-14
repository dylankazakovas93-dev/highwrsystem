"""Convexity sweep: same open-drive eval config (gate2.0xATR30 entry, 11:00 cont,
TP=0.12xATR60 ~ 11pt-2026, 90min/15:55 exit), but bump RR up = TIGHTEN the stop.
stop_dist = TP_dist / RR. Higher RR -> tighter stop -> lower WR, more convex payoff.
Breakeven WR = 1/(1+RR): RR1=50%, RR2=33%, RR3=25%, RR4=20%, RR5=17%."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import scripts.eval_rr075 as R
from rangefade.engine import run_bracket, market_fill

GATE, TF = 2.0, '60min'
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
ents = R.entries(GATE)
mult = R.calib_mult(TF, ents)   # TP multiplier locked (same ~11pt TP as RR0.75 run)

def run(rr):
    rows = []
    for ei, s, fi, td, yr in ents:
        a = R.ATR[TF].get(td, np.nan)
        if not (np.isfinite(a) and a > 0):
            continue
        tp_d = mult * a
        sl_d = tp_d / rr
        entry = market_fill(s, R.o[ei], R.SLIP)
        bx = run_bracket(s, ei, entry, entry - s*sl_d, entry + s*tp_d,
                          R.o, R.h, R.l, R.c, fi, R.TICK, R.SLIP, 1, "loss")
        rows.append({'y': yr, 'pnl': s*(bx.exit_px - entry)*R.PV - R.COMM})
    return pd.DataFrame(rows)

def pf(p):
    gw, gl = p[p > 0].sum(), -p[p < 0].sum()
    return gw/gl if gl > 0 else float('inf')

print(f"config: gate{GATE}xATR30 | 11:00 cont | TP={mult:.3f}xATR60 (~11pt-2026) | 90min/15:55")
print("higher RR = tighter stop = more convex.  per 1 contract.\n")
print(f"{'RR':>4}{'beWR':>6}  PF per year ->  " + " ".join(f"{y:>5}" for y in YEARS) +
      f" | {'allWR':>6}{'allPF':>6}{'recWR':>6}{'recPF':>6}{'avgW$':>7}{'avgL$':>7}{'exp$':>6}")
print("-"*120)
for rr in [0.75, 1, 2, 3, 4, 5]:
    t = run(rr); t['win'] = t.pnl > 0
    bewr = 1/(1+rr)*100
    cells = " ".join(f"{pf(t[t.y==y].pnl):>5.2f}" if len(t[t.y==y])>=8 else "    ." for y in YEARS)
    rec = t[t.y >= 2024]
    avgW = t.pnl[t.win].mean(); avgL = t.pnl[~t.win].mean()
    print(f"{rr:>4.2f}{bewr:>5.0f}%  {'':>14}{cells} | "
          f"{t.win.mean()*100:>5.1f}%{pf(t.pnl):>6.2f}{rec.win.mean()*100:>5.1f}%{pf(rec.pnl):>6.2f}"
          f"{avgW:>7.0f}{avgL:>7.0f}{t.pnl.mean():>6.0f}")
