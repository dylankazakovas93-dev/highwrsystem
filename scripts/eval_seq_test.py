"""Sequencing test for the RR0.75 'best-by-recent-WR' open-drive config (the "2025/2026 fit"
config: gate2.0x30minATR entry, TP=0.12x60minATR calibrated to ~11pt-2026 median, RR0.75 stop,
11:00 snapshot, continuation direction, 90min/15:55 exit).

User's proposal: in a month, bank a normal WIN, then SKIP signals (observe only, don't trade)
until one would have been a LOSS, then take the NEXT signal for real -- on the theory that
win-probability is elevated right after an observed loss (tested earlier on the RR0.2 config:
88.6% vs 84.9% baseline, n=105, NOT statistically significant).

Reports:
 1. recent (2024-26) per-contract avgW/avgL + contract sizing for ~$1500 win / ~$2000 risk
 2. unconditional vs after-observed-loss-conditional WR, with n, for recent AND all-years
 3. frequency of 'loss signals' per month (recent) -- is 'wait for a loss' realistic in a month?
 4. monthly bootstrap MC: Policy A (take every signal) vs Policy B (after each real win this
    month, apply the after-loss filter before the next real trade) -- pass rate to
    +3000/-2000 trailing, at several contract counts.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import scripts.eval_rr075 as R
from rangefade.engine import run_bracket, market_fill

GATE, TF = 2.0, '60min'
ents = R.entries(GATE)
mult = R.calib_mult(TF, ents)

rows = []
for ei, s, fi, td, yr in ents:
    a = R.ATR[TF].get(td, np.nan)
    if not (np.isfinite(a) and a > 0):
        continue
    tp_d = mult * a
    sl_d = tp_d / R.RR
    entry = market_fill(s, R.o[ei], R.SLIP)
    bx = run_bracket(s, ei, entry, entry - s*sl_d, entry + s*tp_d,
                      R.o, R.h, R.l, R.c, fi, R.TICK, R.SLIP, 1, "loss")
    rows.append({'td': td, 'y': yr, 'pnl': s*(bx.exit_px - entry)*R.PV - R.COMM})

t = pd.DataFrame(rows).sort_values('td').reset_index(drop=True)
t['win'] = t.pnl > 0
t['month'] = [(d.year, d.month) for d in t['td']]

print(f"config: gate{GATE}x30minATR entry | TP {mult:.3f}x{TF} (~11pt-2026 median) | "
      f"RR{R.RR} (SL=TP/{R.RR}) | 11:00 snapshot, continuation, 90min/15:55 exit\n")

# ---- 1. sizing ----
rec = t[t.y >= 2024].reset_index(drop=True)
avgW, avgL = rec.pnl[rec.win].mean(), rec.pnl[~rec.win].mean()
print(f"[1] recent (2024-26): n={len(rec)}, WR={rec.win.mean()*100:.1f}%, "
      f"avgW=${avgW:.0f}/contract, avgL=${avgL:.0f}/contract")
for n in [5, 6, 7, 8, 10]:
    print(f"     {n:>2} contracts -> win ${avgW*n:>6.0f}   loss ${avgL*n:>7.0f}")

# ---- 2. conditional WR (after an observed losing signal) ----
def cond_wr(df, label):
    prevloss = (df['win'].shift(1) == False)
    cond = df[prevloss.fillna(False)]
    print(f"[2] {label:<18} unconditional WR {df.win.mean()*100:>5.1f}% (n={len(df):>3}) | "
          f"after-observed-loss WR {cond.win.mean()*100:>5.1f}% (n={len(cond):>3})")
    return cond

cond_wr(t, "all-years 2020-26")
cond_wr(rec, "recent 2024-26")

# ---- 3. frequency ----
n_months = rec['month'].nunique()
loss_signals = (~rec.win).sum()
print(f"\n[3] recent: {len(rec)} signals / {n_months} months = {len(rec)/n_months:.2f}/mo "
      f"({n_months} months); {loss_signals} are losses = {loss_signals/n_months:.2f} "
      f"'wait-for-loss' triggers/mo")

# ---- 4. monthly MC: Policy A (take all) vs Policy B (post-win after-loss filter) ----
def monthly_pass(pnl, win, per_month, contracts, target=3000, dd=2000, npaths=20000, seed=7, seq=False):
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(npaths):
        k = max(1, rng.poisson(per_month))
        draws = rng.integers(0, len(pnl), k)
        eq = peak = 0.0
        ok = False
        mode = 'normal'  # 'normal' -> trade it | 'waiting' -> observe only | 'ready' -> trade it
        for di in draws:
            if seq and mode == 'waiting':
                if not win[di]:
                    mode = 'ready'
                continue  # observed, not traded
            x = pnl[di] * contracts
            eq += x; peak = max(peak, eq)
            mode = ('waiting' if seq and x > 0 else 'normal')
            if eq - peak <= -dd:
                break
            if eq >= target:
                ok = True; break
        wins += ok
    return wins / npaths

pnl = rec.pnl.to_numpy(); win = rec.win.to_numpy()
pm = len(rec) / n_months
print(f"\n[4] {pm:.2f} signals/mo (recent), target +$3000 / trailing -$2000")
print(f"{'contracts':>10}{'avgW$':>8}{'avgL$':>8}{'Policy A (all)':>17}{'Policy B (post-win wait-for-loss)':>36}")
for n in [5, 6, 7, 8, 10]:
    pa = monthly_pass(pnl, win, pm, n, seq=False)
    pb = monthly_pass(pnl, win, pm, n, seq=True)
    print(f"{n:>10}{avgW*n:>8.0f}{avgL*n:>8.0f}{pa*100:>16.1f}%{pb*100:>35.1f}%")
