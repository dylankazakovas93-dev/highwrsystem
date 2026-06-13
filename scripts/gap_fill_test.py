"""New mechanism test: overnight gap-fill (mean reversion to prior RTH close).
At 09:30 measure gap = open(09:30) - prior_RTH_close. If |gap| >= threshold, fade toward
the prior close: TP = prior close (TP_dist = |gap|), SL = |gap|/RR beyond entry. Exit at
RTH close if unresolved. Sweep min-gap x RR. Report per-year WR. Control = play the gap
(continuation) instead of fade. Goal check: 70%+ WR at RR>=0.75 across regimes?"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import scripts.keeper_config as K
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

df = K.add_trading_day(K.load_ohlcv("data/NQ_continuous.parquet"), 18)
df = K.filter_period(df, "2020-01-01", "2026-12-31")
o=df["open"].to_numpy(float);h=df["high"].to_numpy(float);l=df["low"].to_numpy(float);c=df["close"].to_numpy(float)
mins=(df.index.hour*60+df.index.minute).to_numpy();bd=np.array([d.date() for d in df.index])
dcs,_=pd.factorize(df["trading_day"])
order=np.unique(dcs)

# prior RTH close per trading day = close of last bar with time in [15:59] of the prior day
prior_close={}
last_rth=None
for d in order:
    pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]];dm=mins[pos];dd=bd[pos]
    rthbars=pos[(dd==td)&(dm>=570)&(dm<960)]
    prior_close[td]=(c[last_rth] if last_rth is not None else np.nan)
    if len(rthbars):last_rth=rthbars[-1]

def run(min_gap, rr, fade=True, max_gap=400):
    rows=[]
    for d in order:
        pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]];dm=mins[pos];dd=bd[pos]
        pc=prior_close.get(td,np.nan)
        if not np.isfinite(pc):continue
        op=pos[(dd==td)&(dm>=570)&(dm<960)]
        if len(op)<5:continue
        o930=o[op[0]];gap=o930-pc
        if not (min_gap<=abs(gap)<=max_gap):continue
        ei=op[0]+1  # enter next bar after the open
        if ei not in op:  # ensure entry within RTH
            ei=op[1] if len(op)>1 else op[0]
        flat_i=op[-1]
        if ei>=flat_i:continue
        # fade: trade toward prior close (against the gap). continuation: with the gap.
        s = (-int(np.sign(gap))) if fade else int(np.sign(gap))
        entry=market_fill(s,o[ei],SLIP)
        tp_d=abs(gap); sl_d=tp_d/rr
        tp=entry+s*tp_d; stop=entry-s*sl_d
        bx=run_bracket(s,ei,entry,stop,tp,o,h,l,c,flat_i,TICK,SLIP,1,"loss")
        rows.append({'y':td.year,'pnl':s*(bx.exit_px-entry)*PV-COMM,'gap':abs(gap)})
    return pd.DataFrame(rows)

print("GAP-FILL FADE (mean reversion to prior RTH close), NQ. Breakeven WR: RR0.75=57%, RR1.0=50%.")
print("="*98)
print(f"{'minGap':>7}{'RR':>5}{'n':>5}{'/yr':>5}{'gapMed':>7} | "+" ".join(f"{y:>4}" for y in YEARS)+f" | {'ALL':>6}{'rec':>6}")
print("-"*98)
for mg in [8,11,15,20,30]:
    for rr in [0.75,1.0,1.5]:
        t=run(mg,rr,fade=True)
        if len(t)<60:continue
        t['win']=t.pnl>0
        wr={y:t[t.y==y].win.mean()*100 for y in YEARS if len(t[t.y==y])>=8}
        cells=" ".join(f"{wr.get(y,float('nan')):>4.0f}" for y in YEARS)
        rec=t[t.y>=2024]
        print(f"{mg:>7}{rr:>5}{len(t):>5}{len(t)/7:>5.0f}{t.gap.median():>7.1f} | {cells} | "
              f"{t.win.mean()*100:>5.1f}%{rec.win.mean()*100:>5.1f}%")
print("\nCONTROL — play the gap (continuation) at min_gap=11, RR 1.0:")
t=run(11,1.0,fade=False);t['win']=t.pnl>0
print(f"  n={len(t)} WR={t.win.mean()*100:.1f}% (if fade WR>>continuation WR, the mean-reversion edge is real)")
