"""Overnight-drift (ETH) long test. Thesis: ETH (18:00->09:30 ET) carries the index's
positive drift. Entry at 18:00 ET (new trading-day open), optionally on a pullback of
pull*ATR below the 18:00 open. Exit at 09:30 RTH open (time) OR bracket TP=0.75*ATR,
SL=TP/RR. Sweep ATR timeframe. Report per-year WR + expectancy. Control: overnight SHORT."""
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
ATR={tf:K.atr_prior(df,tf) for tf in ['15min','30min','60min']}

def overnight(tf, mode, pull=0.0, rr=1.0, tp_mult=0.75, sidectl=1):
    """mode: 'time' (exit 09:30) or 'bracket'. sidectl: +1 long, -1 short(control)."""
    rows=[]
    for d in np.unique(dcs):
        pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]]
        a=ATR[tf].get(td,np.nan)
        if not(np.isfinite(a)and a>0):continue
        dm=mins[pos]
        # ETH bars = before the first 09:30 RTH bar of this trading day
        rth=pos[(bd[pos]==td)&(dm>=570)]
        if len(rth)==0:continue
        rth_open_i=rth[0]
        eth=pos[pos<rth_open_i]
        if len(eth)<30:continue
        o18=o[eth[0]]
        s=sidectl
        if pull>0:  # wait for a dip of pull*ATR below 18:00 open, limit-long there
            lvl=o18 - pull*a
            hit=np.nonzero(l[eth]<=lvl)[0]
            if len(hit)==0:continue
            ei=eth[hit[0]]; entry=lvl  # limit fill at the level
            if ei+1>=rth_open_i:continue
            ei=ei+1
        else:
            ei=eth[0]+1; entry=market_fill(s,o[ei],SLIP)
        if ei>=rth_open_i:continue
        if mode=='time':
            px=market_fill(-s,c[rth_open_i],SLIP)
            rows.append({'y':td.year,'pnl':s*(px-entry)*PV-COMM})
        else:
            tp_d=tp_mult*a; sl_d=tp_d/rr
            bx=run_bracket(s,ei,entry,entry-s*sl_d,entry+s*tp_d,o,h,l,c,rth_open_i,TICK,SLIP,1,"loss")
            rows.append({'y':td.year,'pnl':s*(bx.exit_px-entry)*PV-COMM})
    return pd.DataFrame(rows)

def line(label,t):
    if len(t)==0:print(f"  {label:<34}(no trades)");return
    t['win']=t.pnl>0
    wr={y:t[t.y==y].win.mean()*100 for y in YEARS if len(t[t.y==y])>=8}
    cells=" ".join(f"{wr.get(y,float('nan')):>4.0f}" for y in YEARS)
    print(f"  {label:<34}n={len(t):>4} | {cells} | {t.win.mean()*100:>5.1f}% exp${t.pnl.mean():>5.0f}")

print("OVERNIGHT (ETH 18:00->09:30) drift test, NQ.   per-year WR | overall WR + exp$/trade")
print("="*100)
print("  RAW PHENOMENON (hold full overnight, exit 09:30):")
line("long 18:00->0930 (time)", overnight('30min','time',sidectl=1))
line("SHORT 18:00->0930 (control)", overnight('30min','time',sidectl=-1))
print("\n  BRACKET long, TP 0.75xATR, exit by 0930 (breakeven WR: RR0.75=57%, RR1.0=50%):")
for tf in ['15min','30min','60min']:
    for rr in [0.75,1.0]:
        line(f"long {tf} TP0.75xATR RR{rr}", overnight(tf,'bracket',pull=0.0,rr=rr))
print("\n  PULLBACK long (limit -0.3xATR below 18:00 open), bracket TP 0.75xATR:")
for tf in ['15min','30min','60min']:
    for rr in [0.75,1.0]:
        line(f"pull0.3 {tf} TP0.75 RR{rr}", overnight(tf,'bracket',pull=0.3,rr=rr))
