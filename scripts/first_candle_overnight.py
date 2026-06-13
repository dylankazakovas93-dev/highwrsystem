"""First-candle-after-18:00 momentum trigger for the overnight session.
Look at the first TF candle (15m/30m/1h/4h) starting 18:00 ET. If its signed move
(close-open) >= thr * ATR(TF, prior), enter at the next bar. Hold to 09:30 (time) or
bracket TP=0.75*ATR/SL=TP/RR. Test long-only (move up) and direction-following.
Sweep TF x threshold. 20% already ruled out -> test 0.3..1.0."""
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
ATRTF={tf:K.atr_prior(df,f"{tf}min") for tf in [15,30,60,240]}

def run(tf, thr, mode='time', rr=0.75, tp_mult=0.75, follow=False):
    """follow=False: long only if first candle up>=thr*ATR. follow=True: trade its direction."""
    rows=[]
    for d in np.unique(dcs):
        pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]]
        a=ATRTF[tf].get(td,np.nan)
        if not(np.isfinite(a)and a>0):continue
        dm=mins[pos]
        start=pos[0]                      # 18:00 bar (first bar of trading day)
        if mins[start]!=18*60:            # require the day to actually start at 18:00
            continue
        # first TF candle = bars [18:00, 18:00+tf)
        end_min=18*60+tf
        cand=pos[(np.arange(len(pos))< (np.searchsorted(dm, end_min if end_min<=1439 else 1440)))]
        # build candle by absolute minute window across the midnight wrap:
        # bars with elapsed minutes < tf from 18:00
        elapsed=(dm-18*60)%1440
        candmask=elapsed<tf
        cand=pos[candmask]
        if len(cand)<max(3,tf//3):continue
        o18=o[cand[0]]; cclose=c[cand[-1]]
        move=cclose-o18
        ei=cand[-1]+1
        # exit deadline = 09:30 RTH open
        rth=pos[(bd[pos]==td)&(dm>=570)]
        if len(rth)==0:continue
        rth_open_i=rth[0]
        if ei>=rth_open_i:continue
        if follow:
            if abs(move)<thr*a:continue
            s=int(np.sign(move))
        else:
            if move<thr*a:continue        # long only, up move
            s=1
        entry=market_fill(s,o[ei],SLIP)
        if mode=='time':
            px=market_fill(-s,c[rth_open_i],SLIP); rows.append({'y':td.year,'pnl':s*(px-entry)*PV-COMM})
        else:
            tp_d=tp_mult*a; sl_d=tp_d/rr
            bx=run_bracket(s,ei,entry,entry-s*sl_d,entry+s*tp_d,o,h,l,c,rth_open_i,TICK,SLIP,1,"loss")
            rows.append({'y':td.year,'pnl':s*(bx.exit_px-entry)*PV-COMM})
    return pd.DataFrame(rows)

def line(label,t):
    if len(t)<40:print(f"  {label:<28}n={len(t):>4} (thin)");return
    t['win']=t.pnl>0
    wr={y:t[t.y==y].win.mean()*100 for y in YEARS if len(t[t.y==y])>=8}
    cells=" ".join(f"{wr.get(y,float('nan')):>4.0f}" for y in YEARS)
    gw=t.pnl[t.pnl>0].sum();gl=-t.pnl[t.pnl<0].sum();pf=gw/gl if gl>0 else 9.9
    print(f"  {label:<28}n={len(t):>4} | {cells} | {t.win.mean()*100:>5.1f}% exp${t.pnl.mean():>5.0f} PF{pf:>4.2f}")

for follow in [False, True]:
    tag="LONG-ONLY (up move)" if not follow else "DIRECTION-FOLLOWING"
    print(f"\n{'='*100}\nFIRST-CANDLE TRIGGER, {tag}, hold to 09:30 (time exit)")
    print(f"  per-year WR | overall WR exp$ PF   (baseline no-gate long: 54.6%/+$106)")
    for tf in [15,30,60,240]:
        for thr in [0.3,0.5,0.75,1.0]:
            line(f"tf{tf} thr{thr} time", run(tf,thr,'time',follow=follow))
print(f"\n{'='*100}\nBRACKET RR0.75 (does a strong first candle make the eval case work? breakeven 57%)")
for tf in [30,60]:
    for thr in [0.5,0.75,1.0]:
        line(f"LONG tf{tf} thr{thr} RR0.75", run(tf,thr,'bracket',rr=0.75,follow=False))
