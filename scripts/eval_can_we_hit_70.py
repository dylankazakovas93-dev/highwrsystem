"""Decisive test: is WR >= 70% achievable at RR >= 0.75 anywhere in this strategy?
Push the only WR levers: earlier snapshot (10:00/11:00) + demanding gate (2.0-3.5),
RR 0.75 & 1.0, TP calibrated to ~11pt in 2026 on 15m/60m ATR. Report recent(24-26) &
all-years WR with sample sizes. Breakeven WR: RR0.75->57.1%, RR1.0->50%."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import scripts.keeper_config as K
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
TP_2026 = 11.0

df = K.add_trading_day(K.load_ohlcv("data/NQ_continuous.parquet"), 18)
df = K.filter_period(df, "2020-01-01", "2026-12-31")
o=df["open"].to_numpy(float);h=df["high"].to_numpy(float);l=df["low"].to_numpy(float);c=df["close"].to_numpy(float)
mins=(df.index.hour*60+df.index.minute).to_numpy();bd=np.array([d.date() for d in df.index])
dcs,_=pd.factorize(df["trading_day"])
ATR={tf:K.atr_prior(df,tf) for tf in ['15min','60min','30min']}
G30=ATR['30min']

def entries(snap_min, gate_mult):
    out=[]
    for d in np.unique(dcs):
        pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]]
        ga=G30.get(td,np.nan)
        if not(np.isfinite(ga)and ga>0):continue
        dm=mins[pos];dd_=bd[pos];op=pos[(dd_==td)&(dm>=570)&(dm<960)]
        if len(op)==0:continue
        ro=o[op[0]];sp=pos[(dd_==td)&(dm>=snap_min)&(dm<960)]
        if len(sp)==0:continue
        si=sp[0];mv=c[si]-ro
        if abs(mv)<gate_mult*ga:continue
        s=int(np.sign(mv));ei=si+1;fp=pos[(dd_==td)&(dm>=955)];eod=fp[0] if len(fp) else pos[-1];fi=min(eod,ei+90)
        if ei>=fi:continue
        out.append((ei,s,fi,td,td.year))
    return out

def run(ents,tf,rr):
    vals=[ATR[tf].get(td) for (_,_,_,td,yr) in ents if yr==2026 and np.isfinite(ATR[tf].get(td,np.nan))]
    if not vals:return None
    mult=TP_2026/np.median(vals)
    rows=[]
    for ei,s,fi,td,yr in ents:
        a=ATR[tf].get(td,np.nan)
        if not(np.isfinite(a)and a>0):continue
        tp_d=mult*a;sl_d=tp_d/rr;entry=market_fill(s,o[ei],SLIP)
        bx=run_bracket(s,ei,entry,entry-s*sl_d,entry+s*tp_d,o,h,l,c,fi,TICK,SLIP,1,"loss")
        rows.append({'y':yr,'pnl':s*(bx.exit_px-entry)*PV-COMM})
    return pd.DataFrame(rows)

print("Hunting for WR >= 70% at RR >= 0.75. (* = >=70% recent with >=40 recent trades)")
print("="*96)
print(f"{'snap':>6}{'gate':>5}{'tf':>6}{'rr':>5}{'n':>5}{'nRec':>6}{'allWR':>7}{'recWR':>7}{'2025':>6}{'2026':>6}")
print("-"*96)
hits=[]
for snap,smin in [('10:00',600),('11:00',660)]:
    for g in [2.0,2.5,3.0,3.5]:
        ents=entries(smin,g)
        for tf in ['15min','60min']:
            for rr in [0.75,1.0]:
                t=run(ents,tf,rr)
                if t is None or len(t)<40:continue
                t['win']=t.pnl>0;rec=t[t.y>=2024]
                allwr=t.win.mean()*100;recwr=rec.win.mean()*100
                w25=t[t.y==2025].win.mean()*100 if len(t[t.y==2025]) else float('nan')
                w26=t[t.y==2026].win.mean()*100 if len(t[t.y==2026]) else float('nan')
                star="*" if (recwr>=70 and len(rec)>=40) else " "
                print(f"{snap:>6}{g:>5}{tf:>6}{rr:>5}{len(t):>5}{len(rec):>6}{allwr:>6.1f}%{recwr:>6.1f}%{w25:>5.0f}%{w26:>5.0f}%{star}")
                if recwr>=70 and len(rec)>=40:hits.append((snap,g,tf,rr,recwr,len(rec)))
print("\n" + ("RESULT: configs hitting >=70% recent WR (>=40 trades): "+str(hits) if hits
             else "RESULT: NO config reaches 70% recent WR at RR>=0.75 with a tradeable sample."))
