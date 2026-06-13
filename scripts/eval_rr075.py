"""Max-WR search at RR 0.75 (reward:risk = $1500:$2000), TP calibrated to ~11pt in 2026.
TP = mult * ATR(tf) with mult chosen so 2026 median TP = 11 (earlier years scale down).
SL = TP / 0.75. 11:00 / continuation / 90min exit. Sweep ATR timeframe x gate. Report WR.
Breakeven WR at RR 0.75 = 1/(1+0.75) = 57.1%."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import scripts.keeper_config as K
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
RR = 0.75
TP_2026 = 11.0

df = K.add_trading_day(K.load_ohlcv("data/NQ_continuous.parquet"), 18)
df = K.filter_period(df, "2020-01-01", "2026-12-31")
o=df["open"].to_numpy(float);h=df["high"].to_numpy(float);l=df["low"].to_numpy(float);c=df["close"].to_numpy(float)
mins=(df.index.hour*60+df.index.minute).to_numpy();bd=np.array([d.date() for d in df.index])
dcs,_=pd.factorize(df["trading_day"])
TFS=['1min','5min','15min','30min','60min']
ATR={tf:K.atr_prior(df,tf) for tf in TFS}
GATE30=ATR['30min']

def entries(gate_mult):
    out=[]
    for d in np.unique(dcs):
        pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]]
        ga=GATE30.get(td,np.nan)
        if not(np.isfinite(ga)and ga>0):continue
        dm=mins[pos];dd_=bd[pos];op=pos[(dd_==td)&(dm>=570)&(dm<960)]
        if len(op)==0:continue
        ro=o[op[0]];sp=pos[(dd_==td)&(dm>=660)&(dm<960)]
        if len(sp)==0:continue
        si=sp[0];mv=c[si]-ro
        if abs(mv)<gate_mult*ga:continue
        s=int(np.sign(mv));ei=si+1;fp=pos[(dd_==td)&(dm>=955)];eod=fp[0] if len(fp) else pos[-1];fi=min(eod,ei+90)
        if ei>=fi:continue
        out.append((ei,s,fi,td,td.year))
    return out

def calib_mult(tf, ents):
    """multiplier so 2026 median TP == 11pt."""
    vals=[ATR[tf].get(td) for (_,_,_,td,yr) in ents if yr==2026 and np.isfinite(ATR[tf].get(td,np.nan))]
    return TP_2026/np.median(vals) if vals else np.nan

def backtest(ents, tf, mult):
    rows=[]
    for ei,s,fi,td,yr in ents:
        a=ATR[tf].get(td,np.nan)
        if not(np.isfinite(a)and a>0):continue
        tp_d=mult*a; sl_d=tp_d/RR
        entry=market_fill(s,o[ei],SLIP)
        bx=run_bracket(s,ei,entry,entry-s*sl_d,entry+s*tp_d,o,h,l,c,fi,TICK,SLIP,1,"loss")
        rows.append({'y':yr,'pnl':s*(bx.exit_px-entry)*PV-COMM,'tp':tp_d})
    return pd.DataFrame(rows)

print(f"RR {RR} (SL=TP/{RR}={1/RR:.2f}xTP) | TP calibrated to ~{TP_2026:.0f}pt median in 2026 | breakeven WR 57.1%")
print("="*94)
print(f"{'TPtf':>6}{'gate':>5}{'mult':>6}{'n':>5}{'2026TP':>7} | "+" ".join(f"{y:>4}" for y in YEARS)+f" | {'ALL':>6}{'rec':>6}")
print("-"*94)
best=[]
for tf in TFS:
    for g in [1.0,1.5,2.0]:
        ents=entries(g)
        m=calib_mult(tf,ents)
        if not np.isfinite(m):continue
        t=backtest(ents,tf,m); t['win']=t.pnl>0
        if len(t)<50:continue
        wr={y:t[t.y==y].win.mean()*100 for y in YEARS if len(t[t.y==y])>=8}
        cells=" ".join(f"{wr.get(y,float('nan')):>4.0f}" for y in YEARS)
        allwr=t.win.mean()*100; rec=t[t.y>=2024].win.mean()*100
        tp26=t[t.y==2026].tp.median()
        print(f"{tf:>6}{g:>5}{m:>6.2f}{len(t):>5}{tp26:>7.1f} | {cells} | {allwr:>5.1f}%{rec:>5.1f}%")
        best.append((rec,allwr,tf,g,m,t))
print("\nBEST by recent (2024-26) WR:")
for rec,allwr,tf,g,m,t in sorted(best,key=lambda x:-x[0])[:3]:
    w=t.win;avgW=t[w].pnl.mean();avgL=t[~w].pnl.mean()
    print(f"  TP {m:.2f}x{tf}, gate {g}x30m: recent WR {rec:.1f}%, all {allwr:.1f}%, "
          f"avgW/L per-1-contract ${avgW:.0f}/${avgL:.0f}, exp ${t.pnl.mean():.0f}")
