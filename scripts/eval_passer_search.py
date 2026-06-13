"""Convex eval-passer search. 11:00 / continuation / 90min exit fixed.
Sweep gate (x30m ATR) x TP definition (floored at 11pt) x RR (reward:risk = TP/SL).
Rank by P(pass eval): MC bootstrap of per-trade $ to +TARGET before -DD (trailing)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import scripts.keeper_config as K
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
TP_FLOOR = 11.0
TARGET, DD = 3000.0, 2000.0   # eval: +$3000 target, $2000 trailing drawdown, 1 contract

df = K.add_trading_day(K.load_ohlcv("data/NQ_continuous.parquet"), 18)
df = K.filter_period(df, "2020-01-01", "2026-12-31")
o=df["open"].to_numpy(float);h=df["high"].to_numpy(float);l=df["low"].to_numpy(float);c=df["close"].to_numpy(float)
mins=(df.index.hour*60+df.index.minute).to_numpy();bd=np.array([d.date() for d in df.index])
dcs,_=pd.factorize(df["trading_day"])
A={'15m':K.atr_prior(df,'15min'),'30m':K.atr_prior(df,'30min')}

def entries(gate_mult):
    """Precompute signal entries (independent of TP/RR)."""
    out=[]
    for d in np.unique(dcs):
        pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]]
        ga=A['30m'].get(td,np.nan);a15=A['15m'].get(td,np.nan);a30=A['30m'].get(td,np.nan)
        if not(np.isfinite(ga)and ga>0 and np.isfinite(a15)and a15>0):continue
        dm=mins[pos];dd_=bd[pos];op=pos[(dd_==td)&(dm>=570)&(dm<960)]
        if len(op)==0:continue
        ro=o[op[0]];sp=pos[(dd_==td)&(dm>=660)&(dm<960)]
        if len(sp)==0:continue
        si=sp[0];mv=c[si]-ro
        if abs(mv)<gate_mult*ga:continue
        s=int(np.sign(mv));ei=si+1;fp=pos[(dd_==td)&(dm>=955)];eod=fp[0] if len(fp) else pos[-1];fi=min(eod,ei+90)
        if ei>=fi:continue
        out.append((ei,s,fi,a15,a30,td.year))
    return out

def backtest(ents, tp_tf, tp_mult, rr):
    rows=[]
    for ei,s,fi,a15,a30,yr in ents:
        atr = a15 if tp_tf=='15m' else a30
        tp_d=max(tp_mult*atr, TP_FLOOR); sl_d=min(tp_d/rr, 200)
        entry=market_fill(s,o[ei],SLIP)
        bx=run_bracket(s,ei,entry,entry-s*sl_d,entry+s*tp_d,o,h,l,c,fi,TICK,SLIP,1,"loss")
        rows.append({'y':yr,'pnl':s*(bx.exit_px-entry)*PV-COMM,'tp':tp_d})
    return pd.DataFrame(rows)

def p_pass(pnl, target=TARGET, dd=DD, npaths=3000, maxtr=400, seed=1):
    rng=np.random.default_rng(seed); arr=pnl.to_numpy(); n=len(arr); wins=0; tppass=[]
    for _ in range(npaths):
        picks=arr[rng.integers(0,n,maxtr)]; eq=0.0; peak=0.0; res=0
        for k,x in enumerate(picks):
            eq+=x; peak=max(peak,eq)
            if eq-peak<=-dd: res=-1; break
            if eq>=target: res=1; tppass.append(k+1); break
        if res==1: wins+=1
    return wins/npaths, (np.median(tppass) if tppass else None)

ENT={g:entries(g) for g in [1.0,1.5,2.0]}
tp_defs=[('0.4xATR15','15m',0.4),('0.6xATR15','15m',0.6),('0.5xATR30','30m',0.5),('0.7xATR30','30m',0.7)]
rrs=[0.2,0.5,0.8,1.0,1.5,2.0,3.0]

results=[]
for g in [1.0,1.5,2.0]:
    for nm,tf,m in tp_defs:
        for rr in rrs:
            t=backtest(ENT[g],tf,m,rr)
            if len(t)<50: continue
            w=t.pnl>0; pp,med=p_pass(t.pnl)
            results.append({'gate':g,'tp':nm,'rr':rr,'n':len(t),'tpmed':t.tp.median(),
                'wr':w.mean()*100,'avgW':t[w].pnl.mean(),'avgL':t[~w].pnl.mean(),
                'exp':t.pnl.mean(),'ppass':pp*100,'med2pass':med,'trades':t})
R=pd.DataFrame(results)

print(f"EVAL: +${TARGET:.0f} target / ${DD:.0f} trailing DD / 1 contract / TP floor {TP_FLOOR:.0f}pt")
print("="*108)
print(f"{'gate':>5}{'TP_def':>11}{'RR':>5}{'TPmed':>7}{'WR':>7}{'avgW$':>7}{'avgL$':>7}{'exp$':>7}{'P(pass)':>9}{'med#2pass':>10}")
print("-"*108)
top=R.sort_values('ppass',ascending=False)
for _,r in top.head(18).iterrows():
    m2p = f"{r['med2pass']:.0f}" if r['med2pass'] else "-"
    print(f"{r['gate']:>5}{r['tp']:>11}{r['rr']:>5}{r['tpmed']:>7.1f}{r['wr']:>6.1f}%{r['avgW']:>7.0f}{r['avgL']:>7.0f}{r['exp']:>7.0f}{r['ppass']:>8.1f}%{m2p:>10}")

print("\nRR axis at gate 1.5, TP 0.5xATR30 (shows the convexity flip):")
print(f"{'RR':>5}{'TPmed':>7}{'SLmed~':>8}{'WR':>7}{'avgW$':>7}{'avgL$':>7}{'exp$':>7}{'P(pass)':>9}")
sub=R[(R.gate==1.5)&(R.tp=='0.5xATR30')].sort_values('rr')
for _,r in sub.iterrows():
    print(f"{r['rr']:>5}{r['tpmed']:>7.1f}{r['tpmed']/r['rr']:>8.1f}{r['wr']:>6.1f}%{r['avgW']:>7.0f}{r['avgL']:>7.0f}{r['exp']:>7.0f}{r['ppass']:>8.1f}%")
