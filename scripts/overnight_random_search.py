"""Random search over the overnight (first-candle-after-18:00) strategy.
Ranking criterion (user-specified): BALANCED year-to-year first, then WR, then PF.
 - balance = the WORST single-year WR (maximize it) among years with >=8 trades
 - require >=6 of 7 years present and >=100 total trades (must work across regimes)
 - then sort by overall WR, then overall PF.
Honest note: random search overfits by construction; treat the top rows as CANDIDATES to
walk-forward, not as validated edges."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import scripts.keeper_config as K
from rangefade.engine import run_bracket, market_fill

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
TFS = [15, 30, 60, 240]

df = K.add_trading_day(K.load_ohlcv("data/NQ_continuous.parquet"), 18)
df = K.filter_period(df, "2020-01-01", "2026-12-31")
o=df["open"].to_numpy(float);h=df["high"].to_numpy(float);l=df["low"].to_numpy(float);c=df["close"].to_numpy(float)
mins=(df.index.hour*60+df.index.minute).to_numpy();bd=np.array([d.date() for d in df.index])
dcs,_=pd.factorize(df["trading_day"])
ATRTF={tf:K.atr_prior(df,f"{tf}min") for tf in TFS}

# Precompute, per tf, per qualifying day: (year, move, atr, entry_i, flat_i)
PRE={}
for tf in TFS:
    recs=[]
    for d in np.unique(dcs):
        pos=np.nonzero(dcs==d)[0];td=df["trading_day"].iloc[pos[0]]
        a=ATRTF[tf].get(td,np.nan)
        if not(np.isfinite(a)and a>0):continue
        dm=mins[pos]
        if mins[pos[0]]!=18*60:continue
        elapsed=(dm-18*60)%1440
        cand=pos[elapsed<tf]
        if len(cand)<max(3,tf//3):continue
        move=c[cand[-1]]-o[cand[0]]
        ei=cand[-1]+1
        rth=pos[(bd[pos]==td)&(dm>=570)]
        if len(rth)==0:continue
        flat_i=rth[0]
        if ei>=flat_i:continue
        recs.append((td.year,move,a,ei,flat_i))
    PRE[tf]=recs

def evaluate(tf, thr, follow, mode, tp_mult, rr):
    rows=[]
    for yr,move,a,ei,flat_i in PRE[tf]:
        if follow:
            if abs(move)<thr*a:continue
            s=int(np.sign(move))
        else:
            if move<thr*a:continue
            s=1
        entry=market_fill(s,o[ei],SLIP)
        if mode=='time':
            px=market_fill(-s,c[flat_i],SLIP); pnl=s*(px-entry)*PV-COMM
        else:
            tp_d=tp_mult*a; sl_d=tp_d/rr
            bx=run_bracket(s,ei,entry,entry-s*sl_d,entry+s*tp_d,o,h,l,c,flat_i,TICK,SLIP,1,"loss")
            pnl=s*(bx.exit_px-entry)*PV-COMM
        rows.append((yr,pnl))
    return rows

def score(rows):
    if len(rows)<100:return None
    t=pd.DataFrame(rows,columns=['y','pnl']); t['win']=t.pnl>0
    peryr={}
    for y in YEARS:
        s=t[t.y==y]
        if len(s)<8:continue
        gw=s.pnl[s.pnl>0].sum();gl=-s.pnl[s.pnl<0].sum()
        peryr[y]=(s.win.mean(), gw/gl if gl>0 else 9.9, len(s))
    if len(peryr)<6:return None
    min_wr=min(v[0] for v in peryr.values())
    gw=t.pnl[t.pnl>0].sum();gl=-t.pnl[t.pnl<0].sum()
    return {'n':len(t),'min_yr_wr':min_wr,'wr':t.win.mean(),'pf':gw/gl if gl>0 else 9.9,
            'peryr':{y:v[0] for y,v in peryr.items()},'exp':t.pnl.mean()}

rng=np.random.default_rng(20260613)
seen=set(); results=[]
N=400
while len(results)<N and len(seen)<5000:
    tf=int(rng.choice(TFS)); thr=round(float(rng.uniform(0.2,1.2)),2)
    follow=bool(rng.integers(0,2)); mode=str(rng.choice(['time','bracket']))
    tp_mult=round(float(rng.uniform(0.4,1.5)),2); rr=float(rng.choice([0.5,0.75,1.0,1.5,2.0]))
    key=(tf,thr,follow,mode,tp_mult if mode=='bracket' else 0,rr if mode=='bracket' else 0)
    if key in seen:continue
    seen.add(key)
    sc=score(evaluate(tf,thr,follow,mode,tp_mult,rr))
    if sc is None:continue
    sc.update({'tf':tf,'thr':thr,'dir':'follow' if follow else 'long','mode':mode,
               'tp':tp_mult if mode=='bracket' else None,'rr':rr if mode=='bracket' else None})
    results.append(sc)

R=pd.DataFrame(results)
# RANK: balanced (min-year WR) -> WR -> PF
R=R.sort_values(['min_yr_wr','wr','pf'],ascending=False).reset_index(drop=True)
print(f"Overnight random search: {len(R)} valid combos (of {len(seen)} tried). "
      f"Ranked by min-year-WR, then WR, then PF.\n")
hdr=f"{'tf':>4}{'thr':>5}{'dir':>7}{'mode':>8}{'tp':>5}{'rr':>5}{'n':>5}{'minWR':>7}{'WR':>6}{'PF':>6} | per-year WR"
print(hdr); print("-"*len(hdr))
for _,r in R.head(25).iterrows():
    py=" ".join(f"{int(r['peryr'].get(y,float('nan')*0)*100):>3}" if y in r['peryr'] else "  ." for y in YEARS)
    tp=f"{r['tp']:.2f}" if r['tp'] else "  - "; rr=f"{r['rr']:.2f}" if r['rr'] else "  - "
    print(f"{r['tf']:>4}{r['thr']:>5}{r['dir']:>7}{r['mode']:>8}{tp:>5}{rr:>5}{r['n']:>5}"
          f"{r['min_yr_wr']*100:>6.1f}%{r['wr']*100:>5.1f}%{r['pf']:>6.2f} | {py}")
print("\n(per-year WR columns = 2020 2021 2022 2023 2024 2025 2026)")
