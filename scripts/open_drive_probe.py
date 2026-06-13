"""Open-drive continuation probe: trade the direction of the 9:30->snapshot move,
gated by 30-min ATR, TP = mult*ATR30, SL = TP/RR. Reuses the conservative bracket engine."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from rangefade.config import load_config
from rangefade.data import load_ohlcv, add_trading_day, filter_period
from rangefade.engine import run_bracket, market_fill
from rangefade.metrics import wilson_ci

TICK, PV, COMM = 0.25, 20.0, 5.0
SLIP = 1 * TICK

def atr30_prior(df):
    """ATR(14) on 30-min bars, taken as of the PRIOR trading day's last bar (no lookahead)."""
    g = df.resample('30min', label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last'}).dropna()
    pc = g['close'].shift(1)
    tr = pd.concat([g['high']-g['low'], (g['high']-pc).abs(), (g['low']-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()
    td = add_trading_day(g.assign(_tr=tr), 18)['trading_day']
    daily = atr.groupby(td.values).last()
    return pd.Series(daily).shift(1)  # prior-day value, indexed by trading_day

def run(df, snapshot='11:00', entry_atr_mult=1.0, tp_mult=1.0, rr=0.3,
        direction='cont', flat='15:55'):
    """direction: 'cont' (momentum) or 'fade' (reversion)."""
    o=df['open'].to_numpy(float); h=df['high'].to_numpy(float)
    l=df['low'].to_numpy(float); c=df['close'].to_numpy(float)
    idx=df.index; mins=(idx.hour*60+idx.minute).to_numpy()
    bar_date=np.array([d.date() for d in idx])  # calendar date of each bar
    day_codes,_=pd.factorize(df['trading_day'])
    atr_map=atr30_prior(df).to_dict()
    sh,sm=map(int,snapshot.split(':')); snap_min=sh*60+sm
    fh,fm=map(int,flat.split(':')); flat_min=fh*60+fm
    rth_open=9*60+30
    trades=[]
    for dc in np.unique(day_codes):
        pos=np.nonzero(day_codes==dc)[0]
        td=df['trading_day'].iloc[pos[0]]
        atr=atr_map.get(td, np.nan)
        if not np.isfinite(atr) or atr<=0: continue
        dmin=mins[pos]; ddate=bar_date[pos]
        # RTH bars only: calendar date == trading_day (daytime session), 09:30-16:00
        op=pos[(ddate==td)&(dmin>=rth_open)&(dmin<16*60)]
        if len(op)==0: continue
        open930=o[op[0]]
        sp=pos[(ddate==td)&(dmin>=snap_min)&(dmin<16*60)]
        if len(sp)==0: continue
        snap_i=sp[0]; snap_px=c[snap_i]
        move=snap_px-open930
        if abs(move)<entry_atr_mult*atr: continue
        sgn=np.sign(move)
        s=int(sgn) if direction=='cont' else int(-sgn)
        ent_i=snap_i+1
        fp=pos[(ddate==td)&(dmin>=flat_min)]
        flat_i=(fp[0] if len(fp) else pos[-1])
        if ent_i>=flat_i: continue
        entry=market_fill(s,o[ent_i],SLIP)
        tp_dist=tp_mult*atr; sl_dist=tp_dist/rr
        tp=entry+s*tp_dist; stop=entry-s*sl_dist
        bx=run_bracket(s,ent_i,entry,stop,tp,o,h,l,c,flat_i,TICK,SLIP,1,'loss')
        pnl=s*(bx.exit_px-entry)*PV-COMM
        trades.append({'td':td,'side':s,'reason':bx.reason,'pnl':pnl,'move':move,'atr':atr})
    return pd.DataFrame(trades)

df=load_ohlcv('data/NQ_continuous.parquet'); df=add_trading_day(df,18)
years=[2020,2021,2022,2023,2024,2025,2026]

def grid(direction, snapshot, tp_mult):
    print(f"\n===== direction={direction}  snapshot={snapshot}  TP={tp_mult}xATR30  (entry gate=1.0xATR30) =====")
    print(f"{'RR':>5} "+" ".join(f"{y:>6}" for y in years)+f"  {'ALL_wr':>7} {'ALL_n':>6} {'exp$':>8}")
    for rr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
        allt=run(filter_period(df,'2020-01-01','2026-12-31'),
                 snapshot, 1.0, tp_mult, rr, direction)
        if allt.empty:
            print(f"{rr:>5} (no trades)"); continue
        allt['y']=pd.to_datetime(allt['td']).dt.year if not np.issubdtype(type(allt['td'].iloc[0]),np.datetime64) else allt['td']
        allt['y']=allt['td'].map(lambda d:d.year)
        allt['win']=allt['pnl']>0
        cells=[]
        for y in years:
            sub=allt[allt['y']==y]
            cells.append(f"{(sub['win'].mean()*100):>5.0f}%" if len(sub)>=10 else f"{'n'+str(len(sub)):>6}")
        wr=allt['win'].mean(); exp=allt['pnl'].mean()
        print(f"{rr:>5} "+" ".join(cells)+f"  {wr*100:>6.1f}% {len(allt):>6} {exp:>8.1f}")

grid('cont','11:00',1.0)
grid('cont','12:00',1.0)
grid('cont','11:00',2.0)
grid('fade','11:00',1.0)

def robustness():
    import numpy as np
    dall=filter_period(df,'2020-01-01','2026-12-31')
    print("\n\n########## ROBUSTNESS (continuation, 11:00, TP=1xATR30) ##########")
    print("\n--- entry-gate sensitivity (RR=0.3) ---")
    print(f"{'gate':>5} {'n':>5} {'wr':>6} {'exp$':>8} {'/yr':>5}")
    for gate in [0.5,0.75,1.0,1.25,1.5,2.0]:
        t=run(dall,'11:00',gate,1.0,0.3,'cont')
        if len(t)<20: print(f"{gate:>5} {len(t):>5} (thin)"); continue
        print(f"{gate:>5} {len(t):>5} {t['pnl'].gt(0).mean()*100:>5.0f}% {t['pnl'].mean():>8.1f} {len(t)/7:>5.0f}")
    print("\n--- slippage stress (gate=1.0, RR=0.3) ---")
    import rangefade.engine as eng
    global SLIP
    for ticks in [0,1,2,3]:
        SLIP=ticks*TICK
        t=run(dall,'11:00',1.0,1.0,0.3,'cont')
        print(f"slip {ticks}t: wr={t['pnl'].gt(0).mean()*100:.0f}% exp=${t['pnl'].mean():.1f} net=${t['pnl'].sum():.0f}")
    SLIP=1*TICK
    print("\n--- per-year max losing streak & worst loss (gate=1.0) ---")
    print(f"{'RR':>4} "+" ".join(f"{y:>11}" for y in years))
    for rr in [0.1,0.3,0.5,0.8]:
        t=run(dall,'11:00',1.0,1.0,rr,'cont'); t['y']=t['td'].map(lambda d:d.year)
        cells=[]
        for y in years:
            s=t[t['y']==y].sort_values('td'); 
            if len(s)<10: cells.append(f"{'n'+str(len(s)):>11}"); continue
            pnl=s['pnl'].to_numpy(); strk=mx=0
            for x in pnl:
                strk=strk+1 if x<0 else 0; mx=max(mx,strk)
            cells.append(f"{str(mx)+'L/'+str(int(pnl.min())):>11}")
        print(f"{rr:>4} "+" ".join(cells))
robustness()
