"""Monthly eval pass-rate MC for the best RR-0.75 open-drive config.
Reuses eval_rr075 (RR=0.75, TP calibrated ~11pt in 2026). Uses the RECENT (2024-26)
per-1-contract trade distribution and frequency; simulates 1-month windows to
+TARGET before -DD (trailing). Sweeps contract size. Shows pass rate tops ~49%."""
import sys; sys.path.insert(0, '.')
import numpy as np
import scripts.eval_rr075 as R

def recent(gate, tf):
    ents=R.entries(gate); m=R.calib_mult(tf,ents); t=R.backtest(ents,tf,m)
    t=t[t.y>=2024]; return t.pnl.to_numpy()

def monthly_pass(pnl1, per_month, contracts, target=3000, dd=2000, npaths=20000, seed=3):
    rng=np.random.default_rng(seed); wins=0
    for _ in range(npaths):
        k=max(1,rng.poisson(per_month)); picks=pnl1[rng.integers(0,len(pnl1),k)]*contracts
        eq=peak=0.0; ok=False
        for x in picks:
            eq+=x; peak=max(peak,eq)
            if eq-peak<=-dd: break
            if eq>=target: ok=True; break
        wins+=ok
    return wins/npaths

if __name__=="__main__":
    for gate,tf in [(2.0,'60min'),(2.5,'60min')]:
        pnl1=recent(gate,tf); pm=len(pnl1)/(2.5*12)
        print(f"\ngate {gate} {tf} RR0.75 | recent WR {(pnl1>0).mean()*100:.1f}% | {pm:.1f} trades/mo")
        for n in [3,5,7,10]:
            print(f"  {n:>2} contracts: P(pass/month)={monthly_pass(pnl1,pm,n)*100:.1f}%")
