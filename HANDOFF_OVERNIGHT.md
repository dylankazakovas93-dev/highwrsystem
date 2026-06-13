# Codex handoff — overnight first-candle strategy random search

Independently implement the strategy below on NQ 1-min data, run a random search over the
parameter space, and rank by: **(1) balanced year-to-year, (2) win rate, (3) profit factor**
— in that order. Report your top ~25. We want to cross-check against an independent run.

## Data
NQ continuous front-month, 1-minute OHLCV, timezone **America/New_York**. CME trading day
rolls at **18:00 ET** (a bar at/after 18:00 belongs to the next trading day). Period
2020-01-01 → 2026-06-07. (Source: Databento GLBX `NQ.FUT`, volume-rolled, unadjusted.)

## ATR (critical — do not use Wilder/`ta.atr`)
ATR = **simple moving average of True Range over 14 bars**, TR = max(H−L, |H−prevC|,
|L−prevC|), computed on resampled bars of the relevant timeframe (`label='right',
closed='right'`). Use each trading day's **last** value, then the **prior** trading day's
value (shift 1) → no lookahead.

## Strategy: first candle after 18:00
For each trading day:
1. Take the **first candle of timeframe `tf`** starting at 18:00 ET (tf ∈ {15, 30, 60, 240}
   minutes — the candle spans [18:00, 18:00+tf)).
2. `move = close(candle) − open(candle)`. `atr = ATR(tf, prior day)`.
3. Entry trigger:
   - `dir = "long"`: enter **long** if `move ≥ thr·atr`.
   - `dir = "follow"`: enter **sign(move)** if `|move| ≥ thr·atr`.
4. Fill at the **next 1-min bar open** after the candle closes, + **1 tick** slippage.
5. Exit (`mode`):
   - `"time"`: market exit at the **09:30 RTH open** bar (close ± 1 tick).
   - `"bracket"`: TP = `tp_mult·atr` (fills at limit), SL = `tp_mult·atr / rr` (fills at stop
     or worse on a gap, +1 tick), hard deadline at 09:30. A bar touching **both** = the
     **stop (loss)**.
6. One trade per day. Costs: **$5 round-turn**, NQ point value **$20**, 1 tick = **0.25 pt**.

## Random search space
```
tf       ∈ {15, 30, 60, 240}
thr      ~ Uniform[0.2, 1.2]
dir      ∈ {long, follow}
mode     ∈ {time, bracket}
tp_mult  ~ Uniform[0.4, 1.5]     # bracket only
rr       ∈ {0.5, 0.75, 1.0, 1.5, 2.0}   # bracket only
```
Draw ~400 unique combos.

## Validity filter (must work across regimes)
Keep a combo only if: **≥ 100 total trades** AND **≥ 6 of the 7 years** have **≥ 8 trades**.

## Ranking (in this exact order)
1. **Balanced y/y** = maximize the **minimum single-year win rate** (over years with ≥8 trades).
2. then **overall win rate** (desc).
3. then **overall profit factor** (desc).

Report the top 25 with: tf, thr, dir, mode, tp_mult, rr, n, min-year WR, overall WR,
overall PF, and the per-year WR for 2020–2026.

## Cross-check targets (my independent run, seed 20260613)
Top balanced rows I got (yours should land near these if the spec matches):
```
 tf  thr   dir    mode   tp   rr    n   minWR   WR    PF | per-yr WR 20-26
 15 0.65 follow bracket 0.57 0.50  400  66.0% 69.8% 1.15 | 66 76 73 69 66 70 68
 15 0.65  long  bracket 0.98 0.50  202  65.5% 68.3% 1.14 | 67 65 70 69 69 66 73
 15 0.52  long  bracket 0.68 0.50  281  62.9% 70.5% 1.43 | 77 65 64 62 66 75 75
 30 0.33 follow bracket 0.51 0.50  590  62.1% 68.1% 1.14 | 62 63 72 71 71 67 74
```
Two findings to confirm or refute: (a) the balanced/high-WR rows cluster at **rr = 0.50**
(wide stop), and (b) requiring **rr ≥ 0.75** drops the best min-year WR to ~54%.

## Important caveats (please report on these, don't just hand back the top row)
- Random search overfits by construction — flag how sensitive the top rows are to small
  parameter nudges, and whether the leaders survive a **walk-forward** (fit 2020–2023, then
  evaluate 2024–2026 once, untouched).
- Note that high WR at rr = 0.5 is **concave** (loss = 2× win); report avg-win/avg-loss $ so
  the payoff asymmetry is visible, not hidden by the win rate.
