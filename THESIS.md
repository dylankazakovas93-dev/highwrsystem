# Thesis: consolidation range-rotation fade on NQ/ES under prop-firm constraints

## Research question

Can we identify a recent-regime (2025–2026) NQ or ES consolidation environment where
fading **failed range-edge moves** yields a **high win rate (target: 80%+, floor: 70%)**
with a **meaningful fixed target (NQ ≥ 15 points, ES ≥ ~6 points)** that **stays inside
the consolidation**, a **hard structural stop**, and bounded, prop-rule-compatible
downside — usable selectively to rescue/extract a prop account rather than as a
portfolio strategy?

## The setup grammar (what a trade IS)

```
wide accepted consolidation          (width 40–100 NQ pts, two-way rotation proven)
  -> probe into/through an edge zone (upper/lower 10–20% of the box, or VAH/VAL)
  -> failure to continue             (overshoot capped: beyond ~10 pts = breakout, no trade)
  -> reclaim close back inside       (the actual signal)
  -> entry (A: market on reclaim close | B: limit on edge retest | C: micro-swing break)
  -> fixed internal target           (15 pts; must clear the opposite edge — else SKIP)
  -> hard structural stop            (edge/excursion + 5–10 pts, capped at 25 pts)
```

The trade is defined by the **wide accepted range**, not by VAH/VAL. Volume-profile
levels are one optional edge definition/confirmation, never the whole setup.

**Skip, don't shrink:** if a 15-pt target cannot fit inside the box from a realistic
entry price, there is no trade. The minimum *usable* width follows from geometry: with
entry ~10% inside the box, the target must clear the far edge, so width must be
≥ TP / 0.9 ≈ 17 pts at the bare minimum — but acceptance filters and stop geometry push
the practical floor to 30–60 pts, which is why `min_width_points` is a first-class sweep
axis (30/40/50/60/75/100).

## Why it could work (mechanism)

1. **Balance persistence.** Auction theory: once a session establishes two-way trade
   (rotations through the mid, POC building), continuation of balance is more likely
   than initiative breakout *per individual edge touch*. Most breakout attempts from
   accepted value fail and rotate back.
2. **Stop-run liquidity.** Failed edge pokes are frequently liquidity grabs; the
   reclaim marks trapped breakout traders whose unwind powers rotation back toward the
   mid — exactly the 15-pt corridor we target.
3. **Entry location asymmetry.** Entering ~10% inside a 60-pt box, the 15-pt target
   points *into* the box (downhill in a mean-reverting regime) while the stop sits
   beyond structure that, conditional on balance, should not trade.

## Why the bar is high (be honest about the math)

For a driftless random walk, P(hit TP before SL) = SL/(TP+SL). To reach 80% WR with
TP=15 you'd need SL=60 under no-edge conditions — four times the risk. With the actual
brackets tested (SL 10–25), the no-edge baseline is only **40–62%**. Everything above
that line must be bought by *conditioning* (regime filters, location, failure/reclaim
sequencing). Measured baseline on real data (2025–2026, all defaults, 313 trades):
**47% WR, −$38/trade** — consistent with the no-edge calculation once costs are
included. The filters must add ~25–30 points of win rate. That is an enormous ask;
this project exists to measure whether any honest subset gets there, not to assume it.

Break-even win rates (15-pt TP, $5 commission RT + 1 tick/side ≈ 0.75 pts cost):

| stop (pts) | BE win rate | exp. at 70% WR | exp. at 80% WR |
|---|---|---|---|
| 10 | 43.0% | +$155/trade | +$185/trade |
| 15 | 51.3% | +$105 | +$135 |
| 20 | 58.0% | +$55  | +$85 |
| 25 | 63.1% | +$5   | +$35 |

So even 70% true WR is comfortably profitable at stops ≤ 15–20 pts. The prop-game
risk is not expectancy, it's **streaks and WR estimation error**:

- P(≥4-loss streak somewhere in 60 trades): ~9% at true WR 80%, ~33% at 70%.
- 4 losses × $400 (20-pt stop) = $1,600 against a $2,000 trailing buffer — survivable
  exactly once. Hence: 1 contract, one trade/day, stop-after-loss, and the Monte Carlo
  sim quantifies P(payout before failure) instead of guessing.
- A backtest WR of 80% on n=50 has a 95% CI of ~[67%, 89%]. **Claiming ≥75% true WR
  requires ~200 trades.** Cells below 40 dev trades are unrankable by design.

## Pre-registered protocol (decided before looking at results)

1. **Primary hypothesis** = `config/nq_default.yaml` exactly: rolling 90-min box,
   width 40–100, TP 15 fixed, stop edge+7.5 capped 25, edge zone 10%, entry A,
   day-move ≤ 0.5%, flat-ish VWAP (≤10 pts/30 min), ≥3 mid-crossings, no fresh
   15-pt session-extreme extension in 30 min, RV ≤ 2× its 20-day median, no news,
   no last hour (flat by 15:00), max 1 trade/day, stop-after-loss, 1 contract.
2. **Dev period**: 2025-01-01 → 2026-06-07. **Holdouts**: 2022–2024, 2020–2021.
   Holdouts are robustness checks, never fitting targets.
3. **Grid discipline**: OFAT first (direction of each axis), then a focused sweep in
   the promising region. Every grid cell faces the same hard rejection rules:
   - < 40 dev trades → rejected (40–79 flagged `low_sample`)
   - WR < 70% in ANY segment with ≥15 trades (dev years, holdout blocks) → rejected
   - top 2 winners > 35% of gross profit → rejected
   - expectancy ≤ 0 at 2 ticks/side slippage → rejected
4. **Ranking among survivors**: worst-segment WR ↓, dev WR ↓, losing streak ↑,
   stressed expectancy ↓. **Net profit is never a ranking key.**
5. **Execution realism is non-negotiable**: market fills at next-bar open ± slip;
   TP limits require trade-through by 1 tick; a bar touching both stop and TP counts
   as a loss; entry limits need trade-through; gapped stops fill at the open, worse.
6. **Prop layer**: trailing-EOD $2,000 buffer, $1,000 daily loss limit, $3,000 payout,
   ≥5 trading days, 50% consistency rule, Monte Carlo over day-resampled sequences;
   rescue-mode comparison (continue baseline vs switch to setup vs stop) at 25/50/80%
   drawdown states. **Account state gates when a valid setup is taken; it never
   creates a signal.**

## Failure modes this design guards against

| failure mode | guard |
|---|---|
| trend/macro transmission day | day-move %, VWAP slope, fresh-extreme extension, RV filters |
| fake "range" that is just a slow trend | adaptive method: extremes age + drift cap; acceptance crossings |
| target only reachable via breakout | hard inside-the-box target check (skip otherwise) |
| perfect-fill fantasy | trade-through fill rules, both-touch = loss, slippage grid 0–3 ticks |
| martingale-adjacent behavior | one bracket, one attempt/side/episode, stop-after-loss, no widening |
| overfit grid winner | pre-registered rejections, holdouts, OFAT-first, min-sample floors |
| roll-gap artifacts (continuous contract) | unadjusted stitch + day-move filter naturally blocks roll days |
| news spikes | optional calendar exclusion ±15 min |

## Current measured state (updated as runs complete)

- Raw defaults, dev 2025–26: 313 trades, **47.3% WR** (CI 41.8–52.8), −$37.7/trade,
  PF 0.78, max streak 7. Shorts 51.2% vs longs 42.6%. Asia absorbs 87% of the daily
  quota under max-1-trade/day. Slippage degrades smoothly (no cliff).
- Conclusion so far: the *unconditioned* edge-fade has no edge at 15/≈16 brackets, as
  the random-walk math predicts. The open question is whether filter/session/stop
  geometry subsets clear 70%+ honestly. See `out/grid_*/grid_report.md`.
