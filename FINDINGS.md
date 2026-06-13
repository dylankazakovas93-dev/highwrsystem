# Findings — does the high-win-rate range fade actually exist on NQ?

_Data: continuous front-month NQ, 1-min, 2020-01-01 → 2026-06-07 (1,663 trading days,
2.27M bars), built from Databento GLBX. Dev = 2025-01-01 → 2026-06-07. Holdout =
2020–2024. All figures net of $5 round-turn commission; slippage as labelled._

## HEADLINE: the fade fails, but open-drive CONTINUATION delivers 85%+ across all years

The range *fade* never reaches the goal (details below). But a different, better-motivated
strategy does: **open-drive continuation.** Wait to 11:00 ET, measure the move from the
9:30 RTH open, and if it exceeds a 30-min-ATR gate, trade *in the direction of the move*
(the morning trend continues into the afternoon). Target and stop scale with the same ATR.

**Best config — 11:00 snapshot, entry gate ≥ 2.0×ATR30, TP = 0.5×ATR30, stop = 10×TP
(RR 0.1), one trade/day, flat 15:55:**

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | overall | exp/trade | trades/yr |
|---|---|---|---|---|---|---|---|---|---|
| 84% | 94% | 86% | 88% | 81% | 83% | 85% | **85.8%** | +$124 | ~72 |

- **Win rate 81–94% in every one of 7 years** — the stability the brief demanded.
- Survives slippage: 85.6–86.0% WR at 0→3 ticks/side; net ~$58–63k over 501 trades.
- The **fade control loses** (−$112 to −$140/trade every year): afternoons continue the
  morning drive, they do not revert. Internal consistency ⇒ real effect, not curve-fit.
- ATR-normalized gate/target means no year-fitting: dollar target auto-scales $353 (2020)
  → $680 (2026); win rate stays flat across the price-regime change.

**Two levers produce the win rate:** (1) a *demanding* entry gate (≥2×ATR move by 11:00 =
only genuine trend days; raising 1.0→2.5× lifts WR 71%→76%), and (2) a *modest* target
(0.5×ATR; shrinking TP 1.0→0.5× lifts WR 71%→86%). 11:00 beats 12:00 by ~6 WR points at
every setting (more afternoon left for the trend to extend).

**The 10×ATR stop is almost never hit (≈0.4% of trades), so it can be CAPPED cheaply.**
Replacing the 10×ATR stop with a hard 200-pt cap costs only 0.6% win rate (85.8% → 85.2%)
and roughly halves the worst loss (−$7,237 → −$4,010). Capping schedule (gate 2.0, TP 0.5):

| stop cap | overall WR | worst loss | exp/trade |
|---|---|---|---|
| 200 pts | 85.2% | −$4,010 | +$80 |
| 150 pts | 84.8% | −$3,010 | +$89 |
| 100 pts | 82.0% | −$2,010 | +$86 |

**Practical setup:** wait to 11:00 ET; if |move from 9:30 open| ≥ 2×ATR30, enter in that
direction; TP 0.5×ATR; **stop 200 pts (or 150)**; flat 15:55; one trade/day. ~85% WR,
stable every year, worst day ≈ −$4k.

**The honest catch — negative skew (bounded, not eliminated).** Even capped, avg win ≈$490
vs avg loss ≈$2,300: frequent small wins, occasional bounded-but-large loss. Net positive
(+$80/trade). At $4k risk this is a funded/personal-account setup; still too big for a $2k
prop buffer (a 100-pt/$2k cap holds 82% WR if a prop-sized version is needed). You can have
85% WR or a tight prop stop, not both. Reproduce: `scripts/open_drive_stats.py` (sweeps)
and the capped-stop sweep in the commit history.

---

## The original question: does the range FADE hit 80%? Short answer

**No — not at 80%, and not robustly even at 70%.** The unconditioned edge-fade has no
edge (47% WR, exactly what the random-walk math predicts for a 15-pt target against a
~16-pt stop). Conditioning the setup down to its single best recent-regime pocket — the
**NY-lunch session, shorts only** — gets dev win rate to **64.8%** with positive
expectancy and a short max losing streak, but that win rate **does not hold out of
sample** (53.3% on 2020–2024) and its own 95% confidence interval is [53%, 75%]. The
"high-probability dice roll" the brief hoped for is not present in this data as a stable,
structural effect. What exists is a modest, regime-dependent shorts-fade in midday balance.

## The numbers that matter

| config | period | n | win rate | 95% CI | exp/trade | PF | max streak |
|---|---|---|---|---|---|---|---|
| all defaults | dev 25–26 | 313 | 47.3% | [41.8, 52.8] | −$37.68 | 0.78 | 7 |
| lunch + shorts | dev 25–26 | 71 | **64.8%** | [53.2, 74.9] | +$67.68 | 1.57 | 3 |
| lunch + shorts | holdout 20–24 | 255 | 53.3% | [47.2, 59.4] | −$3.78 | 0.98 | 10 |

Win rate of the candidate **by year**: 2020 48% · 2021 53% · 2022 57% · 2023 63% ·
2024 49% · 2025 66% · 2026 56%. The good years (2023, 2025) carry it; there is no year
where it clears 70%, and two holdout years sit below 50%. This is the signature of a
regime-sensitive tilt, not a durable mechanical edge.

Slippage is not the killer here — the candidate still shows 62% WR and positive
expectancy at a punishing 3 ticks/side. The killer is **win-rate instability across
time**, which is exactly what the pre-registered rejection rules test for.

## How we got there (audit trail)

1. **Baseline** (registered hypothesis, all defaults): 47.3% WR, −$38/trade on dev.
   Matches the no-edge break-even (P(TP before SL) = SL/(TP+SL) ≈ 52% before costs).
   So the raw probe→reclaim fade with a 15-pt target carries no edge on its own.
2. **OFAT sweep** (73 one-axis variants, full history, pre-registered rejection rules):
   **zero survivors.** Every variant was flagged `wr_collapse` (WR < 70% in some
   ≥15-trade segment) and all but one `dies_with_slippage`. Directional signal, though:
   the strongest single lever was restricting to the **NY-lunch session** (60% WR dev),
   and shorts beat longs in every session.
3. **Conditional diagnostics** on the baseline trade set confirmed: shorts > longs
   (51% vs 43%), lunch/morning RTH sessions > overnight, and small "real" edge pokes
   (0.5–3 pt overshoot then reclaim) won more than flat touches. Combining the two
   robust levers (lunch + shorts) produced the candidate above.
4. **Stage-2 random sweep** (250 combos over the promising region): see below.

## Stage-2 sweep result

<!-- STAGE2 -->
_(filled in when out/grid_stage2/grid_report.md completes)_

## Prop-firm game (candidate, dev distribution, trailing $2k / $3k payout / 1 contract)

| metric | value |
|---|---|
| P(payout before failure) | 93.4% |
| P(account failure) | 6.6% |
| median trading days to payout | 37 |
| EV per attempt | +$2,356 |
| EV with up to 3 restarts | +$2,523 |

…but run the **same simulator on the holdout distribution** and P(payout) falls to
**20%**, P(fail) **78%**. The prop result is entirely a function of which regime's trade
distribution you feed it — it does not add an edge, it just amplifies whatever the win
rate already is. That is the central trap the brief asked us to avoid, made quantitative.

**Risk sensitivity (dev stream):** going from 1→2→3 contracts *lowers* P(payout)
(93%→67%→56%) while shortening time-to-payout — classic variance-vs-ruin tradeoff. For a
$2k trailing buffer, **1 contract is the only sane size**; 2+ trades the high win rate
away through drawdown breaches.

## Rescue mode (does taking the setup from a drawdown state help?)

Comparing three policies from a given drawdown state, with the candidate's dev
distribution as the "rescue" setup and a synthetic 45%-WR strategy as the "baseline" the
account got into trouble with:

| drawdown state | continue baseline | switch to rescue | stop trading |
|---|---|---|---|
| down 25% of buffer | 1.3% payout / 99% fail | **12.5% payout / 8% fail** | 0 / 0 (frozen) |
| down 50% | 0.7% / 99% | **6.8% / 19%** | 0 / 0 |
| down 80% | 0.1% / 99.9% | **2.6% / 45%** | 0 / 0 |

Switching to the disciplined setup dominates continuing a losing strategy at every
drawdown depth, and mostly "survives without paying out" rather than blowing up — because
it trades far less often (only on valid lunch setups). But note this uses the *favorable*
dev distribution; on the holdout distribution the rescue edge largely evaporates. The
honest takeaway: **discipline + low frequency beats tilt, but neither manufactures an edge
that isn't in the market.** Account state correctly never generates a signal here — it only
gates whether a valid market setup is taken.

## What would change the conclusion

- **More 2025–2026 data.** The candidate has only 71 dev trades; the CI is too wide to
  assert 70%. A forward sample of another 100+ lunch-session days would tighten it.
- **Tick data** for honest limit-fill modeling (entry type B) and finer stop behavior.
- **A genuinely different conditioning variable** (e.g. a volatility/term-structure
  regime flag, or order-flow imbalance) rather than more slicing of the same OHLCV — the
  OHLCV-derivable filters have been swept and none stabilizes the win rate.

## Recommended forward-test protocol (if pursued at all)

1. Trade **NQ, lunch session (12:00–15:00 ET), shorts only**, the registered candidate
   config, **1 contract**, 1 trade/day, stop-after-loss.
2. Pre-commit to a **70% win-rate floor over a rolling 40-trade window**; if it breaches,
   stop — this is the live version of the rejection rule that the holdout already fails.
3. Treat it as a **regime bet, not a mechanical edge.** Expect ~55–65% WR, not 80%.
4. Paper/SIM-trade ≥ 40 setups before risking an evaluation; the dev edge must reproduce
   forward before it funds anything.
