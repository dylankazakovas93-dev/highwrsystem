# Handoff: independent verification of the "open-drive continuation" NQ strategy

## What I need from you (Codex)
Independently re-implement and verify the intraday strategy specified below on NQ
1-minute data (2020-01-01 → 2026-06-07). Reproduce the year-by-year win rate / profit
factor and tell me whether they hold up. **If your numbers structurally match (the
"green light" criteria at the bottom), we will then build this EXACT config as a
TradingView Pine v6 strategy (or indicator).** Do not change the rules; verify them.

A reference Python implementation already exists (`scripts/keeper_config.py`, ~120 lines,
depends only on pandas/numpy + a tiny bracket engine in `rangefade/engine.py`). You may
run it, or re-implement from scratch in your own engine and compare — re-implementing from
scratch is the stronger check.

## Thesis (why this should work)
On a trending NY session, the morning direction tends to *continue* into the early
afternoon (classic "trend day" behavior). So: wait until 11:00 ET, measure how far price
has traveled from the 09:30 RTH open, and if that move is large relative to recent
volatility (an ATR gate), enter **in the direction of the move**. Take a small
volatility-scaled profit; protect with a wide structural stop that is rarely hit; and
time-exit anything that hasn't resolved. The control test (fading the move instead of
continuing) loses money every year, which supports the effect being real rather than fit.

## EXACT specification

**Instrument / data**
- NQ continuous front-month, 1-minute OHLCV. Timezone **America/New_York** throughout.
- CME trading day rolls at **18:00 ET** (a bar at/after 18:00 belongs to the next
  trading day). RTH open = the **09:30 ET** bar.
- Source used here: Databento GLBX.MDP3 `NQ.FUT`, stitched to a continuous front month by
  volume (roll at the 18:00 boundary), **prices unadjusted**. (Build script:
  `scripts/build_continuous.py`; output `data/NQ_continuous.parquet`.)

**ATR definition (read carefully — this is the #1 reproduction gotcha)**
- ATR = **simple moving average of True Range over 14 bars**, NOT Wilder's RMA.
  TR = max(High−Low, |High−prevClose|, |Low−prevClose|).
- Computed on **resampled bars** with `label='right', closed='right'`.
- Take each **trading day's last** ATR value, then use the **prior trading day's** value
  (shift by one day) so there is **no same-day lookahead**.
- Two ATRs are used: **30-minute ATR** for the entry gate, **15-minute ATR** for the target.

**Entry (one trade per trading day)**
1. At **11:00 ET**, compute `move = close(11:00 bar) − open(09:30 RTH bar)`.
2. Trade only if `abs(move) >= 1.5 × ATR30_prior`. Otherwise no trade that day.
3. Direction = `sign(move)` → **long if move > 0, short if move < 0** (continuation).
4. Fill on the **next bar's open (11:01)**, worsened by **1 tick** slippage.

**Exits — whichever comes first**
- **Target:** `TP_dist = 0.40 × ATR15_prior`; TP price = entry + dir×TP_dist. Fills at the
  limit price (no slippage on TP).
- **Structural stop (always active):** `stop_dist = TP_dist / 0.2` (= 5×TP_dist = 2.0×ATR15),
  **capped at 200 points**. Fills at the stop price, or at the bar open if it gaps through,
  worsened by 1 tick.
- **Timed exit:** if neither TP nor stop is hit within **90 minutes** of entry, exit at
  **market** (bar close, worsened by 1 tick). Hard backstop: flatten at **15:55 ET** if 90
  minutes would run past it.
- **Both-touch rule:** if a single bar touches BOTH the stop and the target, count it as the
  **stop (a loss)** — conservative.

**Costs**
- Commission **$5.00 round-turn**; NQ **point value $20**; **1 tick = 0.25 pt**, slippage
  **1 tick per side** on entry, stop, and market/time exits (not on TP limit fills).

## Verification targets (my reference engine, 2020-01-01 → 2026-06-07)
```
  Year  Trades  WinRate     PF  TPmed(pts)  SLmed(pts)    net$
  2020      93    79.6%   0.80      9.0        44.8      -3,596
  2021     108    88.9%   2.21      8.3        41.6       9,378
  2022     116    82.8%   1.26     13.6        67.8       5,970
  2023     102    88.2%   1.73      8.9        44.8       7,226
  2024     108    82.4%   1.06     11.4        56.9       1,211
  2025     111    84.7%   1.41     13.1        65.4       8,193
  2026      58    89.7%   2.13     15.6        77.8       9,073
  ALL      696    84.9%   1.35     10.9        54.6      37,456
```
Overall: **696 trades, ~99/yr, 84.9% win rate, PF 1.35, +$54/trade, +$37,456.**
Exit-reason mix: 591 target / 80 stop / 25 timed. Targets are ~13–16 pts in 2025–26 and
smaller in earlier (lower-ATR) years — they scale with ATR by design.

## "Green light" criteria (what counts as verified)
Exact ticks will differ with your fill/slippage modeling; I expect overall win rate to move
by ≤ ~1.5 points and net by ≤ ~15%. Call it **verified** if all of these hold:
1. Overall win rate **83–86%**, every individual year **≥ 78%** (2020 is the weakest).
2. Every year profitable **except 2020** (the COVID-whipsaw year may be ~breakeven/negative).
3. Overall **PF ≥ 1.25**; ~**90–105 trades/year**.
4. The **fade** (same rules, opposite direction) is **net negative** every year.
5. Results are **not driven by 1–2 trades** and survive **2 ticks/side** slippage
   (win rate should barely move; this strategy's stop is hit only ~11% of the time).

## Things most likely to break a reproduction (check these first if you diverge)
- Using **Wilder's ATR (`ta.atr`)** instead of **SMA(14) of TR** — most common cause.
- Same-day ATR lookahead instead of **prior trading day's** value.
- Wrong session anchoring: trading day must roll at **18:00 ET**, RTH open at **09:30 ET**,
  snapshot at **11:00 ET**, in **America/New_York** (handle DST).
- Counting both-touch bars as wins instead of **losses**.
- Letting the timed exit replace the structural stop — **both** are active; the 90-min exit
  only acts on trades not already stopped/targeted.
- Filling the entry on the 11:00 bar instead of the **next** bar's open.

## After a green light — Pine v6 build notes (for later, not now)
- Implement as a `strategy` (or `indicator` with alerts). It is fully codeable.
- ATR: `request.security(syminfo.tickerid, "30", ta.sma(ta.tr, 14))` and the same on "15",
  using the **prior day's last completed value** (offset so there's no repaint/lookahead).
  Do **not** use `ta.atr` (Wilder).
- Use `timestamp`/session logic in `"America/New_York"`; gate at 11:00, hard flat 15:55.
- One entry/day; bracket via `strategy.exit` (limit = TP, stop = structural stop); add a
  bar-count/time check for the 90-minute market exit.
- Expect Pine's intrabar stop/TP resolution to differ slightly from a 1-min backtest; enable
  bar-magnifier / `calc_on_every_tick` and treat fills as slightly worse than ideal.

## Repo pointers
- `scripts/keeper_config.py` — canonical reference (run it; prints the table above).
- `rangefade/engine.py` — `run_bracket` (the conservative fill engine the reference uses).
- `scripts/build_continuous.py` — rebuilds `data/NQ_continuous.parquet` from Databento raw.
- `FINDINGS.md` / `THESIS.md` — how this config was arrived at and what was rejected.
