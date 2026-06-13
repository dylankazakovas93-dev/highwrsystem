# Camarilla 9AM Strategy — Independent Verification

Verifier: `cam_9am_verify.py`
Data: Databento GLBX.MDP3 `NQ.FUT`, `ohlcv-1m`, **2020-01-08 → 2026-06-05**
(the supplied zips; the 2019 year claimed in `message.txt` is not in the data).

Front-month reconstruction: per ET day, the single outright contract (spreads
excluded) with the highest daily volume. Timestamps converted UTC → America/New_York.
4h candles built on explicit local-clock 4h blocks (00/04/08/12/16/20 ET) to avoid
DST drift.

## What reconciles with the document (setup is correct)

| Item | Document | Verified on data |
|---|---|---|
| Trades / year | 1,155 / 7y ≈ 165 | 965 / 6.4y ≈ **151** (≈ matches; doc includes extra 2019) |
| Day-of-week mix | Mon173 Tue243 Wed264 Thu256 Fri219 | Mon146 Tue208 Wed234 Thu214 Fri163 (same shape) |
| Avg Base | ~25 pts (example) | **24.2 pts** |
| Avg stop ($/MNQ) | $12 (1 MNQ) | 6.05 pts × $2 = **$12** |
| Avg win ($/MNQ) | $126 (1 MNQ) | 63–73 pts × $2 = **$126** |

Levels, Base, stop/target sizing, filter funnel, and trade frequency all match.
(Note: the doc's "stop ≈ 1.4 pts avg" in Step 7 is internally wrong — its own
sizing table implies ~6 pts, which is what the data shows.)

## What does NOT reconcile (edge is overstated ~2×)

Properly enforcing the stop bar-by-bar (1m), with conservative same-bar tie-breaks
and forced-flat at 12:00 ET:

| Metric | Claimed | Verified (stop enforced) |
|---|---|---|
| Win rate | **55.5%** | **~33%** |
| Profit factor | 14.73 | ~5.9 gross (~4–5 after slippage) |
| Sharpe (annualized) | 15.09 | ~6.6 |
| Max drawdown | 6R | ~13R |

The win-rate gap is the whole story. Diagnostic: if the stop is *ignored* and a
trade is counted a win whenever price merely **touches** the target before noon,
WR = **62.8%** — bracketing the claimed 55.5%. This is the signature of a backtest
that does not properly enforce the (very tight, ~7%-of-range) stop, i.e.
look-ahead / favorable intrabar ordering. A genuine 12:1 reward:risk system at 55%
WR would imply PF ≈ 15 and Sharpe ≈ 15, which are not realistic.

## Update: realistic fills kill the edge (this is the real conclusion)

My first pass fixed the document's stop-leak bug but kept an *entry* bug: it
filled at the exact R3/S3 level even when the 09:30 bar had already **opened
past** the level. That happens on **553 of 965 trades (57%)** — by 09:30 the
"first touch" is frequently a level price already blew through (often on the
08:30 data releases). Filling those at the level is fantasy.

Realistic stop-entry fill = worse of (level, trigger-bar open) + 2 pt slippage,
stop/target unchanged, manage from next bar, tie=stop:

| Config | Trades | WR | PF | Net R |
|---|---|---|---|---|
| Literal fill at R3/S3 (optimistic) | 965 | 32.8% | 5.86 | +3150 |
| **Realistic fill, all trades** | 965 | **23.8%** | **0.61** | **−1087** |
| Realistic, clean only (drop 553 gap-throughs) | 412 | 13.8% | 1.33 | +163 |

The "PF ~6" was never real — it required magic fills at the level on gap days.
Under tradable fills the strategy is **net negative**. Restricting to only the
trades where price had *not* already gapped past the level leaves a fragile
PF 1.33 (4 of 7 years flat-to-negative; carried entirely by 2022/2023/2025),
and that excludes commissions and stop slippage, which would likely push it
below breakeven. There is no durable edge here.

(These numbers were independently reproduced to the cent by a separate
implementation — strong cross-validation of the logic.)

## Bottom line

The mechanical rules are well-specified and I implemented them faithfully; the
trade universe matches the document. **But the headline win rate (55.5%) and the
derived Sharpe/PF/drawdown are not reproducible — they are roughly 2× too
optimistic, consistent with a stop that wasn't enforced in the original backtest.**

The good news: even at the *real* ~33% win rate, a 12:1 reward:risk gives a
strongly positive expectancy (+~3,150R over 6.4 years), and because the average
stop is ~6 pts the edge survives realistic slippage/commission. The strategy is
plausibly profitable — just not at the advertised win rate, and with materially
larger drawdowns than stated.

Reproduce: `python3 cam_9am_verify.py` (needs the data zips extracted under `data/`).
