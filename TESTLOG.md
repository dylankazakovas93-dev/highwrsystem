# Test log — every config run, and the honest verdict

This is the complete, faithful inventory of what was tested. Where a number is a
"recent (2024–26)" figure, treat it as **in-sample / fitted** — it was selected by
measuring the recent regime, so it overstates what you'd get out-of-sample.

## Honesty note (what was curve-fit)
The base **open-drive continuation** edge (85% WR @ RR 0.2, all 7 years) is real and was
independently reproduced by Codex. But the **eval-oriented tuning layered on top** —
choosing gate 2.5/3.0, the 60-min ATR basis, TP "calibrated to 11pt in 2026", and quoting
"recent-regime WR" — was selection on 2024–26 performance. Those 70%+ figures are fitted,
not validated. The only legitimately out-of-sample claim is the all-years base edge.

---

## Family 1 — Range-rotation FADE (the `rangefade/` framework)
Mechanism: fade failed pushes at the edge of an intraday consolidation; target inside box.

| run | axes swept | verdict |
|---|---|---|
| `nq_default` baseline (dev 25–26) | the registered config | 47.3% WR, −$38/trade — no edge |
| `grid_nq.yaml` OFAT (73 combos) | window[45-180], method[rolling/adaptive/session], min_width[30-100], day_move[.3-.75], range_vs_atr[.15-.5], vwap_slope[5-20], mid_crossings[2-6], extreme_ext[10/15/20/null], entry.type[A/B/C], edge_def[abs/pct/vah_val/hybrid], edge_pct[.10/.15/.20], overshoot[5/10/15], tp[10-25], tp_mode[fixed/mid/poc/opp25], stop_model[edge/excursion/range_frac], stop_buf[5/7.5/10], invalidate[T/F], sessions[asia/london/ny_morning/ny_lunch/all] | **0 survivors** of the rejection rules |
| `grid_nq_stage2_fast` (random ~90, 2023-26) | narrowed region of the above | nothing robust |
| `nq_candidate_lunch_shorts` | lunch session, shorts only | dev 64.8% WR → **holdout 53.3%** (overfit, failed OOS) |
| `es_default`/`grid_es` | ES-scaled variants | **never run — no ES data provided** |

**Conclusion: the fade has no real edge.**

---

## Family 2 — Open-drive CONTINUATION (`scripts/open_drive_*.py`)
Mechanism: at a snapshot time, if move from 09:30 open ≥ gate×ATR, trade WITH the move;
ATR-scaled TP, structural stop, time exit.

| script | axes swept | verdict (honest) |
|---|---|---|
| `open_drive_probe` | snapshot[11/12], TP[1x/2x ATR30], RR[0.1–0.8], dir[cont/fade] | cont +EV, fade −EV every yr (real directional effect); WR rises as RR falls |
| `open_drive_stats` | gate[1.0–2.5], snapshot[11/12], TP[0.5/0.75/1.0/1.5×ATR], slippage[0–3t], per-yr TP/SL | smaller TP→higher WR; gate→higher WR; 11:00>12:00 |
| `open_drive_final` | snapshot[10/11/12/13], gate 2.0, TP0.5×ATR, stop-cap 200 | 11:00 best (85.2%), monotonic by snapshot |
| `open_drive_gate_perturb` | gate-ATR tf[15/30/60m] × gate mult[1.5–3.0] | 30m gate flat/robust (good); 60m a fragile peak |
| `open_drive_rr_perturb` | RR[0.1–0.8] y/y | WR plateau 84–85% at RR 0.1–0.2; smooth decline after |
| `open_drive_rr02` | RR 0.2, gate[1.0–2.5] | flat-WR plateau gate≥1.5 |
| `open_drive_tp_time` | TP basis[0.25/0.5 ATR30, 0.4/0.5 ATR15] × time-exit[30–120m/EOD] | ATR15 TP + 90-min exit → PF 1.28→1.35 |
| `open_drive_card` | locked card (gate1.5, TP0.5×ATR30, RR0.2) | 84.1% WR, PF 1.28 |
| `open_drive_be` | breakeven-stop trigger[0.5–2.0×TP] | **hurts** — skip |
| `open_drive_timecut` | underwater cut[10–60min] × [with/without structural stop] | **hurts** net/WR — skip |
| `keeper_config` | locked combined (gate1.5/2.0, TP0.4×ATR15, RR0.2, 90m) | 84.9% WR, PF 1.35 — Codex-verified 85.3% |

**Conclusion: base continuation edge is real (≈85% @ RR 0.2, all years). It is NOT convex
and does not survive at RR ≥ 0.75 in any robust way.**

---

## Family 3 — Eval-passer searches (`scripts/eval_*.py`) — this is where fitting crept in
| script | axes swept | verdict |
|---|---|---|
| `eval_passer_search` | gate[1/1.5/2] × TP[0.4/0.6 ATR15, 0.5/0.7 ATR30] × RR[0.2–3.0], ranked by P(pass) MC | best eval-passer RR 0.8, gate 2.0, ~63% WR, P(pass) ~67% (on +3000/−2000) |
| `eval_rr075` | RR 0.75, TP cal 11pt-2026, ATR tf[1/5/15/30/60m] × gate[1/1.5/2] | best **recent** WR ~67.5% (fitted) |
| `eval_can_we_hit_70` | snapshot[10/11] × gate[2/2.5/3/3.5] × tf[15/60m] × RR[0.75/1.0] | only 2/32 touch 70% recent (fitted, wide CI) |
| `monthly_eval_mc` | contracts[3/5/7/10], monthly window | P(pass) tops ~49% |

**Conclusion: 70%+ WR at RR ≥ 0.75 is not robustly achievable; monthly pass ~49%.**

---

## Family 4 — Other mechanisms
| script | axes swept | verdict |
|---|---|---|
| `gap_fill_test` | min_gap[8–30pt] × RR[0.75/1.0/1.5], fade vs continuation | **no edge** — coin flip once stopped (55%@RR0.75, below 57% breakeven) |
| `overnight_drift_test` | long/short, time-exit vs bracket, pullback[0/0.3×ATR], tf[15/30/60m], RR[0.75/1.0] | drift real: long +$106/time-exit, short −$136; **bracketed = negative**; 54.6% WR |
| `first_candle_overnight` | first candle tf[15/30/60/240m] × thr[0.3/0.5/0.75/1.0] × [long-only/follow] × [time/bracket RR0.75] | gate helps; bracketed RR0.75 ~73% **recent** (fitted, low freq ~25/yr, all-yrs ~60%) |

---

## What is ACTUALLY real vs fitted (one-line summary)
- **Real, out-of-sample:** open-drive continuation has a directional edge (~85% WR at the
  wide-stop RR 0.2; ~55–60% all-years at RR 0.75). Overnight session has a long-drift edge
  (~54% WR, +EV by holding). Fade and gap-fill have **no** edge.
- **Fitted (don't trust the headline):** every "recent-regime 67–73% WR at RR ≥ 0.75."

## What can still be tuned (and the honest cost of doing so)
Untried knobs: ATR period (always 14); gate on range vs close-move; gate as %-of-price;
snapshot times outside 10:00–13:00; multiple entries/day; trailing/partial exits; flat
time; filters never tried (day-of-week, prior-day direction, trend filter vs a moving
average, volatility/VIX regime, news, seasonality); ES and other instruments; contract
sizing/pyramiding.

**But:** with ~1,660 trading days and already ~10–15 degrees of freedom exercised, **more
in-sample tuning will manufacture better-looking numbers without making the edge more
real.** The honest next step is the opposite of tuning:
1. **Walk-forward**: optimize only on 2020–2023, then run 2024–2026 **once, untouched**, and
   accept whatever it gives. That is the only number worth trusting for go/no-go.
2. **Out-of-sample instrument**: run the frozen config on **ES** (needs ES 1-min data).
3. **Reduce, don't add**, degrees of freedom: fewer parameters = less overfit.
