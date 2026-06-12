# highwrsystem — range-rotation fade research framework (NQ/ES)

Intraday backtesting + prop-firm game simulation for a high-win-rate consolidation
fade strategy. Read **THESIS.md** first: research question, mechanism, math,
pre-registered protocol, and rejection rules.

## Layout

```
rangefade/            the framework
  data.py             flexible OHLCV loading, tz handling, trading-day roll, QC
  sessions.py         Asia / London / NY sub-sessions (overnight wrap aware)
  features.py         VWAP+slope, prior-day ATR, realized vol, day move, extremes
  volume_profile.py   POC / VAH / VAL over arbitrary windows
  ranges.py           consolidation boxes: rolling | adaptive | session methods
  strategy.py         probe -> failure -> reclaim state machine, targets/stops
  engine.py           conservative fills (trade-through TP, both-touch = loss)
  metrics.py          WR/CI, PF, expectancy, streaks, segment breakdowns
  grid.py             OFAT/random/full sweeps + pre-registered rejection rules
  prop_sim.py         drawdown buffer / daily loss / consistency / payout MC + rescue mode
  monte_carlo.py      day-level resampling
  reports.py          trades.csv, skips.csv, summary.md/json, charts
  synthetic.py        synthetic data for pipeline validation only
config/
  nq_default.yaml     PRIMARY registered hypothesis (NQ)
  es_default.yaml     instrument-adjusted ES variant
  grid_nq.yaml        sweep axes + dev/holdout windows + rejection thresholds
scripts/
  build_continuous.py Databento GLBX parent-symbology -> continuous front-month parquet
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                  # 35 tests pin engine semantics

# 1) build the continuous contract from Databento raw files in data/raw/
python scripts/build_continuous.py         # -> data/NQ_continuous.parquet + NQ_rolls.csv

# 2) sanity-check the data
python -m rangefade validate-data data/NQ_continuous.parquet

# 3) primary backtest on the dev window, full slippage grid
python -m rangefade backtest --config config/nq_default.yaml \
    --data data/NQ_continuous.parquet --start 2025-01-01 --end 2026-06-07 \
    --out out/nq_base_dev --slippage-grid

# 4) parameter sweep with rejection rules (full history; dev/holdout split inside)
python -m rangefade grid --config config/nq_default.yaml --grid config/grid_nq.yaml \
    --data data/NQ_continuous.parquet --out out/grid_ofat --workers 4

# 5) prop-firm Monte Carlo + rescue-mode comparison from any trades.csv
python -m rangefade propsim --config config/nq_default.yaml \
    --trades out/nq_base_dev/trades.csv --out out/prop
```

Outputs land in `out/<run>/`: `summary.md` (tables + charts), `summary.json`,
`trades.csv` (every trade with full context snapshot), `skips.csv` (every filtered
trigger with reasons), `grid_results.csv` / `grid_report.md`, `prop_report.md`.

## Data expectations

1-minute OHLCV, full Globex session, see `data/README.md`. The provided Databento
NQ.FUT batch files are converted by `scripts/build_continuous.py` (volume-based roll
at the 18:00 ET day boundary, unadjusted prices — roll days get blocked by the
day-move filter instead of rewriting history).

## Non-negotiables baked into the engine

- target must remain INSIDE the consolidation or the trade is skipped
- one bracket per attempt; no averaging, no martingale, no stop widening
- conservative fills: next-bar-open market entries, trade-through limit fills,
  stop+TP same bar = loss, gapped stops fill at the open
- win rate and its stability rank results; net profit never does
