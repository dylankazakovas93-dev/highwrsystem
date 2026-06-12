# Data drop zone

Put your market data here (gitignored — never committed).

## Required format: 1-minute OHLCV

CSV or parquet. Column names are matched case-insensitively; accepted layouts:

```
timestamp,open,high,low,close,volume          # single timestamp column
date,time,open,high,low,close,volume          # split date/time (vendor exports)
```

- `timestamp` may also be named `datetime`, `date_time`, `ts`; epoch seconds/ms also work.
- Timestamps should be the bar OPEN time.
- If timestamps are tz-naive, tell the loader the source tz: `--tz UTC` (or set
  `data.tz_input` in the config). Tz-aware stamps are converted automatically.
- Include the FULL Globex session (overnight), not just RTH — Asia/London sessions and
  the trading-day features need it.
- Prefer a back-adjusted continuous front-month contract with real volume. Note the
  roll method used; unadjusted series put gaps at every roll, which poisons
  prior-close-based filters.

Quick sanity check after dropping a file:

```
python -m rangefade validate-data data/NQ_1min.csv --tz UTC
```

Watch for: missing days, days with <50% bar coverage, zero-volume share, wrong tz
(e.g. RTH appearing at the wrong hours).

## Optional: news calendar

Copy `news_template.csv`, fill with high-impact events (CPI, FOMC, NFP, ...), then set
`filters.news_file: data/news.csv` in the config. Naive times are assumed ET unless
`filters.news_tz` says otherwise.
