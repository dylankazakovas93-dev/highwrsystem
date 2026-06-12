"""Command-line interface.

    python -m rangefade validate-data data/NQ_1min.csv [--tz UTC]
    python -m rangefade backtest --config config/nq_default.yaml --data data/NQ_1min.csv
    python -m rangefade grid --config config/nq_default.yaml --grid config/grid_nq.yaml --data ...
    python -m rangefade propsim --config config/nq_default.yaml --trades out/backtest/trades.csv
    python -m rangefade selftest          # synthetic end-to-end pipeline check (no real data needed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import add_trading_day, data_quality_report, filter_period, load_ohlcv
from .grid import load_grid_spec, run_grid
from .metrics import trade_frame
from .prop_sim import run_prop_mc, run_rescue_mc, sensitivity_tables
from .reports import write_backtest_report, write_prop_report
from .strategy import prepare, run_backtest


def _load(args, cfg):
    if args.tz:
        cfg.data.tz_input = args.tz
    df = load_ohlcv(args.data, cfg.data)
    df = filter_period(add_trading_day(df, cfg.sessions.day_roll_hour), args.start, args.end)
    return df.drop(columns=["trading_day"])  # prepare() re-adds it


def cmd_validate(args) -> int:
    cfg = load_config(args.config) if args.config else load_config()
    if args.tz:
        cfg.data.tz_input = args.tz
    df = load_ohlcv(args.data, cfg.data)
    df = add_trading_day(df, cfg.sessions.day_roll_hour)
    rep = data_quality_report(df, cfg.data.bar_minutes)
    print(json.dumps(rep, indent=2))
    return 0


def cmd_backtest(args) -> int:
    cfg = load_config(args.config)
    df_raw = _load(args, cfg)
    dfp = prepare(df_raw, cfg)
    scenarios = cfg.engine.slippage_scenarios if args.slippage_grid else [cfg.instrument.slippage_ticks]
    tdf_by_slip = {}
    skips_df = pd.DataFrame()
    for ticks in scenarios:
        res = run_backtest(dfp, cfg, slippage_ticks=ticks)
        tdf_by_slip[ticks] = trade_frame(res.trades)
        if ticks == cfg.instrument.slippage_ticks:
            skips_df = pd.DataFrame(
                [{"ts": s.ts, "side": s.side, "reasons": s.reasons} for s in res.skips]
            )
        print(f"[backtest] slip={ticks}t: {len(res.trades)} trades, "
              f"{res.n_valid_box_bars} valid-box bars over {res.n_days} days")
    out = write_backtest_report(
        args.out, tdf_by_slip, cfg.instrument.slippage_ticks, skips_df,
        meta={
            "label": args.label or f"{cfg.instrument.symbol} {cfg.range.method} "
                                   f"w{cfg.range.window_minutes} tp{cfg.exit.tp_points}",
            "config": args.config, "data": args.data,
            "period": f"{args.start or 'all'} .. {args.end or 'all'}",
            "instrument": cfg.instrument.symbol,
        },
        charts=cfg.report.charts and not args.no_charts,
    )
    print(f"[backtest] report -> {out}/summary.md")
    return 0


def cmd_grid(args) -> int:
    cfg = load_config(args.config)
    spec = load_grid_spec(args.grid)
    df_raw = _load(args, cfg)
    run_grid(df_raw, cfg, spec, args.out, workers=args.workers)
    return 0


def cmd_propsim(args) -> int:
    cfg = load_config(args.config)
    tdf = pd.read_csv(args.trades, parse_dates=["entry_time", "exit_time"])
    n_days_total = args.calendar_days or tdf["trading_day"].nunique()
    headline = run_prop_mc(tdf, cfg.prop)
    sens = sensitivity_tables(tdf, cfg.prop)
    rescue = run_rescue_mc(tdf, cfg.prop, cfg.rescue, n_days_total) if cfg.rescue.enabled else None
    out = write_prop_report(args.out, headline, sens, rescue)
    print(json.dumps(headline, indent=2, default=str))
    print(f"[propsim] report -> {out}/prop_report.md")
    return 0


def cmd_selftest(args) -> int:
    """End-to-end pipeline check on synthetic data. Numbers are MEANINGLESS for live
    trading (synthetic balance segments mean-revert by construction)."""
    from .synthetic import make_synthetic

    print("[selftest] generating synthetic NQ-like data "
          f"({args.days} days, seed {args.seed}) — ENGINE VALIDATION ONLY")
    df_raw = make_synthetic(n_days=args.days, seed=args.seed)
    cfg = load_config(args.config) if args.config else load_config()
    cfg.report.charts = not args.no_charts
    dfp = prepare(df_raw, cfg)
    tdf_by_slip = {}
    skips_df = pd.DataFrame()
    for ticks in cfg.engine.slippage_scenarios:
        res = run_backtest(dfp, cfg, slippage_ticks=ticks)
        tdf_by_slip[ticks] = trade_frame(res.trades)
        if ticks == cfg.instrument.slippage_ticks:
            skips_df = pd.DataFrame(
                [{"ts": s.ts, "side": s.side, "reasons": s.reasons} for s in res.skips]
            )
        print(f"[selftest] slip={ticks}t: trades={len(res.trades)}")
    out = write_backtest_report(
        args.out, tdf_by_slip, cfg.instrument.slippage_ticks, skips_df,
        meta={"label": "SYNTHETIC SELFTEST — pipeline validation, not an edge claim",
              "instrument": cfg.instrument.symbol},
        charts=cfg.report.charts,
    )
    tdf = tdf_by_slip[cfg.instrument.slippage_ticks]
    if not tdf.empty:
        headline = run_prop_mc(tdf, cfg.prop, n_paths=1000)
        sens = sensitivity_tables(tdf, cfg.prop)
        rescue = run_rescue_mc(tdf, cfg.prop, cfg.rescue, tdf["trading_day"].nunique(), n_paths=500)
        write_prop_report(args.out, headline, sens, rescue)
        print(f"[selftest] prop sim p_payout={headline.get('p_payout')}")
    print(f"[selftest] OK -> {out}/summary.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rangefade")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/nq_default.yaml")
    common.add_argument("--data", help="path to 1-min OHLCV csv/parquet")
    common.add_argument("--tz", help="tz of naive input timestamps (e.g. UTC)")
    common.add_argument("--start", help="trading-day start filter, YYYY-MM-DD")
    common.add_argument("--end", help="trading-day end filter, YYYY-MM-DD")

    v = sub.add_parser("validate-data", help="data quality report")
    v.add_argument("data")
    v.add_argument("--config")
    v.add_argument("--tz")
    v.set_defaults(fn=cmd_validate)

    b = sub.add_parser("backtest", parents=[common], help="single-config backtest + report")
    b.add_argument("--out", default="out/backtest")
    b.add_argument("--label")
    b.add_argument("--slippage-grid", action="store_true", help="run all slippage scenarios")
    b.add_argument("--no-charts", action="store_true")
    b.set_defaults(fn=cmd_backtest)

    g = sub.add_parser("grid", parents=[common], help="parameter sweep with rejection rules")
    g.add_argument("--grid", default="config/grid_nq.yaml")
    g.add_argument("--out", default="out/grid")
    g.add_argument("--workers", type=int, default=1)
    g.set_defaults(fn=cmd_grid)

    pr = sub.add_parser("propsim", help="prop-firm Monte Carlo from a trades.csv")
    pr.add_argument("--config", default="config/nq_default.yaml")
    pr.add_argument("--trades", required=True)
    pr.add_argument("--out", default="out/prop")
    pr.add_argument("--calendar-days", type=int,
                    help="total trading days in the backtest period (for setup frequency)")
    pr.set_defaults(fn=cmd_propsim)

    st = sub.add_parser("selftest", help="synthetic end-to-end pipeline validation")
    st.add_argument("--config")
    st.add_argument("--days", type=int, default=100)
    st.add_argument("--seed", type=int, default=42)
    st.add_argument("--out", default="out/selftest")
    st.add_argument("--no-charts", action="store_true")
    st.set_defaults(fn=cmd_selftest)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
