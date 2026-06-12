"""Parameter sweep + robustness harness.

Discipline (pre-registered in THESIS.md):
  * the user's default config is the PRIMARY hypothesis; the grid is exploratory
  * optimize on the dev period (2025–2026), treat earlier years as holdout
  * hard rejection rules — not judgement calls applied after seeing results:
      - fewer than `min_trades` dev trades            -> too_few_trades
      - WR < `wr_floor` in any major segment          -> wr_collapse
      - top 2 winners > `top2_share_max` of gross win -> outlier_dependent
      - expectancy at `slip_stress` ticks <= 0        -> dies_with_slippage
  * survivors ranked by (worst-segment WR, dev WR, losing streak, stressed expectancy);
    net profit is never a ranking key.

Sweep modes: ofat (one-factor-at-a-time around the base config), random (seeded random
combos from the axes), full (cartesian product, capped).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .config import Config, apply_overrides, clone_config, load_config
from .metrics import summarize, trade_frame
from .reports import _md_table
from .strategy import prepare, prepare_signature, run_backtest

# Worker state inherited via fork (Linux). Avoids pickling the dataframe per task.
_G: dict = {}


@dataclass
class GridSpec:
    axes: dict[str, list] = field(default_factory=dict)
    mode: str = "ofat"                 # ofat | random | full
    random_samples: int = 200
    max_combos: int = 2000
    seed: int = 11
    dev_start: str = "2025-01-01"
    dev_end: str = "2026-12-31"
    holdouts: list[dict] = field(default_factory=lambda: [
        {"name": "2022_2024", "start": "2022-01-01", "end": "2024-12-31"},
        {"name": "pre_2022", "start": "2000-01-01", "end": "2021-12-31"},
    ])
    min_trades: int = 40
    low_sample: int = 80
    wr_floor: float = 0.70
    min_segment_trades: int = 15
    top2_share_max: float = 0.35
    slip_eval: int = 1
    slip_stress: int = 2


def load_grid_spec(path: str) -> GridSpec:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    spec = GridSpec()
    for k, v in raw.items():
        if hasattr(spec, k):
            setattr(spec, k, v)
        else:
            print(f"[grid] warning: unknown grid key '{k}' ignored")
    return spec


def expand_combos(spec: GridSpec) -> list[dict]:
    axes = {k: list(v) for k, v in spec.axes.items()}
    if not axes:
        return [{}]
    combos: list[dict] = [{}]  # always include the base config itself
    if spec.mode == "ofat":
        for key, values in axes.items():
            for v in values:
                combos.append({key: v})
    elif spec.mode == "random":
        rng = np.random.default_rng(spec.seed)
        keys = list(axes)
        seen = set()
        while len(combos) - 1 < spec.random_samples:
            combo = {k: axes[k][rng.integers(0, len(axes[k]))] for k in keys}
            sig = json.dumps(combo, sort_keys=True, default=str)
            if sig not in seen:
                seen.add(sig)
                combos.append(combo)
            if len(seen) >= np.prod([len(v) for v in axes.values()]):
                break
    else:  # full
        keys = list(axes)
        for values in itertools.product(*(axes[k] for k in keys)):
            combos.append(dict(zip(keys, values)))
            if len(combos) > spec.max_combos:
                print(f"[grid] truncated at max_combos={spec.max_combos}")
                break
    # dedupe
    out, seen = [], set()
    for cb in combos:
        sig = json.dumps(cb, sort_keys=True, default=str)
        if sig not in seen:
            seen.add(sig)
            out.append(cb)
    return out


def _segment(tdf: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if tdf.empty:
        return tdf
    m = (tdf["entry_time"] >= pd.Timestamp(start, tz=tdf["entry_time"].dt.tz)) & (
        tdf["entry_time"] <= pd.Timestamp(end, tz=tdf["entry_time"].dt.tz) + pd.Timedelta(days=1)
    )
    return tdf[m]


def _prepared_for(cfg: Config) -> pd.DataFrame:
    """Per-process cache: re-prepare only when a combo changes prepare-relevant params."""
    sig = prepare_signature(cfg)
    cache = _G.setdefault("cache", {})
    if sig not in cache:
        cache[sig] = prepare(_G["raw"], cfg)
    return cache[sig]


def evaluate_combo(overrides: dict) -> dict:
    base_cfg: Config = _G["base"]
    spec: GridSpec = _G["spec"]
    cfg = clone_config(base_cfg)
    apply_overrides(cfg, overrides)
    row: dict = {"overrides": json.dumps(overrides, default=str) if overrides else "BASE"}
    try:
        df_prepared = _prepared_for(cfg)
        res_eval = run_backtest(df_prepared, cfg, slippage_ticks=spec.slip_eval)
        res_stress = run_backtest(df_prepared, cfg, slippage_ticks=spec.slip_stress)
    except Exception as e:  # a bad combo must not kill the sweep
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    tdf = trade_frame(res_eval.trades)
    tdf_stress = trade_frame(res_stress.trades)
    dev = _segment(tdf, spec.dev_start, spec.dev_end)
    dev_stress = _segment(tdf_stress, spec.dev_start, spec.dev_end)

    s_dev = summarize(dev)
    row.update({f"dev_{k}": v for k, v in s_dev.items()})
    row["dev_expectancy_stressed"] = summarize(dev_stress).get("expectancy_usd", float("nan"))
    row["dev_wr_stressed"] = summarize(dev_stress).get("win_rate", float("nan"))

    segment_wrs: list[float] = []
    if not dev.empty:
        for yr, sub in dev.groupby("year"):
            if len(sub) >= spec.min_segment_trades:
                wr = float((sub["pnl_usd"] > 0).mean())
                segment_wrs.append(wr)
                row[f"wr_{yr}"] = round(wr, 4)
    for hold in spec.holdouts:
        seg = _segment(tdf, hold["start"], hold["end"])
        s = summarize(seg)
        row[f"hold_{hold['name']}_trades"] = s.get("trades", 0)
        row[f"hold_{hold['name']}_wr"] = s.get("win_rate", float("nan"))
        row[f"hold_{hold['name']}_expectancy"] = s.get("expectancy_usd", float("nan"))
        if s.get("trades", 0) >= spec.min_segment_trades:
            segment_wrs.append(s["win_rate"])

    flags = []
    n_dev = s_dev.get("trades", 0)
    if n_dev < spec.min_trades:
        flags.append("too_few_trades")
    elif n_dev < spec.low_sample:
        flags.append("low_sample")
    if segment_wrs and min(segment_wrs) < spec.wr_floor:
        flags.append("wr_collapse")
    if n_dev and s_dev.get("top2_win_share", 0) > spec.top2_share_max:
        flags.append("outlier_dependent")
    exp_stress = row.get("dev_expectancy_stressed")
    if n_dev and isinstance(exp_stress, (int, float)) and np.isfinite(exp_stress) and exp_stress <= 0:
        flags.append("dies_with_slippage")
    row["min_segment_wr"] = round(min(segment_wrs), 4) if segment_wrs else float("nan")
    row["flags"] = ",".join(flags)
    row["rejected"] = any(f in ("too_few_trades", "wr_collapse", "outlier_dependent", "dies_with_slippage")
                          for f in flags)
    return row


def run_grid(
    df_raw: pd.DataFrame,
    base_cfg: Config,
    spec: GridSpec,
    out_dir: str | Path,
    workers: int = 1,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combos = expand_combos(spec)
    print(f"[grid] {len(combos)} combos, mode={spec.mode}")
    _G["raw"] = df_raw
    _G["base"] = base_cfg
    _G["spec"] = spec
    _G["cache"] = {prepare_signature(base_cfg): prepare(df_raw, base_cfg)}

    rows: list[dict] = []
    if workers > 1:
        # relies on fork inheriting _G (Linux default); use workers=1 elsewhere
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(evaluate_combo, cb) for cb in combos]
            for i, f in enumerate(futs):
                rows.append(f.result())
                if (i + 1) % 10 == 0:
                    print(f"[grid] {i + 1}/{len(combos)} done")
    else:
        for i, cb in enumerate(combos):
            rows.append(evaluate_combo(cb))
            if (i + 1) % 10 == 0:
                print(f"[grid] {i + 1}/{len(combos)} done")

    res = pd.DataFrame(rows)
    if "rejected" in res.columns:
        survivors = res[~res["rejected"].fillna(True)]
        survivors = survivors.sort_values(
            by=["min_segment_wr", "dev_win_rate", "dev_max_losing_streak", "dev_expectancy_stressed"],
            ascending=[False, False, True, False],
        )
        rejected = res[res["rejected"].fillna(True)]
        res = pd.concat([survivors, rejected])
    res.to_csv(out / "grid_results.csv", index=False)

    lines = ["# Grid sweep results", "",
             f"- combos evaluated: {len(res)}",
             f"- survivors after rejection rules: {int((~res.get('rejected', pd.Series(dtype=bool)).fillna(True)).sum())}",
             f"- dev period: {spec.dev_start} .. {spec.dev_end}; eval slip {spec.slip_eval} tick(s), stress {spec.slip_stress}",
             "", "## Top candidates (survivors, ranked by worst-segment WR then dev WR)", ""]
    show_cols = [c for c in (
        "overrides", "dev_trades", "dev_win_rate", "min_segment_wr", "dev_max_losing_streak",
        "dev_expectancy_usd", "dev_expectancy_stressed", "dev_profit_factor",
        "dev_max_drawdown_usd", "flags") if c in res.columns]
    top = res[~res.get("rejected", pd.Series(True, index=res.index)).fillna(True)]
    lines.append(_md_table(top[show_cols].head(25) if show_cols else top.head(25)))
    lines += ["", "## Rejection counts", ""]
    if "flags" in res.columns:
        counts = res["flags"].str.split(",").explode().replace("", np.nan).dropna().value_counts()
        lines.append(_md_table(counts.rename_axis("flag").reset_index(name="count")))
    (out / "grid_report.md").write_text("\n".join(lines))
    print(f"[grid] wrote {out / 'grid_results.csv'} and grid_report.md")
    return res
