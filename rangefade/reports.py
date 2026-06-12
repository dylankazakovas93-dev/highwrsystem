"""Report generation: trade CSVs, markdown summary tables, charts, prop-sim reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import regime_buckets, slippage_table, summarize, summarize_by


def _md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.4g}" if np.isfinite(v) else "-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _try_charts(out: Path, tdf: pd.DataFrame) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    eq = tdf["pnl_usd"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tdf["entry_time"], eq)
    ax.set_title("Equity curve ($ per contract, net)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(charts / "equity.png", dpi=110); plt.close(fig)
    made.append("equity.png")

    dd = eq - eq.cummax()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(tdf["entry_time"], dd, 0, color="tab:red", alpha=0.5)
    ax.set_title("Drawdown ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(charts / "drawdown.png", dpi=110); plt.close(fig)
    made.append("drawdown.png")

    by_m = tdf.groupby("month")["win"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_m.index, by_m["mean"] * 100)
    ax.axhline(80, color="k", ls="--", lw=1, label="80% goal")
    ax.set_ylabel("win rate %"); ax.set_title("Win rate by month (bar count varies!)")
    ax.tick_params(axis="x", rotation=60); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(charts / "winrate_by_month.png", dpi=110); plt.close(fig)
    made.append("winrate_by_month.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tdf["pnl_usd"], bins=40)
    ax.set_title("Per-trade PnL distribution ($)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(charts / "pnl_hist.png", dpi=110); plt.close(fig)
    made.append("pnl_hist.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(tdf["win"], "tab:green", "tab:red")
    ax.scatter(tdf["mae_points"], tdf["mfe_points"], c=colors, s=14, alpha=0.6)
    ax.set_xlabel("MAE (points)"); ax.set_ylabel("MFE (points)")
    ax.set_title("MAE vs MFE per trade"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(charts / "mae_mfe.png", dpi=110); plt.close(fig)
    made.append("mae_mfe.png")
    return made


def write_backtest_report(
    out_dir: str | Path,
    tdf_by_slip: dict[int, pd.DataFrame],
    default_slip: int,
    skips_df: pd.DataFrame,
    meta: dict,
    charts: bool = True,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tdf = tdf_by_slip.get(default_slip, next(iter(tdf_by_slip.values())))

    if not tdf.empty:
        tdf.to_csv(out / "trades.csv", index=False)
    if skips_df is not None and not skips_df.empty:
        skips_df.to_csv(out / "skips.csv", index=False)

    overall = summarize(tdf)
    payload = {"meta": meta, "overall": overall}
    sections: list[tuple[str, pd.DataFrame]] = []
    if not tdf.empty:
        sections = [
            ("Slippage sensitivity", slippage_table(tdf_by_slip)),
            ("By year", summarize_by(tdf, "year")),
            ("By month", summarize_by(tdf, "month")),
            ("By session", summarize_by(tdf, "session")),
            ("By hour of day (entry, ET)", summarize_by(tdf, "hour")),
            ("By volatility regime (prior-day ATR terciles)", regime_buckets(tdf)),
            ("By side", summarize_by(tdf, "side")),
            ("By exit reason", summarize_by(tdf, "exit_reason")),
        ]
        for name, seg in sections:
            payload[name] = seg.to_dict(orient="records") if not seg.empty else []

    with open(out / "summary.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    lines = ["# Backtest summary", ""]
    lines.append(f"_{meta.get('label', '')}_  ")
    for k, v in meta.items():
        if k != "label":
            lines.append(f"- **{k}**: {v}")
    lines += ["", "## Overall (default slippage scenario)", ""]
    lines.append(_md_table(pd.DataFrame([overall])))
    if not tdf.empty:
        skip_counts = (
            skips_df["reasons"].explode().value_counts().rename_axis("skip_reason").reset_index(name="count")
            if skips_df is not None and not skips_df.empty else pd.DataFrame()
        )
        for name, seg in sections:
            lines += ["", f"## {name}", "", _md_table(seg)]
        lines += ["", "## Skip reasons (filtered triggers)", "", _md_table(skip_counts)]
        if charts:
            for png in _try_charts(out, tdf):
                lines.append(f"\n![{png}](charts/{png})")
    (out / "summary.md").write_text("\n".join(lines))
    return out


def write_prop_report(out_dir: str | Path, headline: dict, sens: dict[str, pd.DataFrame],
                      rescue: pd.DataFrame | None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Prop-firm game simulation", "", "## Headline (configured rules)", ""]
    lines.append(_md_table(pd.DataFrame([headline])))
    for name, df in sens.items():
        lines += ["", f"## Sensitivity: {name}", "", _md_table(df)]
        df.to_csv(out / f"prop_sens_{name}.csv", index=False)
    if rescue is not None and not rescue.empty:
        lines += ["", "## Rescue mode: policy comparison by drawdown state", "",
                  "_Account state gates WHEN valid setups are taken; it never creates signals._", "",
                  _md_table(rescue)]
        rescue.to_csv(out / "prop_rescue.csv", index=False)
    with open(out / "prop_summary.json", "w") as fh:
        json.dump({"headline": headline}, fh, indent=2, default=str)
    (out / "prop_report.md").write_text("\n".join(lines))
    return out
