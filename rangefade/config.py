"""Configuration schema and YAML loading.

Every research knob lives here so parameter sweeps are pure config overrides.
Unknown YAML keys produce warnings instead of silent acceptance.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

import yaml


@dataclass
class InstrumentCfg:
    symbol: str = "NQ"
    tick_size: float = 0.25
    point_value: float = 20.0          # $ per full point per contract
    commission_per_side: float = 2.50  # $ per contract per side, all-in
    slippage_ticks: int = 1            # default scenario, per side, market/stop fills


@dataclass
class DataCfg:
    tz_input: Optional[str] = None        # tz of naive input timestamps ("UTC", ...). None = already market tz
    tz_market: str = "America/New_York"   # all session logic runs in this tz
    bar_minutes: int = 1
    timestamp_format: Optional[str] = None


@dataclass
class SessionDef:
    name: str
    start: str            # "HH:MM" market tz; start > end means the session wraps midnight
    end: str
    tradable: bool = True


def _default_sessions() -> list[SessionDef]:
    return [
        SessionDef("asia", "18:00", "02:00"),
        SessionDef("london_early", "02:00", "05:00"),
        SessionDef("pre_ny", "05:00", "09:30", tradable=False),
        SessionDef("ny_morning", "09:30", "12:00"),
        SessionDef("ny_lunch_pm", "12:00", "15:00"),
        SessionDef("ny_last_hour", "15:00", "16:00", tradable=False),  # default: no last-hour trading
        SessionDef("ny_close", "16:00", "17:00", tradable=False),
    ]


@dataclass
class SessionsCfg:
    definitions: list[SessionDef] = field(default_factory=_default_sessions)
    trade: list[str] = field(default_factory=lambda: ["asia", "london_early", "ny_morning", "ny_lunch_pm"])
    day_roll_hour: int = 18           # CME trading day starts 18:00 ET
    flat_at_session_end: bool = True  # force-flat when the sub-session ends


@dataclass
class RangeCfg:
    method: str = "rolling"            # rolling | adaptive | session
    window_minutes: int = 90
    min_width_points: float = 40.0
    max_width_points: float = 100.0
    min_window_coverage: float = 0.8   # fraction of expected bars that must exist in the window
    # method == "session": box frozen from a formation sub-session, traded later in the day
    formation_session: str = "asia"
    # adaptive extras (structure baked into validity, not just trigger-time filters)
    extremes_min_age_minutes: int = 30   # no new window high/low within the last N minutes
    max_net_drift_frac: float = 0.5      # |close_end - close_start| / width over the window
    # volume profile
    profile_bin_points: float = 1.0
    value_area: float = 0.70


@dataclass
class FiltersCfg:
    # current-day move filter (trend/macro transmission guard)
    max_day_move_pct: Optional[float] = 0.5      # percent, e.g. 0.5 => 0.5%
    day_move_refs: list[str] = field(default_factory=lambda: ["prior_close", "rth_open"])
    # prior-day ATR containment
    max_range_vs_prior_atr: Optional[float] = 0.5
    prior_atr_days: int = 14
    # VWAP slope (flatness requirement)
    max_vwap_slope_points: Optional[float] = 10.0  # |vwap_now - vwap_then| over the slope window
    vwap_slope_window_minutes: int = 30
    vwap_anchor: str = "auto"                      # auto: RTH anchor inside RTH, day anchor overnight
    # two-way auction acceptance (proof of rotation)
    min_mid_crossings: int = 3
    # fresh strong session extreme guard
    extreme_extension_points: Optional[float] = 15.0
    extreme_extension_window_minutes: int = 30
    # volatility expansion guard
    rv_window_minutes: int = 30
    rv_max_vs_median: Optional[float] = 2.0        # rv vs its own 20-day rolling median
    rv_median_days: int = 20
    # news exclusion
    news_file: Optional[str] = None
    news_tz: str = "America/New_York"
    news_block_before_min: int = 15
    news_block_after_min: int = 15
    news_min_impact: str = "high"                  # low < medium < high


@dataclass
class EntryCfg:
    type: str = "A"               # A market-on-reclaim-close | B limit-on-retest | C micro-swing-break stop
    sides: list[str] = field(default_factory=lambda: ["long", "short"])
    edge_def: str = "pct"         # absolute | pct | vah_val | hybrid
    edge_pct: float = 0.10        # zone depth as a fraction of range width (pct/hybrid)
    max_overshoot_points: float = 10.0   # probe beyond box edge above this => breakout, cancel setup
    min_overshoot_points: float = 0.0    # probe must exceed the edge by at least this (real failed break)
    probe_ttl_bars: int = 30      # reclaim must occur within this many bars of probe start
    reclaim_bars: int = 1         # consecutive closes back inside required
    retest_ttl_bars: int = 15     # entry B: limit order lifetime
    retest_level: str = "edge"    # edge (zone boundary) | halfway (between reclaim close and box edge)
    confirm_swing_bars: int = 3   # entry C: micro swing lookback
    confirm_ttl_bars: int = 10    # entry C: stop-order lifetime
    one_attempt_per_side: bool = True    # per consolidation episode
    max_trades_per_day: int = 1
    stop_after_daily_loss: bool = True
    min_minutes_to_flat: int = 15        # no entry if less than this remains before forced flat


@dataclass
class ExitCfg:
    tp_mode: str = "fixed"        # fixed | mid | poc | opposite_25
    tp_points: float = 15.0
    min_tp_points: float = 15.0   # floor for dynamic tp modes
    require_target_inside: bool = True
    target_inside_buffer_points: float = 0.0   # target must clear opposite edge by this much
    stop_model: str = "edge_plus"              # edge_plus | excursion_plus | range_frac
    stop_buffer_points: float = 7.5
    stop_range_frac: float = 0.35
    max_stop_points: float = 25.0
    stop_cap_mode: str = "clamp"               # clamp | skip
    min_stop_points: float = 3.0
    invalidate_close_outside: bool = False     # extra exit: bar closes outside box beyond buffer
    invalidation_buffer_points: float = 5.0
    max_holding_minutes: int = 120
    flat_by: Optional[str] = "15:00"           # hard flat time (market tz); keeps us out of the last hour


@dataclass
class EngineCfg:
    both_touch: str = "loss"      # bar touches both stop & target -> count as loss (conservative)
    tp_fill_through_ticks: int = 1     # price must trade through the tp limit by this many ticks
    limit_fill_through_ticks: int = 1  # same for entry limit orders
    slippage_scenarios: list[int] = field(default_factory=lambda: [0, 1, 2, 3])


@dataclass
class PropCfg:
    buffer: float = 2000.0
    dd_type: str = "trailing_eod"     # static | trailing_eod | trailing_intraday
    trailing_caps_at_start: bool = True
    daily_loss_limit: Optional[float] = 1000.0
    activation_cost: float = 165.0
    payout_threshold: float = 3000.0
    profit_split: float = 0.90
    min_trading_days: int = 5
    consistency_max_day_frac: Optional[float] = 0.5   # best day <= frac * total profit at payout
    contracts: int = 1
    max_trading_days: int = 120       # give up after this many traded days
    n_paths: int = 5000
    resample: str = "shuffle_days"    # shuffle_days | bootstrap_days
    seed: int = 7
    max_attempts: int = 3             # sequential account restarts for EV-of-the-game
    sens_costs: list[float] = field(default_factory=lambda: [100, 165, 300, 500])
    sens_payouts: list[float] = field(default_factory=lambda: [1500, 2000, 3000, 4000])
    sens_contracts: list[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass
class RescueCfg:
    enabled: bool = True
    dd_states: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.8])  # fraction of buffer consumed
    # synthetic "original strategy" the account got into trouble with
    baseline_wr: float = 0.45
    baseline_avg_win: float = 300.0
    baseline_avg_loss: float = 350.0
    baseline_trades_per_day: int = 3


@dataclass
class ReportCfg:
    out_dir: str = "out"
    charts: bool = True


@dataclass
class Config:
    instrument: InstrumentCfg = field(default_factory=InstrumentCfg)
    data: DataCfg = field(default_factory=DataCfg)
    sessions: SessionsCfg = field(default_factory=SessionsCfg)
    range: RangeCfg = field(default_factory=RangeCfg)
    filters: FiltersCfg = field(default_factory=FiltersCfg)
    entry: EntryCfg = field(default_factory=EntryCfg)
    exit: ExitCfg = field(default_factory=ExitCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    prop: PropCfg = field(default_factory=PropCfg)
    rescue: RescueCfg = field(default_factory=RescueCfg)
    report: ReportCfg = field(default_factory=ReportCfg)


def _merge_into(obj: Any, d: dict, path: str = "") -> None:
    valid = {f.name: f for f in fields(obj)}
    for key, val in d.items():
        if key not in valid:
            print(f"[config] warning: unknown key '{path}{key}' ignored", file=sys.stderr)
            continue
        cur = getattr(obj, key)
        if key == "definitions" and isinstance(val, list):  # session definitions replace wholesale
            setattr(obj, key, [SessionDef(**s) for s in val])
        elif is_dataclass(cur) and isinstance(val, dict):
            _merge_into(cur, val, path=f"{path}{key}.")
        else:
            setattr(obj, key, val)


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """Build a Config from defaults, an optional YAML file, and dotted-key overrides."""
    cfg = Config()
    if path:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        _merge_into(cfg, raw)
    if overrides:
        apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Apply {'exit.tp_points': 20, ...} style overrides in place (used by the grid runner)."""
    for dotted, val in overrides.items():
        parts = dotted.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"unknown config key: {dotted}")
        setattr(obj, leaf, copy.deepcopy(val))
    return cfg


def clone_config(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def validate_config(cfg: Config) -> None:
    r, x, e = cfg.range, cfg.exit, cfg.entry
    problems = []
    if r.min_width_points >= r.max_width_points:
        problems.append("range.min_width_points must be < max_width_points")
    if r.method not in ("rolling", "adaptive", "session"):
        problems.append(f"range.method '{r.method}' invalid")
    if e.type not in ("A", "B", "C"):
        problems.append(f"entry.type '{e.type}' invalid")
    if e.edge_def not in ("absolute", "pct", "vah_val", "hybrid"):
        problems.append(f"entry.edge_def '{e.edge_def}' invalid")
    if x.tp_mode not in ("fixed", "mid", "poc", "opposite_25"):
        problems.append(f"exit.tp_mode '{x.tp_mode}' invalid")
    if x.stop_model not in ("edge_plus", "excursion_plus", "range_frac"):
        problems.append(f"exit.stop_model '{x.stop_model}' invalid")
    if x.tp_mode == "fixed" and x.tp_points > r.max_width_points:
        problems.append("exit.tp_points exceeds range.max_width_points — every trade would be skipped")
    names = [s.name for s in cfg.sessions.definitions]
    for t in cfg.sessions.trade:
        if t not in names:
            problems.append(f"sessions.trade '{t}' not among definitions {names}")
    if r.method == "session" and r.formation_session not in names:
        problems.append(f"range.formation_session '{r.formation_session}' not among definitions")
    if problems:
        raise ValueError("config invalid:\n  - " + "\n  - ".join(problems))
