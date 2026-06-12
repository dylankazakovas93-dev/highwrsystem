"""Range-rotation fade strategy.

Setup grammar (mirrored for both sides):
  wide accepted consolidation -> probe into/through the edge zone -> failure to continue
  -> reclaim close back inside -> entry (A market / B retest limit / C micro-swing break)
  -> fixed meaningful target that must remain INSIDE the box -> hard structural stop.

Sign convention: side = +1 fades the lower edge (long), side = -1 fades the upper edge
(short). With `s` as the side sign, the geometry collapses to single formulas, e.g. the
zone boundary is `edge + s * edge_pct * width` for both sides.

No averaging, no martingale, no stop widening: one bracket per attempt, one attempt per
side per consolidation episode, optional one trade / stop-after-loss per day.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .data import add_trading_day
from .engine import BracketExit, market_fill, run_bracket, scan_limit_entry, scan_stop_entry
from .features import compute_features, load_news_blocks
from .ranges import RangeView, detect_ranges, mid_crossings
from .sessions import assign_sessions, day_reference_prices, parse_hhmm
from .volume_profile import Profile, volume_profile


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_px: float
    exit_px: float
    tp_px: float
    stop_px: float
    pnl_points: float
    pnl_usd: float          # per contract, net of commissions (slippage embedded in fills)
    costs_usd: float
    exit_reason: str
    duration_min: float
    session: str
    trading_day: object
    episode: int
    entry_type: str
    range_method: str
    tp_mode: str
    stop_model: str
    box_high: float
    box_low: float
    box_width: float
    box_mid: float
    probe_extreme: float
    overshoot_points: float
    crossings: int
    vwap_slope: float
    day_move_pct: float
    prior_day_atr: float
    width_vs_atr: float
    rv_ratio: float
    mae_points: float
    mfe_points: float
    tp_distance: float
    stop_distance: float
    target_inside: bool
    slippage_ticks: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Skip:
    ts: pd.Timestamp
    side: str
    reasons: list[str]


@dataclass
class BacktestResult:
    trades: list[Trade]
    skips: list[Skip]
    n_days: int
    n_valid_box_bars: int
    slippage_ticks: int


@dataclass
class _Snapshot:
    box_high: float
    box_low: float
    width: float
    mid: float
    win_start: int
    win_end: int
    episode: int
    zone_boundary: float
    profile: Optional[Profile]


@dataclass
class _ProbeState:
    active: bool = False
    bars: int = 0
    extreme: float = float("nan")
    reclaims: int = 0
    snap: Optional[_Snapshot] = None

    def reset(self) -> None:
        self.active = False
        self.bars = 0
        self.extreme = float("nan")
        self.reclaims = 0
        self.snap = None


def prepare(df_raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Heavy, strategy-independent preprocessing (cacheable across grid combos)."""
    df = add_trading_day(df_raw, cfg.sessions.day_roll_hour)
    df = assign_sessions(df, cfg.sessions)
    df = day_reference_prices(df)
    df = compute_features(df, cfg)
    return df


def prepare_signature(cfg: Config) -> tuple:
    """Configs sharing this signature can reuse the same prepared frame.
    Only inputs of `prepare()` matter here — range-detection params are applied
    later, inside run_backtest, and need no re-preparation."""
    f, s = cfg.filters, cfg.sessions
    return (
        s.day_roll_hour,
        tuple((d.name, d.start, d.end) for d in s.definitions),
        f.vwap_anchor, f.vwap_slope_window_minutes, f.prior_atr_days,
        f.rv_window_minutes, f.rv_median_days, tuple(f.day_move_refs),
        f.extreme_extension_window_minutes,
    )


def _session_block_ends(day_codes: np.ndarray, sess_codes: np.ndarray) -> np.ndarray:
    n = len(day_codes)
    block_end = np.empty(n, dtype=np.int64)
    start = 0
    for i in range(1, n + 1):
        if i == n or day_codes[i] != day_codes[start] or sess_codes[i] != sess_codes[start]:
            block_end[start:i] = i - 1
            start = i
    return block_end


def _day_flat_deadline(minutes: np.ndarray, day_codes: np.ndarray, flat_by: Optional[str]) -> np.ndarray:
    """Last allowed bar position per day given the hard flat time (afternoon of the trading day)."""
    n = len(day_codes)
    deadline = np.empty(n, dtype=np.int64)
    if flat_by is None:
        flat_min = None
    else:
        t = parse_hhmm(flat_by)
        flat_min = t.hour * 60 + t.minute
    start = 0
    for i in range(1, n + 1):
        if i == n or day_codes[i] != day_codes[start]:
            last = i - 1
            if flat_min is not None:
                pm = np.nonzero((minutes[start:i] >= flat_min) & (minutes[start:i] < 17 * 60))[0]
                if len(pm):
                    last = start + pm[0] - 1
            deadline[start:i] = max(last, start)
            start = i
    return deadline


def run_backtest(df: pd.DataFrame, cfg: Config, slippage_ticks: Optional[int] = None) -> BacktestResult:
    """Single-pass event backtest over a prepared frame. `slippage_ticks` overrides config."""
    ins, ent, ext, eng, flt, rng = (
        cfg.instrument, cfg.entry, cfg.exit, cfg.engine, cfg.filters, cfg.range,
    )
    slip_ticks = ins.slippage_ticks if slippage_ticks is None else slippage_ticks
    slip = slip_ticks * ins.tick_size
    tick = ins.tick_size
    rt_cost = 2.0 * ins.commission_per_side
    bar_min = cfg.data.bar_minutes

    idx = df.index
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    day_codes, _ = pd.factorize(df["trading_day"])
    sess_vals = df["session"].astype(object).fillna("none").to_numpy()
    sess_codes, _ = pd.factorize(sess_vals)
    # derive tradability from the CURRENT cfg (cheap), so sweeping sessions.trade
    # never forces a re-prepare of the heavy feature frame
    tradable_names = [d.name for d in cfg.sessions.definitions
                      if d.tradable and d.name in cfg.sessions.trade]
    tradable = np.isin(sess_vals, tradable_names)
    vwap_slope = df["vwap_slope"].to_numpy(float)
    prior_atr = df["prior_day_atr"].to_numpy(float)
    rv_ratio = df["rv_ratio"].to_numpy(float)
    day_move = df["day_move_pct"].to_numpy(float)
    ext_up = df["ext_up"].to_numpy(float)
    ext_dn = df["ext_dn"].to_numpy(float)
    news_blocked = load_news_blocks(df, cfg)

    rv: RangeView = detect_ranges(df, cfg)
    block_end = _session_block_ends(day_codes, sess_codes)
    flat_deadline = _day_flat_deadline(minutes, day_codes, ext.flat_by)
    max_hold_bars = max(1, ext.max_holding_minutes // bar_min)
    min_bars_to_flat = max(1, ent.min_minutes_to_flat // bar_min)

    need_profile = ent.edge_def in ("vah_val", "hybrid") or ext.tp_mode == "poc"
    profile_cache: dict[tuple[int, int], Optional[Profile]] = {}

    def get_profile(ws: int, we: int) -> Optional[Profile]:
        key = (ws // 5, we // 5)
        if key not in profile_cache:
            profile_cache[key] = volume_profile(
                h[ws:we], l[ws:we], v[ws:we], rng.profile_bin_points, rng.value_area
            )
        return profile_cache[key]

    def zone_boundary(s: int, bh: float, bl: float, width: float, prof: Optional[Profile]) -> float:
        edge = bh if s < 0 else bl
        zb_pct = edge + s * ent.edge_pct * width
        if ent.edge_def == "absolute":
            return edge
        if ent.edge_def == "pct":
            return zb_pct
        va = (prof.vah if s < 0 else prof.val) if prof else None
        if va is None:
            return zb_pct
        if ent.edge_def == "vah_val":
            return va
        return max(zb_pct, va) if s < 0 else min(zb_pct, va)  # hybrid: beyond BOTH

    trades: list[Trade] = []
    skips: list[Skip] = []
    sides_enabled = tuple(
        s for s, name in ((+1, "long"), (-1, "short")) if name in ent.sides
    )
    states = {+1: _ProbeState(), -1: _ProbeState()}

    n = len(df)
    t = 0
    cur_day = -1
    trades_today = 0
    losses_today = 0
    attempted: set[tuple[int, int]] = set()

    def check_filters(t: int, s: int, snap: _Snapshot) -> list[str]:
        fails: list[str] = []

        def gate(value: float, limit: Optional[float], name: str, mode: str = "le") -> None:
            if limit is None:
                return
            if not np.isfinite(value):
                fails.append(f"warmup_{name}")
            elif mode == "le" and value > limit:
                fails.append(name)
            elif mode == "lt" and value >= limit:
                fails.append(name)

        gate(day_move[t], flt.max_day_move_pct, "day_move")
        if flt.max_range_vs_prior_atr is not None:
            if not np.isfinite(prior_atr[t]):
                fails.append("warmup_prior_atr")
            elif snap.width > flt.max_range_vs_prior_atr * prior_atr[t]:
                fails.append("range_vs_atr")
        gate(vwap_slope[t], flt.max_vwap_slope_points, "vwap_slope")
        gate(rv_ratio[t], flt.rv_max_vs_median, "rv_expansion")
        ext_move = ext_up[t] if s < 0 else ext_dn[t]
        gate(ext_move, flt.extreme_extension_points, "fresh_extreme", mode="lt")
        if news_blocked[t]:
            fails.append("news")
        if flt.min_mid_crossings and snap.win_start >= 0:
            x = mid_crossings(c[snap.win_start:snap.win_end], snap.mid)
            if x < flt.min_mid_crossings:
                fails.append("acceptance")
        return fails

    def build_exit_levels(s: int, entry_px: float, snap: _Snapshot, probe_extreme: float):
        """Returns (stop_px, tp_px, fail_reasons)."""
        fails: list[str] = []
        edge = snap.box_high if s < 0 else snap.box_low
        opposite = snap.box_low if s < 0 else snap.box_high
        if ext.stop_model == "edge_plus":
            stop_px = edge - s * ext.stop_buffer_points
        elif ext.stop_model == "excursion_plus":
            stop_px = probe_extreme - s * ext.stop_buffer_points
        else:  # range_frac
            stop_px = entry_px - s * ext.stop_range_frac * snap.width
        stop_dist = s * (entry_px - stop_px)
        if stop_dist > ext.max_stop_points:
            if ext.stop_cap_mode == "skip":
                fails.append("stop_too_wide")
            stop_px = entry_px - s * ext.max_stop_points
            stop_dist = ext.max_stop_points
        if stop_dist < ext.min_stop_points:
            stop_px = entry_px - s * ext.min_stop_points
            stop_dist = ext.min_stop_points

        if ext.tp_mode == "fixed":
            tp_px = entry_px + s * ext.tp_points
        elif ext.tp_mode == "mid":
            tp_px = snap.mid
        elif ext.tp_mode == "poc":
            tp_px = snap.profile.poc if snap.profile else snap.mid
        else:  # opposite_25
            tp_px = opposite - s * 0.25 * snap.width
        tp_dist = s * (tp_px - entry_px)
        min_tp = ext.tp_points if ext.tp_mode == "fixed" else ext.min_tp_points
        if tp_dist < min_tp - 1e-9:
            fails.append("tp_too_small")
        if ext.require_target_inside:
            inside_ok = s * ((opposite - s * ext.target_inside_buffer_points) - tp_px) >= -1e-9
            if not inside_ok:
                fails.append("target_outside_range")
        return stop_px, tp_px, fails

    while t < n:
        if day_codes[t] != cur_day:
            cur_day = day_codes[t]
            trades_today = 0
            losses_today = 0
            for st in states.values():
                st.reset()

        for s in sides_enabled:
            st = states[s]
            box_ok = rv.valid[t]
            bh, bl = rv.box_high[t], rv.box_low[t]

            if not st.active:
                if not box_ok:
                    continue
                width = bh - bl
                prof = get_profile(rv.win_start[t], rv.win_end[t]) if need_profile else None
                zb = zone_boundary(s, bh, bl, width, prof)
                probe_px = h[t] if s < 0 else l[t]
                if s * (probe_px - zb) <= 0:  # reached into the edge zone
                    st.active = True
                    st.bars = 0
                    st.extreme = probe_px
                    st.reclaims = 0
                    st.snap = _Snapshot(
                        box_high=bh, box_low=bl, width=width, mid=(bh + bl) / 2.0,
                        win_start=int(rv.win_start[t]), win_end=int(rv.win_end[t]),
                        episode=int(rv.episode[t]), zone_boundary=zb, profile=prof,
                    )
                continue

            # --- probe in progress ---
            snap = st.snap
            st.bars += 1
            st.extreme = max(st.extreme, h[t]) if s < 0 else min(st.extreme, l[t])
            edge = snap.box_high if s < 0 else snap.box_low
            breakout = s * (c[t] - (edge - s * ent.max_overshoot_points)) < 0
            if breakout or st.bars > ent.probe_ttl_bars:
                st.reset()
                continue

            opposite = snap.box_low if s < 0 else snap.box_high
            reclaimed = (s * (c[t] - snap.zone_boundary) > 0) and (s * (c[t] - opposite) < 0)
            if not reclaimed:
                st.reclaims = 0
                continue
            st.reclaims += 1
            if st.reclaims < ent.reclaim_bars:
                continue

            # --- trigger: evaluate once per probe, then consume it ---
            probe_extreme = st.extreme
            snap_local = snap
            st.reset()

            if ent.one_attempt_per_side and (snap_local.episode, s) in attempted:
                continue
            if trades_today >= ent.max_trades_per_day:
                continue
            if ent.stop_after_daily_loss and losses_today > 0:
                continue
            entry_start = t + 1
            if entry_start >= n or day_codes[entry_start] != cur_day:
                continue
            if not tradable[entry_start]:
                skips.append(Skip(idx[t], "long" if s > 0 else "short", ["session_not_tradable"]))
                continue
            flat_idx = min(
                block_end[entry_start],
                flat_deadline[entry_start],
                entry_start + max_hold_bars,
            )
            if flat_idx - entry_start < min_bars_to_flat:
                skips.append(Skip(idx[t], "long" if s > 0 else "short", ["too_close_to_flat"]))
                continue
            if news_blocked[entry_start]:
                skips.append(Skip(idx[t], "long" if s > 0 else "short", ["news"]))
                continue

            overshoot = max(0.0, (probe_extreme - edge) if s < 0 else (edge - probe_extreme))
            if overshoot < ent.min_overshoot_points:
                skips.append(Skip(idx[t], "long" if s > 0 else "short", ["overshoot_too_small"]))
                continue

            fails = check_filters(t, s, snap_local)
            if fails:
                skips.append(Skip(idx[t], "long" if s > 0 else "short", fails))
                continue

            if ext.tp_mode == "poc" and snap_local.profile is None:
                snap_local.profile = get_profile(snap_local.win_start, snap_local.win_end)

            # --- entry execution ---
            entry_type = ent.type
            if entry_type == "A":
                entry_idx = entry_start
                entry_px = market_fill(s, o[entry_idx], slip)
                stop_px, tp_px, lvl_fails = build_exit_levels(s, entry_px, snap_local, probe_extreme)
                if lvl_fails:
                    skips.append(Skip(idx[t], "long" if s > 0 else "short", lvl_fails))
                    continue
                bx = run_bracket(
                    s, entry_idx, entry_px, stop_px, tp_px, o, h, l, c, flat_idx,
                    tick, slip, eng.tp_fill_through_ticks, eng.both_touch,
                    invalidation_level=(edge - s * ext.invalidation_buffer_points)
                    if ext.invalidate_close_outside else None,
                )
            elif entry_type == "B":
                if ent.retest_level == "halfway":
                    limit_px = (c[t] + snap_local.zone_boundary) / 2.0
                else:
                    limit_px = snap_local.zone_boundary
                stop_px, tp_px, lvl_fails = build_exit_levels(s, limit_px, snap_local, probe_extreme)
                if lvl_fails:
                    skips.append(Skip(idx[t], "long" if s > 0 else "short", lvl_fails))
                    continue
                last_order_bar = min(entry_start + ent.retest_ttl_bars - 1, flat_idx)
                fill = scan_limit_entry(
                    s, limit_px, stop_px, entry_start, last_order_bar, o, h, l,
                    tick, eng.limit_fill_through_ticks,
                )
                if fill is None:
                    skips.append(Skip(idx[t], "long" if s > 0 else "short", ["no_retest_fill"]))
                    continue
                entry_idx, entry_px, stopped_same_bar = fill
                if stopped_same_bar:
                    stop_fill = (min(o[entry_idx], stop_px) - slip) if s > 0 else (max(o[entry_idx], stop_px) + slip)
                    bx = BracketExit(entry_idx, stop_fill, "stop",
                                     mae_points=s * (entry_px - stop_px), mfe_points=0.0)
                elif entry_idx >= flat_idx:
                    px = market_fill(-s, c[flat_idx], slip)
                    bx = BracketExit(flat_idx, px, "time",
                                     mae_points=max(0.0, s * (entry_px - (l[entry_idx] if s > 0 else h[entry_idx]))),
                                     mfe_points=max(0.0, s * ((h[entry_idx] if s > 0 else l[entry_idx]) - entry_px)))
                else:
                    seed_mae = max(0.0, s * (entry_px - (l[entry_idx] if s > 0 else h[entry_idx])))
                    seed_mfe = max(0.0, s * ((h[entry_idx] if s > 0 else l[entry_idx]) - entry_px))
                    bx = run_bracket(
                        s, entry_idx + 1, entry_px, stop_px, tp_px, o, h, l, c, flat_idx,
                        tick, slip, eng.tp_fill_through_ticks, eng.both_touch,
                        invalidation_level=(edge - s * ext.invalidation_buffer_points)
                        if ext.invalidate_close_outside else None,
                    )
                    bx.mae_points = max(bx.mae_points, seed_mae)
                    bx.mfe_points = max(bx.mfe_points, seed_mfe)
            else:  # entry C: micro swing break stop order
                k = max(1, ent.confirm_swing_bars)
                lo_k = max(0, t - k + 1)
                trig_px = (np.max(h[lo_k:t + 1]) + tick) if s > 0 else (np.min(l[lo_k:t + 1]) - tick)
                last_order_bar = min(entry_start + ent.confirm_ttl_bars - 1, flat_idx)
                cancel_level = edge - s * ent.max_overshoot_points
                fill = scan_stop_entry(s, trig_px, entry_start, last_order_bar, o, h, l, c, slip, cancel_level)
                if fill is None:
                    skips.append(Skip(idx[t], "long" if s > 0 else "short", ["no_confirmation_fill"]))
                    continue
                entry_idx, entry_px = fill
                stop_px, tp_px, lvl_fails = build_exit_levels(s, entry_px, snap_local, probe_extreme)
                if lvl_fails:
                    skips.append(Skip(idx[t], "long" if s > 0 else "short", lvl_fails))
                    continue
                bx = run_bracket(
                    s, entry_idx, entry_px, stop_px, tp_px, o, h, l, c, flat_idx,
                    tick, slip, eng.tp_fill_through_ticks, eng.both_touch,
                    invalidation_level=(edge - s * ext.invalidation_buffer_points)
                    if ext.invalidate_close_outside else None,
                )

            pnl_points = s * (bx.exit_px - entry_px)
            pnl_usd = pnl_points * ins.point_value - rt_cost
            attempted.add((snap_local.episode, s))
            trades_today += 1
            if pnl_usd < 0:
                losses_today += 1
            x = mid_crossings(c[snap_local.win_start:snap_local.win_end], snap_local.mid) \
                if snap_local.win_start >= 0 else 0
            trades.append(Trade(
                side="long" if s > 0 else "short",
                entry_time=idx[entry_idx], exit_time=idx[bx.exit_idx],
                entry_px=float(entry_px), exit_px=float(bx.exit_px),
                tp_px=float(tp_px), stop_px=float(stop_px),
                pnl_points=float(pnl_points), pnl_usd=float(pnl_usd), costs_usd=float(rt_cost),
                exit_reason=bx.reason,
                duration_min=float((bx.exit_idx - entry_idx) * bar_min),
                session=str(sess_vals[entry_idx]), trading_day=df["trading_day"].iloc[entry_idx],
                episode=int(snap_local.episode), entry_type=entry_type,
                range_method=rng.method, tp_mode=ext.tp_mode, stop_model=ext.stop_model,
                box_high=float(snap_local.box_high), box_low=float(snap_local.box_low),
                box_width=float(snap_local.width), box_mid=float(snap_local.mid),
                probe_extreme=float(probe_extreme),
                overshoot_points=float(overshoot),
                crossings=int(x),
                vwap_slope=float(vwap_slope[t]), day_move_pct=float(day_move[t]),
                prior_day_atr=float(prior_atr[t]),
                width_vs_atr=float(snap_local.width / prior_atr[t]) if np.isfinite(prior_atr[t]) and prior_atr[t] > 0 else float("nan"),
                rv_ratio=float(rv_ratio[t]),
                mae_points=float(bx.mae_points), mfe_points=float(bx.mfe_points),
                tp_distance=float(s * (tp_px - entry_px)), stop_distance=float(s * (entry_px - stop_px)),
                target_inside=bool(
                    s * ((snap_local.box_low if s < 0 else snap_local.box_high) - tp_px) >= 0
                ),
                slippage_ticks=int(slip_ticks),
            ))
            t = bx.exit_idx  # no overlapping trades; resume scanning after the exit
            for st2 in states.values():
                st2.reset()
            break
        t += 1

    return BacktestResult(
        trades=trades,
        skips=skips,
        n_days=int(pd.unique(df["trading_day"]).size),
        n_valid_box_bars=int(rv.valid.sum()),
        slippage_ticks=slip_ticks,
    )
