"""Conservative execution engine.

Fill assumptions (deliberately pessimistic — setups that need perfect fills must die here):
  * market orders fill at the NEXT bar open, worsened by slippage
  * stop (loss) orders fill at stop price worsened by slippage; gaps fill at the open, worse
  * profit-target limits only fill if price trades THROUGH them by `tp_fill_through_ticks`,
    and gaps through the target still fill at the target price, never better
  * a bar touching both stop and target counts as a LOSS by default
  * entry limit orders need trade-through as well; a fill bar that also touches the stop
    is an immediate loss
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BracketExit:
    exit_idx: int
    exit_px: float
    reason: str           # tp | stop | invalidation | time
    mae_points: float     # max adverse excursion, positive points
    mfe_points: float     # max favorable excursion, positive points


def market_fill(side: int, open_px: float, slip_pts: float) -> float:
    """side: +1 long, -1 short. Market orders fill worse by slippage."""
    return open_px + side * slip_pts


def run_bracket(
    side: int,
    entry_idx: int,
    entry_px: float,
    stop_px: float,
    tp_px: float,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    flat_idx: int,
    tick: float,
    slip_pts: float,
    tp_through_ticks: int = 1,
    both_touch: str = "loss",
    invalidation_level: float | None = None,  # close beyond this (against us) => exit next open
) -> BracketExit:
    """Walk bars from entry to forced-flat, applying conservative exit rules."""
    through = tp_through_ticks * tick
    mae = 0.0
    mfe = 0.0
    pending_invalidation = False

    j = entry_idx
    while j <= flat_idx:
        if pending_invalidation:
            px = market_fill(-side, open_[j], slip_pts)
            return BracketExit(j, px, "invalidation", mae, mfe)

        if side > 0:
            mae = max(mae, entry_px - low[j])
            mfe = max(mfe, high[j] - entry_px)
            stop_hit = low[j] <= stop_px
            tp_hit = high[j] >= tp_px + through
            stop_fill = min(open_[j], stop_px) - slip_pts
        else:
            mae = max(mae, high[j] - entry_px)
            mfe = max(mfe, entry_px - low[j])
            stop_hit = high[j] >= stop_px
            tp_hit = low[j] <= tp_px - through
            stop_fill = max(open_[j], stop_px) + slip_pts

        if stop_hit and tp_hit:
            if both_touch == "loss":
                return BracketExit(j, stop_fill, "stop", mae, mfe)
            return BracketExit(j, tp_px, "tp", mae, mfe)
        if stop_hit:
            return BracketExit(j, stop_fill, "stop", mae, mfe)
        if tp_hit:
            return BracketExit(j, tp_px, "tp", mae, mfe)

        if invalidation_level is not None:
            broke = close[j] > invalidation_level if side < 0 else close[j] < invalidation_level
            if broke:
                if j == flat_idx:
                    px = market_fill(-side, close[j], slip_pts)
                    return BracketExit(j, px, "invalidation", mae, mfe)
                pending_invalidation = True

        if j == flat_idx:
            px = market_fill(-side, close[j], slip_pts)
            return BracketExit(j, px, "time", mae, mfe)
        j += 1

    # unreachable for well-formed inputs, but never leave a position dangling
    px = market_fill(-side, close[flat_idx], slip_pts)
    return BracketExit(flat_idx, px, "time", mae, mfe)


def scan_limit_entry(
    side: int,
    limit_px: float,
    stop_px: float,
    start_idx: int,
    last_idx: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tick: float,
    through_ticks: int = 1,
) -> tuple[int, float, bool] | None:
    """Look for a conservative limit-entry fill.

    Returns (fill_idx, fill_px, stopped_same_bar) or None if never filled.
    Long limits sit below price, short limits above. A fill requires trade-through;
    a bar that fills AND touches the stop counts as filled-then-stopped.
    """
    through = through_ticks * tick
    for j in range(start_idx, last_idx + 1):
        if side > 0:
            filled = low[j] <= limit_px - through or open_[j] <= limit_px
            stopped = low[j] <= stop_px
        else:
            filled = high[j] >= limit_px + through or open_[j] >= limit_px
            stopped = high[j] >= stop_px
        if filled:
            return j, limit_px, bool(stopped)
    return None


def scan_stop_entry(
    side: int,
    trigger_px: float,
    start_idx: int,
    last_idx: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    slip_pts: float,
    cancel_level: float | None = None,
) -> tuple[int, float] | None:
    """Stop-order entry (entry type C): trade-through trigger, slipped fill, optional
    cancel if a bar CLOSES beyond cancel_level against the setup before the fill."""
    for j in range(start_idx, last_idx + 1):
        if side > 0:
            if high[j] >= trigger_px:
                return j, max(open_[j], trigger_px) + slip_pts
            if cancel_level is not None and close[j] < cancel_level:
                return None
        else:
            if low[j] <= trigger_px:
                return j, min(open_[j], trigger_px) - slip_pts
            if cancel_level is not None and close[j] > cancel_level:
                return None
    return None
