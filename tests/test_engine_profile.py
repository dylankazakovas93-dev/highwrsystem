import numpy as np

from rangefade.engine import run_bracket, scan_limit_entry, scan_stop_entry
from rangefade.volume_profile import volume_profile


def _bars(rows):
    """rows: list of (open, high, low, close)"""
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def test_bracket_tp_long():
    o, h, l, c = _bars([
        (100, 101, 99.5, 100.5),
        (100.5, 106.0, 100.0, 105.5),   # trades through tp 105 by >1 tick
    ])
    bx = run_bracket(+1, 0, 100.0, 95.0, 105.0, o, h, l, c, flat_idx=1, tick=0.25,
                     slip_pts=0.0, tp_through_ticks=1)
    assert bx.reason == "tp"
    assert bx.exit_px == 105.0
    assert bx.exit_idx == 1


def test_bracket_tp_needs_trade_through():
    # touches 105.0 exactly but not through -> NOT filled, exits on time
    o, h, l, c = _bars([(100, 105.0, 99.5, 104.0), (104, 105.0, 103, 104.5)])
    bx = run_bracket(+1, 0, 100.0, 95.0, 105.0, o, h, l, c, flat_idx=1, tick=0.25,
                     slip_pts=0.0, tp_through_ticks=1)
    assert bx.reason == "time"


def test_bracket_both_touch_is_loss():
    o, h, l, c = _bars([(100, 106.0, 94.0, 100.0)])
    bx = run_bracket(+1, 0, 100.0, 95.0, 105.0, o, h, l, c, flat_idx=0, tick=0.25,
                     slip_pts=0.0)
    assert bx.reason == "stop"
    assert bx.exit_px <= 95.0


def test_bracket_stop_gap_fills_at_open():
    o, h, l, c = _bars([(92.0, 93.0, 91.0, 92.5)])  # gaps below the 95 stop
    bx = run_bracket(+1, 0, 100.0, 95.0, 105.0, o, h, l, c, flat_idx=0, tick=0.25,
                     slip_pts=0.5)
    assert bx.reason == "stop"
    assert bx.exit_px == 92.0 - 0.5  # open, worsened by slippage


def test_bracket_short_side():
    o, h, l, c = _bars([(100, 100.5, 99, 99.5), (99.5, 100, 84.0, 85.0)])
    bx = run_bracket(-1, 0, 100.0, 107.5, 85.0, o, h, l, c, flat_idx=1, tick=0.25,
                     slip_pts=0.0)
    assert bx.reason == "tp"
    assert bx.exit_px == 85.0
    assert bx.mfe_points >= 15.0


def test_limit_entry_trade_through_and_stop_same_bar():
    o, h, l, c = _bars([(101, 102, 100.5, 101.5), (101.5, 102, 94.0, 95.0)])
    fill = scan_limit_entry(+1, 100.0, 95.0, 0, 1, o, h, l, tick=0.25, through_ticks=1)
    assert fill is not None
    idx, px, stopped = fill
    assert idx == 1 and px == 100.0 and stopped  # filled then stopped same bar


def test_limit_entry_no_fill():
    o, h, l, c = _bars([(101, 102, 100.5, 101.5)])
    assert scan_limit_entry(+1, 100.0, 95.0, 0, 0, o, h, l, tick=0.25) is None


def test_stop_entry_fill_and_cancel():
    o, h, l, c = _bars([(100, 100.5, 99.5, 100), (100, 103.0, 99.9, 102.5)])
    fill = scan_stop_entry(+1, 102.0, 0, 1, o, h, l, c, slip_pts=0.25)
    assert fill == (1, 102.25)
    # cancel: closes below cancel level before trigger
    o, h, l, c = _bars([(100, 100.5, 94.0, 94.5)])
    assert scan_stop_entry(+1, 102.0, 0, 0, o, h, l, c, 0.25, cancel_level=95.0) is None


def test_volume_profile_poc_value_area():
    # all volume concentrated at 100-101 -> POC there, VA contains it
    high = np.array([101.0, 101.0, 101.0, 105.0])
    low = np.array([100.0, 100.0, 100.0, 104.0])
    vol = np.array([1000.0, 1000.0, 1000.0, 10.0])
    p = volume_profile(high, low, vol, bin_points=1.0)
    assert 100.0 <= p.poc <= 101.5
    assert p.val <= 100.5 and p.vah >= 100.5
