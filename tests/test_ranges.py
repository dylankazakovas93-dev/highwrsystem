import numpy as np

from rangefade.ranges import detect_ranges, mid_crossings
from rangefade.strategy import prepare

from helpers import lab_config, make_bars, wave


def _prep(closes, cfg, **kw):
    return prepare(make_bars(closes, **kw), cfg)


def test_rolling_detects_flat_range():
    cfg = lab_config()
    closes = wave(21000, 21030, period=10, n=120)
    df = _prep(closes, cfg)
    rv = detect_ranges(df, cfg)
    assert rv.valid.any(), "flat oscillation must produce a valid box"
    i = np.nonzero(rv.valid)[0][0]
    assert 21025 <= rv.box_high[i] <= 21035
    assert 20995 <= rv.box_low[i] <= 21005
    assert rv.episode[i] >= 0
    assert rv.win_start[i] >= 0 and rv.win_end[i] > rv.win_start[i]


def test_adaptive_rejects_trend_rolling_does_not():
    cfg = lab_config()
    closes = [21000 + 1.5 * i for i in range(120)]  # steady trend; window width ~45pts
    df = _prep(closes, cfg)
    cfg.range.method = "rolling"
    roll = detect_ranges(df, cfg)
    cfg.range.method = "adaptive"
    adap = detect_ranges(df, cfg)
    assert roll.valid.sum() > 0, "rolling alone is fooled by a steady trend"
    assert adap.valid.sum() == 0, "adaptive must reject one-directional structure"


def test_adaptive_accepts_balance():
    cfg = lab_config()
    cfg.range.method = "adaptive"
    cfg.range.extremes_min_age_minutes = 10
    closes = wave(21000, 21030, period=10, n=180)
    df = _prep(closes, cfg)
    rv = detect_ranges(df, cfg)
    assert rv.valid.any()


def test_session_method_freezes_formation_box():
    cfg = lab_config()
    cfg.range.method = "session"
    cfg.range.formation_session = "asia"
    # asia bars 18:00-02:00 oscillating, then NY-morning bars
    asia = make_bars(wave(21000, 21040, 12, 240), start="2025-03-03 20:00")
    ny = make_bars(wave(21010, 21030, 12, 120), start="2025-03-04 09:30")
    import pandas as pd
    df = prepare(pd.concat([asia, ny]), cfg)
    rv = detect_ranges(df, cfg)
    ny_pos = np.arange(len(df))[df.index >= ny.index[0]]
    assert rv.valid[ny_pos].all()
    assert np.unique(rv.box_high[ny_pos]).size == 1, "session box must be frozen"
    assert rv.box_high[ny_pos][0] == 21041.0  # asia high + hi_off
    assert rv.box_low[ny_pos][0] == 20999.0


def test_min_width_rejects_tight_range():
    cfg = lab_config()
    cfg.range.min_width_points = 50.0
    closes = wave(21000, 21020, 10, 120)  # ~22pt wide < 50 -> no trades allowed
    df = _prep(closes, cfg)
    rv = detect_ranges(df, cfg)
    assert rv.valid.sum() == 0


def test_mid_crossings():
    assert mid_crossings(np.array([1.0, -1, 1, -1, 1]), 0.0) == 4
    assert mid_crossings(np.array([1.0, 2, 3]), 0.0) == 0
