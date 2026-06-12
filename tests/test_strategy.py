import numpy as np

from rangefade.strategy import prepare, run_backtest

from helpers import lab_config, make_bars, wave


def _run(closes, cfg, **kw):
    df = prepare(make_bars(closes, **kw), cfg)
    return run_backtest(df, cfg)


def test_fade_wins_in_clean_rotation():
    cfg = lab_config()
    closes = wave(21000, 21030, period=12, n=200)
    res = _run(closes, cfg)
    assert len(res.trades) >= 1
    tps = [t for t in res.trades if t.exit_reason == "tp"]
    assert tps, f"expected tp exits, got {[t.exit_reason for t in res.trades]}"
    for t in tps:
        assert abs(t.pnl_points - cfg.exit.tp_points) < 1e-9
        assert t.target_inside
    # bounded downside everywhere
    for t in res.trades:
        assert t.pnl_points >= -(cfg.exit.max_stop_points + 2 * cfg.instrument.tick_size)


def _breakout_closes():
    # rotation forms the box, then probe + reclaim -> short trigger, then a rip
    closes = wave(21000, 21030, period=8, n=48)
    closes += [21033.0, 21025.0]           # probe + reclaim
    closes += [21030.0, 21036.0, 21045.0]  # rip through edge_plus stop
    closes += [21050.0] * 20               # room so min_minutes_to_flat doesn't block entry
    return closes


def test_stop_loss_is_bounded_on_breakout():
    cfg = lab_config()
    cfg.entry.one_attempt_per_side = False  # the wave itself fades earlier peaks
    res = _run(_breakout_closes(), cfg)
    stops = [t for t in res.trades if t.exit_reason == "stop" and t.side == "short"]
    assert stops, f"expected a stopped short, trades={[(t.side, t.exit_reason) for t in res.trades]}"
    t = stops[0]
    assert t.pnl_points < 0
    assert -t.pnl_points <= cfg.exit.max_stop_points + 2 * cfg.instrument.tick_size
    assert t.stop_px > t.box_high  # structural stop outside the box


def test_target_outside_range_is_skipped():
    cfg = lab_config()
    cfg.exit.tp_points = 50.0   # box is only ~32 wide -> target cannot stay inside
    cfg.exit.min_tp_points = 50.0
    closes = wave(21000, 21030, period=12, n=200)
    res = _run(closes, cfg)
    assert len(res.trades) == 0
    reasons = {r for s in res.skips for r in s.reasons}
    assert "target_outside_range" in reasons or "tp_too_small" in reasons


def test_max_trades_per_day_and_one_attempt():
    cfg = lab_config()
    cfg.entry.max_trades_per_day = 1
    closes = wave(21000, 21030, period=12, n=300)
    res = _run(closes, cfg)
    assert len(res.trades) <= 1


def test_no_trades_when_box_too_narrow():
    cfg = lab_config()
    cfg.range.min_width_points = 40.0   # NQ default: don't trade tiny consolidations
    closes = wave(21000, 21020, period=12, n=240)
    res = _run(closes, cfg)
    assert len(res.trades) == 0


def test_filters_block_and_record_skips():
    cfg = lab_config()
    cfg.filters.min_mid_crossings = 999  # impossible acceptance requirement
    closes = wave(21000, 21030, period=12, n=200)
    res = _run(closes, cfg)
    assert len(res.trades) == 0
    assert any("acceptance" in s.reasons for s in res.skips)


def test_overshoot_cancels_probe():
    cfg = lab_config()
    cfg.entry.one_attempt_per_side = False
    cfg.entry.max_overshoot_points = 5.0
    # box, then a bar closing 12pts above the box high: breakout, not edge failure
    closes = wave(21000, 21030, period=8, n=48) + [21043.0] + [21044.0] * 10
    df = prepare(make_bars(closes), cfg)
    res = run_backtest(df, cfg)
    breakout_ts = df.index[48]
    late_shorts = [t for t in res.trades if t.side == "short" and t.entry_time >= breakout_ts]
    assert not late_shorts, "a >overshoot breakout must cancel the fade setup"


def test_slippage_worsens_stop_losses():
    # fixed TP is bracket-relative to the fill (entry and target shift together),
    # so slippage must show up in stopped trades: structural stop fixed, entry worse
    cfg = lab_config()
    cfg.entry.one_attempt_per_side = False
    df = prepare(make_bars(_breakout_closes()), cfg)
    res0 = run_backtest(df, cfg, slippage_ticks=0)
    res2 = run_backtest(df, cfg, slippage_ticks=2)
    s0 = [t for t in res0.trades if t.exit_reason == "stop"]
    s2 = [t for t in res2.trades if t.exit_reason == "stop"]
    assert s0 and s2
    assert s2[0].pnl_usd < s0[0].pnl_usd
    assert sum(t.pnl_usd for t in res2.trades) < sum(t.pnl_usd for t in res0.trades)


def test_entry_types_b_and_c_run():
    for etype in ("B", "C"):
        cfg = lab_config()
        cfg.entry.type = etype
        closes = wave(21000, 21030, period=12, n=300)
        res = _run(closes, cfg)
        for t in res.trades:
            assert t.entry_type == etype
            assert t.pnl_points >= -(cfg.exit.max_stop_points + 2 * cfg.instrument.tick_size)
