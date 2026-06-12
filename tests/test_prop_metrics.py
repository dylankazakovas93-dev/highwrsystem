import numpy as np
import pandas as pd

from rangefade.config import PropCfg
from rangefade.metrics import max_losing_streak, summarize, wilson_ci
from rangefade.prop_sim import run_prop_mc, simulate_account


def _rules(**kw):
    r = PropCfg()
    r.buffer = 2000.0
    r.daily_loss_limit = 1000.0
    r.payout_threshold = 3000.0
    r.min_trading_days = 5
    r.consistency_max_day_frac = 0.5
    r.dd_type = "static"
    r.max_trading_days = 60
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_account_all_wins_pays_out():
    days = [np.array([300.0])] * 20
    out = simulate_account(days, _rules())
    assert out.status == "payout"
    assert out.traded_days == 10  # 10 * 300 = 3000
    assert out.trades == 10


def test_account_all_losses_fails():
    days = [np.array([-300.0])] * 20
    out = simulate_account(days, _rules())
    assert out.status == "fail"
    assert out.traded_days == 7  # cum -2100 <= -2000 on day 7


def test_daily_loss_limit_locks_day_but_survives():
    days = [np.array([-600.0, -600.0, 99999.0])] + [np.array([800.0])] * 10
    out = simulate_account(days, _rules())
    # third trade must NOT execute (daily lockout): equity after day1 = -1200
    assert out.status == "payout"
    assert out.best_day <= 800.0 + 1e-9


def test_trailing_vs_static_drawdown():
    days = [np.array([500.0]), np.array([500.0]), np.array([-900.0]), np.array([-900.0])]
    static = simulate_account(days, _rules(buffer=1000.0, dd_type="static"))
    trailing = simulate_account(days, _rules(buffer=1000.0, dd_type="trailing_eod"))
    assert static.status != "fail"     # equity -800 > -1000 floor
    assert trailing.status == "fail"   # floor trailed up to 0 after +1000 HWM


def test_consistency_rule_delays_payout():
    days = [np.array([800.0]), np.array([300.0])] + [np.array([200.0])] * 20
    out = simulate_account(days, _rules())
    assert out.status == "payout"
    assert out.traded_days > 5
    assert out.best_day <= 0.5 * out.final_pnl + 1e-9


def test_prop_mc_runs():
    tdf = pd.DataFrame({
        "entry_time": pd.date_range("2025-01-06 10:00", periods=60, freq="D", tz="America/New_York"),
        "trading_day": [d.date() for d in pd.date_range("2025-01-06", periods=60, freq="D")],
        "pnl_usd": [300.0] * 48 + [-300.0] * 12,
    })
    res = run_prop_mc(tdf, _rules(n_paths=200))
    assert res["paths"] == 200
    assert 0.9 <= res["p_payout"] <= 1.0


def test_metrics_basics():
    s = pd.Series([1, -1, -1, -1, 2, -1])
    assert max_losing_streak(s) == 3
    lo, hi = wilson_ci(80, 100)
    assert 0.70 < lo < 0.81 < hi < 0.88
    tdf = pd.DataFrame({
        "pnl_usd": [300.0, 300.0, -150.0],
        "pnl_points": [15.0, 15.0, -7.5],
        "duration_min": [10, 12, 8],
        "trading_day": ["a", "a", "b"],
        "target_inside": [True, True, True],
    })
    out = summarize(tdf)
    assert out["trades"] == 3
    assert abs(out["win_rate"] - 2 / 3) < 1e-3  # summarize() rounds to 4 decimals
    assert out["profit_factor"] == 4.0
    assert out["max_losing_streak"] == 1
