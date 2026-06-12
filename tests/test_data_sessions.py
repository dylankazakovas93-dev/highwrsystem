import pandas as pd
import pytest

from rangefade.config import load_config
from rangefade.data import add_trading_day, load_ohlcv
from rangefade.sessions import assign_sessions, day_reference_prices

from helpers import make_bars


def test_load_csv_variants(tmp_path):
    df = make_bars([100, 101, 102], start="2025-01-06 18:00")
    p = tmp_path / "a.csv"
    df.reset_index().rename(columns={"ts": "DateTime", "open": "Open"}).to_csv(p, index=False)
    out = load_ohlcv(p)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 3
    assert str(out.index.tz) == "America/New_York"


def test_load_split_date_time_utc(tmp_path):
    raw = pd.DataFrame({
        "Date": ["2025-01-06", "2025-01-06"],
        "Time": ["23:30:00", "23:31:00"],   # UTC -> 18:30 ET (EST)
        "Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [0.5, 1.5],
        "Close": [1.5, 2.5], "Volume": [10, 20],
    })
    p = tmp_path / "b.csv"
    raw.to_csv(p, index=False)
    cfg = load_config()
    cfg.data.tz_input = "UTC"
    out = load_ohlcv(p, cfg.data)
    assert out.index[0].hour == 18 and out.index[0].minute == 30
    out = add_trading_day(out)
    # 18:30 ET on Jan 6 belongs to the Jan 7 trading day
    assert str(out["trading_day"].iloc[0]) == "2025-01-07"


def test_session_assignment_and_wrap():
    cfg = load_config()
    bars = make_bars([1] * 5, start="2025-01-06 18:30")  # 18:30 ET
    bars = add_trading_day(bars)
    bars = assign_sessions(bars, cfg.sessions)
    assert (bars["session"] == "asia").all()

    late = make_bars([1] * 5, start="2025-01-07 01:30")  # 01:30 ET still asia
    late = add_trading_day(late)
    late = assign_sessions(late, cfg.sessions)
    assert (late["session"] == "asia").all()
    assert str(late["trading_day"].iloc[0]) == "2025-01-07"

    rth = make_bars([1] * 5, start="2025-01-07 09:45")
    rth = assign_sessions(add_trading_day(rth), cfg.sessions)
    assert (rth["session"] == "ny_morning").all()
    assert rth["tradable"].all()

    last = make_bars([1] * 5, start="2025-01-07 15:30")
    last = assign_sessions(add_trading_day(last), cfg.sessions)
    assert (last["session"] == "ny_last_hour").all()
    assert not last["tradable"].any()  # default: no last-hour trading


def test_reference_prices():
    cfg = load_config()
    d1 = make_bars([100, 110, 120], start="2025-01-06 10:00")
    d2 = make_bars([130, 140, 150], start="2025-01-07 10:00")
    df = pd.concat([d1, d2])
    df = assign_sessions(add_trading_day(df), cfg.sessions)
    df = day_reference_prices(df)
    # day 2 prior_close = day 1 last close
    assert df["prior_close"].iloc[-1] == 120
    assert df["day_open"].iloc[-1] == df["open"].iloc[3]
