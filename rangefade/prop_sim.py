"""Prop-firm game simulator.

Models an evaluation/funded account with: drawdown buffer (static / trailing EOD /
trailing intraday), daily loss limit, payout threshold, minimum trading days,
consistency rule, activation cost. Monte Carlo over resampled trade days estimates
P(payout before failure), time-to-payout, and the EV of the whole game including
sequential re-attempts.

Rescue mode: the SAME strategy stream is gated by account drawdown state. Account
state never generates signals — it only decides whether a valid market setup is
taken. We compare: continue baseline strategy vs switch to rescue setups vs stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PropCfg, RescueCfg
from .monte_carlo import days_from_trades, flat_day_rate, resample_day_paths


@dataclass
class AccountOutcome:
    status: str          # payout | fail | timeout
    traded_days: int
    trades: int
    final_pnl: float
    best_day: float


def simulate_account(day_seq, rules: PropCfg, contracts: int | None = None) -> AccountOutcome:
    """Run one account through a sequence of day PnL arrays (per-contract $)."""
    k = contracts if contracts is not None else rules.contracts
    equity = 0.0                       # PnL relative to starting balance
    hwm = 0.0
    floor = -rules.buffer              # equity level that kills the account
    traded_days = 0
    n_trades = 0
    best_day = 0.0
    total_profit_days: list[float] = []

    for day in day_seq:
        traded_days += 1
        day_pnl = 0.0
        failed = False
        for pnl in day:
            x = pnl * k
            day_pnl += x
            equity += x
            n_trades += 1
            if rules.dd_type == "trailing_intraday":
                hwm = max(hwm, equity)
                floor = _floor_from_hwm(hwm, rules)
            if equity <= floor + 1e-9:
                failed = True
                break
            if rules.daily_loss_limit is not None and day_pnl <= -rules.daily_loss_limit:
                break  # locked out for the day, account survives
        best_day = max(best_day, day_pnl)
        total_profit_days.append(day_pnl)
        if failed:
            return AccountOutcome("fail", traded_days, n_trades, equity, best_day)
        if rules.dd_type in ("trailing_eod", "trailing_intraday"):
            hwm = max(hwm, equity)
            floor = _floor_from_hwm(hwm, rules)

        if equity >= rules.payout_threshold and traded_days >= rules.min_trading_days:
            if rules.consistency_max_day_frac is not None and equity > 0:
                if best_day > rules.consistency_max_day_frac * equity:
                    continue  # consistency rule not yet satisfied; keep trading
            return AccountOutcome("payout", traded_days, n_trades, equity, best_day)

        if traded_days >= rules.max_trading_days:
            return AccountOutcome("timeout", traded_days, n_trades, equity, best_day)

    return AccountOutcome("timeout", traded_days, n_trades, equity, best_day)


def _floor_from_hwm(hwm: float, rules: PropCfg) -> float:
    floor = hwm - rules.buffer
    if rules.trailing_caps_at_start:
        floor = min(floor, 0.0)
    return floor


def run_prop_mc(tdf: pd.DataFrame, rules: PropCfg, contracts: int | None = None,
                n_paths: int | None = None) -> dict:
    """Monte Carlo over resampled day sequences. Returns the headline probabilities."""
    days = days_from_trades(tdf)
    if not days:
        return {"error": "no trades"}
    n_paths = n_paths or rules.n_paths
    outcomes: list[AccountOutcome] = []
    for seq in resample_day_paths(days, n_paths, rules.max_trading_days, rules.resample, rules.seed):
        outcomes.append(simulate_account(seq, rules, contracts))
    payouts = [o for o in outcomes if o.status == "payout"]
    fails = [o for o in outcomes if o.status == "fail"]
    p_payout = len(payouts) / len(outcomes)
    p_fail = len(fails) / len(outcomes)
    payout_cash = rules.payout_threshold * rules.profit_split
    ev_single = p_payout * payout_cash - rules.activation_cost

    # sequential restarts: keep buying accounts until payout or attempts exhausted
    p_no = 1.0 - p_payout
    ev_game = 0.0
    for attempt in range(1, rules.max_attempts + 1):
        ev_game += (p_no ** (attempt - 1)) * (p_payout * payout_cash - rules.activation_cost)
    return {
        "paths": len(outcomes),
        "p_payout": round(p_payout, 4),
        "p_fail": round(p_fail, 4),
        "p_timeout": round(1 - p_payout - p_fail, 4),
        "median_days_to_payout": float(np.median([o.traded_days for o in payouts])) if payouts else None,
        "median_trades_to_payout": float(np.median([o.trades for o in payouts])) if payouts else None,
        "median_days_to_fail": float(np.median([o.traded_days for o in fails])) if fails else None,
        "ev_single_attempt_usd": round(ev_single, 2),
        "ev_with_restarts_usd": round(ev_game, 2),
        "payout_cash_usd": round(payout_cash, 2),
        "contracts": contracts if contracts is not None else rules.contracts,
    }


def sensitivity_tables(tdf: pd.DataFrame, rules: PropCfg) -> dict[str, pd.DataFrame]:
    """Sensitivity of the game EV to cost, payout size and risk (contracts)."""
    import copy

    out: dict[str, pd.DataFrame] = {}
    rows = []
    for cost in rules.sens_costs:
        r = copy.deepcopy(rules)
        r.activation_cost = cost
        res = run_prop_mc(tdf, r)
        rows.append({"activation_cost": cost, **{k: res.get(k) for k in
                     ("p_payout", "p_fail", "ev_single_attempt_usd", "ev_with_restarts_usd")}})
    out["cost"] = pd.DataFrame(rows)

    rows = []
    for pay in rules.sens_payouts:
        r = copy.deepcopy(rules)
        r.payout_threshold = pay
        res = run_prop_mc(tdf, r)
        rows.append({"payout_threshold": pay, **{k: res.get(k) for k in
                     ("p_payout", "p_fail", "median_days_to_payout", "ev_single_attempt_usd")}})
    out["payout"] = pd.DataFrame(rows)

    rows = []
    for k in rules.sens_contracts:
        res = run_prop_mc(tdf, rules, contracts=k)
        rows.append({"contracts": k, **{r_: res.get(r_) for r_ in
                     ("p_payout", "p_fail", "median_days_to_payout", "ev_single_attempt_usd")}})
    out["contracts"] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Rescue mode
# ---------------------------------------------------------------------------

def _baseline_day(rng: np.random.Generator, rc: RescueCfg) -> np.ndarray:
    n = rc.baseline_trades_per_day
    wins = rng.random(n) < rc.baseline_wr
    return np.where(wins, rc.baseline_avg_win, -rc.baseline_avg_loss).astype(float)


def run_rescue_mc(tdf: pd.DataFrame, rules: PropCfg, rc: RescueCfg,
                  n_days_total: int, n_paths: int = 3000) -> pd.DataFrame:
    """For each drawdown state, compare three policies from that state onward:
       continue_baseline | switch_to_rescue | stop_trading.

    The rescue stream only offers a trade on days where the strategy actually had a
    setup historically (frequency matters: waiting costs evaluation days)."""
    days = days_from_trades(tdf)
    if not days:
        return pd.DataFrame()
    p_flat = flat_day_rate(tdf, n_days_total)
    rng = np.random.default_rng(rules.seed + 1)
    rows = []
    for dd_frac in rc.dd_states:
        start_equity = -dd_frac * rules.buffer
        for policy in ("continue_baseline", "switch_to_rescue", "stop_trading"):
            n_pay = n_fail = 0
            days_to_payout = []
            for _ in range(n_paths):
                equity = start_equity
                hwm = 0.0  # account already saw its high water at 0
                floor = _floor_from_hwm(hwm, rules)
                traded = 0
                status = "timeout"
                while traded < rules.max_trading_days:
                    traded += 1
                    if policy == "stop_trading":
                        break
                    if policy == "continue_baseline":
                        day = _baseline_day(rng, rc)
                    else:  # switch_to_rescue
                        if rng.random() < p_flat:
                            continue  # no valid setup today; account state is NOT a signal
                        day = days[rng.integers(0, len(days))] * rules.contracts
                    day_pnl = 0.0
                    dead = False
                    for pnl in day:
                        day_pnl += pnl
                        equity += pnl
                        if rules.dd_type == "trailing_intraday":
                            hwm = max(hwm, equity)
                            floor = _floor_from_hwm(hwm, rules)
                        if equity <= floor + 1e-9:
                            dead = True
                            break
                        if rules.daily_loss_limit is not None and day_pnl <= -rules.daily_loss_limit:
                            break
                    if dead:
                        status = "fail"
                        break
                    if rules.dd_type in ("trailing_eod", "trailing_intraday"):
                        hwm = max(hwm, equity)
                        floor = _floor_from_hwm(hwm, rules)
                    if equity >= rules.payout_threshold and traded >= rules.min_trading_days:
                        status = "payout"
                        days_to_payout.append(traded)
                        break
                if status == "payout":
                    n_pay += 1
                elif status == "fail":
                    n_fail += 1
            rows.append({
                "dd_state": dd_frac,
                "policy": policy,
                "p_payout": round(n_pay / n_paths, 4),
                "p_fail": round(n_fail / n_paths, 4),
                "p_survive_no_payout": round(1 - (n_pay + n_fail) / n_paths, 4),
                "median_days_to_payout": float(np.median(days_to_payout)) if days_to_payout else None,
            })
    return pd.DataFrame(rows)
