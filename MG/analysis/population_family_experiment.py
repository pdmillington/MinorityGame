#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:43:46 2025

@author: petermillington

Plot Attendance Analysis Module
- Runs a generalised MG (dollar game to be checked) for various values of m and s.
- Outputs plots of attendance time series, attendance distribution, points distribution
and success rate distribution.
- Reduced logging capability 
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #ensure modules are accessible

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from core.game import Game
from payoffs.mg import BinaryMGPayoff,  ScaledMGPayoff, SmallMinorityPayoff, AssymetricMinorityPayoff, DollarGamePayoff
from utils.logger import RunLogger, log_simulation  # Moved logging utility to reusable utils
from typing import Optional, Dict, Any, List, Tuple

# Helper to calculate risk and return from wealth data.

def risk_from_wealth(wealth):
    daily_return = [wealth[1:]-wealth[:-1]]
    avg_return = np.mean(daily_return)
    risk = np.var(daily_return)
    return avg_return, risk

# Core simulation code

def run_games_collect_data(m_values, s_values, payoff_scheme, N=501,
                           rounds=50000, price_init=100, lambda_value=None,
                           market_maker=None, position_limit=None):
    results = {}
    for m in m_values:
        for s in s_values:
            game = Game(
                num_players=N,
                memory=m,
                num_strategies=s,
                rounds=rounds,
                payoff_scheme=payoff_scheme(),
                lambda_value=lambda_value,
                market_maker=market_maker,
                position_limit=position_limit
            )
            game.run()
            pairs = [risk_from_wealth(np.array(p.wealth_per_round)) for p in game.players]
            avg_ret, risk = map(np.array, zip(*pairs))
            key = (m, s)
            results[key] = {
                "actions": game.actions,
                "prices": game.prices,
                "player_returns": avg_ret,
                "player_risk": risk,
                "wealth": [p.wealth_per_round[-1] for p in game.players],
                "points": [p.points for p in game.players],
                "success_rates": [np.mean(p.wins_per_round) for p in game.players]
            }
    return results




def build_price_series(A: np.ndarray,
                       lam: float,
                       noise_std: float = 0.0,
                       seed: Optional[int] = None
                       ) -> np.ndarray:
    """Construct price series p given Δp(t+1)/p = lam*A(t) + ε_{t+1}. p[0]=100."""
    rng = np.random.default_rng(seed)
    T = A.size
    eps = rng.normal(0.0, noise_std, size=T) if noise_std > 0 else np.zeros(T)
    r = A / lam +eps
    p = np.zeros(T + 1)
    p[0] = 100
    p[1:] = p[0] * np.exp(np.cumsum(r))
    return p



# ------------------------------
# Plotting helpers
# ------------------------------

def plot_attendance_series(results,
                           m_values,
                           s_values,
                           payoff_scheme_name,
                           N,
                           rounds,
                           save_dir="plots/attendance",
                           filename_prefix=None
                           ):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values),
                              len(s_values),
                              figsize=(12, 9),
                              sharex=True,
                              sharey=True)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            ax.plot(results[(m, s)]["actions"], lw=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values) - 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Total Action A(t)")

    plt.suptitle("Attendance Time Series", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits =["attendance_grid", payoff_scheme_name, f"{N}_agents", f"{rounds}_rounds"]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_price_graph(results,
                     m_values,
                     s_values,
                     payoff_scheme_name,
                     N,
                     rounds,
                     lambda_value,
                     save_dir="plots/prices",
                     filename_prefix=None):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True)

    # Construct prices from the attendance data and plot for each m, s
    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            p = results[(m, s)]["prices"]
            ax = axes[i, j]
            ax.plot(p, lw=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values)-1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Total Action A(t)")

    plt.suptitle("Price(t)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits = ["price_time_series", payoff_scheme_name, f"{N}_agents", f"{rounds}_rounds"]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits)+".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_attendance_distribution(results,
                                 m_values,
                                 s_values,
                                 payoff_scheme_name,
                                 N,
                                 rounds,
                                 save_dir="plots/attendance_dist",
                                 filename_prefix=None):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            ax.hist(results[(m, s)]["actions"], bins=30, alpha=0.75, edgecolor='black')
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values) - 1:
                ax.set_xlabel("Total Action A(t)")
            if j == 0:
                ax.set_ylabel("Frequency")

    plt.suptitle("Attendance Frequency Distribution", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits = ["attendance_frequency", payoff_scheme_name, f"{N}_agents",f"{rounds}_rounds"]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_wealth_distribution(results,
                             m_values,
                             s_values,
                             payoff_scheme_name,
                             N,
                             rounds,
                             save_dir="plots/wealth_dist",
                             filename_prefix=None):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            ax.hist(results[(m, s)]["wealth"], bins=30, alpha=0.75, edgecolor='black')
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values) - 1:
                ax.set_xlabel("Agent Terminal Wealth")
            if j == 0:
                ax.set_ylabel("Frequency")

    plt.suptitle("Agent Wealth Distribution", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits = ["wealth", payoff_scheme_name, f"{N}_agents",f"{rounds}_rounds"]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_point_distributions(results,
                             m_values,
                             s_values,
                             payoff_scheme_name,
                             N,
                             rounds,
                             save_dir="plots/points",
                             filename_prefix=None):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            ax.hist(results[(m, s)]["points"], bins=30, alpha=0.75, edgecolor='black')
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values) - 1:
                ax.set_xlabel("Final Points")
            if j == 0:
                ax.set_ylabel("Frequency")

    plt.suptitle("Point Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits = ["points_grid", payoff_scheme_name]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_success_rate_distributions(results,
                                    m_values,
                                    s_values,
                                    payoff_scheme_name,
                                    N,
                                    rounds,
                                    save_dir="plots/success",
                                    filename_prefix=None):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            success_rates = results[(m, s)]["success_rates"]
            ax.hist(success_rates, bins=40, range=(0.2, 0.8), edgecolor="black", alpha=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values) - 1:
                ax.set_xlabel("Success Rate")
            if j == 0:
                ax.set_ylabel("Frequency")

    plt.suptitle("Success Rate Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits= ["success_rate_grid", payoff_scheme_name]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_risk_return(results,
                     m_values,
                     s_values,
                     payoff_scheme_name,
                     N,
                     rounds,
                     save_dir="plots/risk_return",
                     filename_prefix=None):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=False)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            risk = results[(m, s)]["player_risk"]
            av_ret = results[(m, s)]["player_returns"]
            ax.scatter(risk, av_ret, s=18, marker='o', alpha=0.6, linewidths=0)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values) - 1:
                ax.set_xlabel("Risk")
            if j == 0:
                ax.set_ylabel("Return")

    plt.suptitle("Risk Return distribution for players", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits= ["risk_return", payoff_scheme_name]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

# -----------------------------------------
# Metrics helper for logging
# -----------------------------------------

def summarize_cell(cell):
    """Compute compact metrics for a (m,s) cell: mean/std of success, var(A), mean points."""
    sr = np.array(cell['success_rates'], dtype=float)
    A = np.array(cell['actions'], dtype=float)
    pts = np.array(cell['points'], dtype=float)
    return{
        "mean_success": float(np.mean(sr)) if sr.size else np.nan,
        "std_success": float(np.std(sr)) if sr.size else np.nan,
        "var_attendance": float(np.var(A)) if A.size else np.nan,
        "mean_points": float(np.mean(pts)) if pts.size else np.nan,
        }

# --------------------------------------
# Main
# --------------------------------------



def main():    

    m_values = [5, 8, 12]
    s_values = [3, 5, 7]
    N=501
    rounds=5000
    payoff = BinaryMGPayoff # ScaledMGPayoff or BinaryMGPayoff
    seed = 12345
    lambda_value = N * 30
    market_maker = None
    position_limit = None

    # --- logger with seed (creates timestamped run dir) ---
    logger = RunLogger(module="Grid_m_s",
                       payoff=payoff.__name__,
                       run_id=f"N{N}_T{rounds}",
                       seed=seed)
    logger.log_params({
        "m_values": m_values,
        "s_values": s_values,
        "N": N,
        "rounds": rounds,
        "payoff": payoff.__name__,
        "market_maker": market_maker,
        "position_limit": position_limit,
        "seed": seed,
        })

    # --- run simulations ---
    results = run_games_collect_data(m_values, s_values, payoff_scheme=payoff, N=N, rounds=rounds,
                                     lambda_value=lambda_value, market_maker=market_maker,
                                     position_limit=position_limit)

    # --- save plots ---
    prefix = logger.run_info["start_time"]
    p1 = plot_attendance_series(results,
                                m_values,
                                s_values,
                                payoff_scheme_name=payoff.__name__,
                                N=N,
                                rounds=rounds,
                                filename_prefix=prefix)
    p2 = plot_point_distributions(results,
                                  m_values,
                                  s_values,
                                  payoff_scheme_name=payoff.__name__,
                                  N=N,
                                  rounds=rounds,
                                  filename_prefix=prefix)
    p3 = plot_success_rate_distributions(results,
                                         m_values,
                                         s_values,
                                         payoff_scheme_name=payoff.__name__,
                                         N=N,
                                         rounds=rounds,
                                         filename_prefix=prefix)
    p4 = plot_attendance_distribution(results,
                                      m_values,
                                      s_values,
                                      payoff_scheme_name=payoff.__name__,
                                      N=N,
                                      rounds=rounds,
                                      filename_prefix=prefix)
    p5 = plot_price_graph(results,
                          m_values,
                          s_values,
                          payoff_scheme_name=payoff.__name__,
                          N=N,
                          rounds=rounds,
                          lambda_value=lambda_value,
                          filename_prefix=prefix)

    p6 = plot_risk_return(results,
                          m_values,
                          s_values,
                          payoff_scheme_name=payoff.__name__,
                          N=N,
                          rounds=rounds,
                          filename_prefix=prefix)

    p7 = plot_wealth_distribution(results,
                                  m_values,
                                  s_values,
                                  payoff_scheme_name=payoff.__name__,
                                  N=N,
                                  rounds=rounds,
                                  filename_prefix=prefix)

    # Register artifacts in the run folder (will copy in plots/ too)
    logger.log_artifact(p1)
    logger.log_artifact(p2)
    logger.log_artifact(p3)
    logger.log_artifact(p4)
    logger.log_artifact(p5)
    logger.log_artifact(p6)
    logger.log_artifact(p7)

    # --- log per-cell metrics into metrics.csv (one row per (m,s)) ---
    step = 0
    for m in m_values:
        for s in s_values:
            metrics = summarize_cell(results[(m, s)])
            metrics.update({"m": m, "s": s})
            logger.log_metrics(metrics, step=step)
            step += 1

    # --- optional: save a JSON dict with all compact summaries (handy for quick inspection) ---
    compact = {(m, s): summarize_cell(results[(m, s)]) for m in m_values for s in s_values}
    # Convert tuple keys to strings for JSON
    compact_json = {f"m{m}_s{s}": v for (m, s), v in compact.items()}
    logger.save_dict("cell_summaries", compact_json)

    # --- human-readable rolling log entry ---
    log_simulation([
        f"Module: Grid_m_s",
        f"Payoff: {payoff.__name__}",
        f"m_values: {m_values}",
        f"s_values: {s_values}",
        f"N: {N}",
        f"Rounds: {rounds}",
        f"Seed: {seed}",
        f"Saved plots: {p1}, {p2}, {p3}, {p4}, {p5}",
        f"Run dir: {logger.get_dir()}",
    ])

    logger.close()

if __name__ == "__main__":
    main()