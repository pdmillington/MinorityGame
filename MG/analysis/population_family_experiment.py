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
sys.path.append(os.path.abspath(os.path.join
                                (os.path.dirname(__file__), '..')))  #ensure modules are accessible
from dataclasses import dataclass
import json
import argparse

import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter, AutoMinorLocator
from datetime import datetime
from core.game import Game
from core.game import Player
from core.game_config import GameConfig
from analysis.population_spec import build_population_variant, CohortConfig, PopulationConfig, PopulationFamilyConfig
from analysis.plot_utils import plot_population_grid, plot_series, plot_hist, plot_scatter
from utils.logger import RunLogger, log_simulation  # Moved logging utility to reusable utils
from typing import Optional, Iterable, Dict, Any, List, Tuple

def load_config(path:str) -> PopulationFamilyConfig:
    with open(path, "r") as f:
        data = json.load(f)
        
    fam_cfg = PopulationFamilyConfig(**data)
    game_cfg = GameConfig(
        rounds=fam_cfg.rounds,
        lambda_=fam_cfg.lambda_,
        mm=fam_cfg.market_maker,
        price=fam_cfg.price,
        record_agent_series=fam_cfg.record_agent_series,
        )
    return fam_cfg, game_cfg


# Helper to calculate risk and return from wealth data.
def returns_from_prices(prices):
    """
    Return vector from a given vector of prices
    """
    prices = np.asarray(prices)

    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations")

    if np.any(prices <= 0):
        raise ValueError("Prices must be positive for log returns")

    returns = np.diff(np.log(prices))

    returns = returns[~np.isnan(returns)]
    return returns

def autocorr_lag(r: np.ndarray, lag: int = 1) -> float:
    """
    Generates scalar autocorrelation value for a given lag
    """
    if r.size <= lag:
        return np.nan
    a = r[lag:]
    b = r[:-lag]
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt((a0 @ a0) * (b0 @ b0))
    return float((a0 @ b0) / denom) if denom > 0 else np.nan

def risk_from_wealth(wealth):
    """
    Calculates average return and risk for a wealth series
    """
    daily_change = np.diff(wealth)
    avg_return = np.mean(daily_change)
    risk = np.var(daily_change)
    return avg_return, risk

def plot_point_distributions(results,
                             m_values,
                             s_values,
                             payoff_scheme_name,
                             N,
                             market_maker,
                             position_limit,
                             ):
    """
    Produces a matplotlib figure of the point distributions for players in the games.
    """
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True,
                             squeeze=False)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            points = results[(m, s)]["points"]
            points_av = np.mean(points)
            points_std = np.std(points)
            ax.hist(points, bins=30, alpha=0.75, edgecolor='black')
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            stats_text = (
                f"μ: {points_av:.1f}\n"
                f"σ: {points_std:.1f}"
            )
            # position stat summary in up left with small font
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=7,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.8,
                            linewidth=0.5))
            if i == len(m_values) - 1:
                ax.set_xlabel("Final Points")
            if j == 0:
                ax.set_ylabel("Frequency")

    fig.suptitle(f"Point Distributions: {payoff_scheme_name}, N={N},"
                 f" MM={market_maker}, Limit={position_limit}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def plot_success_rate_distributions(results,
                                    m_values,
                                    s_values,
                                    payoff_scheme_name,
                                    N,
                                    market_maker,
                                    position_limit,
                                    ):
    """
    Produces a matplotlib figure of the success rate distributions for agents.
    """
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True,
                             squeeze=False)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            success_rates = results[(m, s)]["success_rates"]
            success_av = np.mean(success_rates)
            success_std = np.std(success_rates)
            ax.hist(success_rates, bins=60, range=(0.35, 0.65), edgecolor="black", alpha=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            stats_text = (
                f"μ: {success_av:.2f}\n"
                f"σ: {success_std:.2f}"
            )
            # position stat summary in up left with small font
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=7,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.8,
                            linewidth=0.5))
            if i == len(m_values) - 1:
                ax.set_xlabel("Success Rate")
            if j == 0:
                ax.set_ylabel("Frequency")

    fig.suptitle(f"Success Rate Distributions: {payoff_scheme_name}, N={N}, "
                 f"MM={market_maker}, Limit={position_limit}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def plot_risk_return(results,
                     m_values,
                     s_values,
                     payoff_scheme_name,
                     N,
                     market_maker,
                     position_limit
                     ):
    """
    Matplotlib scatter of risk and average return for the agents of each game.
    """
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=False,
                             sharey=False,
                             squeeze=False)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            risk = results[(m, s)]["player_risk"]
            av_ret = results[(m, s)]["player_returns"]
            risk_av = np.mean(risk)
            returns_av = np.mean(av_ret)
            ax.scatter(risk, av_ret, s=18, marker='o', alpha=0.6, linewidths=0)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            stats_text = (
                f"μ: {returns_av:.2f}\n"
                f"σ²: {risk_av:.2f}"
                )
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=7,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.8,
                            linewidth=0.5))
            
            if i == len(m_values) - 1:
                ax.set_xlabel("Risk Var σ²")
            if j == 0:
                ax.set_ylabel("Mean daily return μ")

    for a in axes.ravel():
        a.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        a.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    fig.suptitle(f"Player Risk Return: {payoff_scheme_name}, N={N}, "
                 f"MM={market_maker}, Limit={position_limit}", fontsize=14)
    fig.tight_layout(rect=[0.04, 0.04, 1, 0.98])

    return fig


# Metrics helper for logging

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

# Main

def run_population_family(family_cfg, game_cfg, Player):
    """
    Runs games for various cohorts.  A dictionary captures the output for these 
    games. 
    """
    populations = [build_population_variant(family_cfg, v) for v in family_cfg.values]
    results = []
    
    for v, pop_spec in zip(family_cfg.values, populations):
        game = Game(
            population_spec=pop_spec,
            cfg=game_cfg,
            PlayerClass=Player
        )
        res = game.run()
        results.append((v, res))
        
    return results

def plot_family_grid(fam_cfg: PopulationFamilyConfig, game_cfg: GameConfig):
    
    

    # logger with seed (creates timestamped run dir)
    logger = RunLogger(module="PopulationFamily", run_id=None, seed=fam_cfg.seed)
    logger.log_config(fam_cfg)

    # run simulation
    results = run_population_family(fam_cfg, game_cfg, Player)

    plots_dir = logger.subdir("plots")  # single, per-run home for all figures
    prefix = logger.run_info["start_time"]
    
    
    metric_specs = {
    "A_series": {
        "extract": lambda res: res["A_series"],
        "extract2": None,
        "plot_fn": plot_series,          # your callback
        "y_label": "Attendance A_t",
        "x_label": "Round",
        "stat_sum": True,
        "stub": "attendance",
    },
    "price_series": {
        "extract": lambda res: res["price_series"],
        "extract2": None,
        "plot_fn": plot_series,
        "y_label": "Price",
        "x_label": "Time/Round",
        "stat_sum": True,
        "stub": "price",
    },
    "A_series_attend": {
        "extract": lambda res: res["A_series"],
        "extract2": None,
        "plot_fn": plot_hist,          # your callback
        "y_label": "Frequency",
        "x_label": "Attendance",
        "stat_sum": False,
        "stub": "attendance_hist",
    },
    "succes_rates": {
        "extract": lambda res: res["final_wins"]/game_cfg.rounds,
        "extract2": None,
        "plot_fn": plot_hist,
        "y_label": "Frequency",
        "x_label": "Success Rate",
        "stat_sum": False,
        "stub": "success_hist",
    },
    "wealth_wins":{
        "extract": lambda res: res["final_wealth"],
        "extract2": lambda res: res["final_wins"]/game_cfg.rounds,
        "plot_fn": plot_scatter,
        "x_label": "Wealth",
        "y_label": "Win Frequency",
        "stat_sum": False,
        "stub": "wealth_v_wins",
    },
    "wealth_switches":{
        "extract": lambda res: res["final_wealth"],
        "extract2": lambda res: res["strategy_switches"],
        "plot_fn": plot_scatter,
        "y_label": "Num strat switches",
        "x_label": "Wealth",
        "stat_sum": False,
        "stub": "wealth_v_switches",
    },
    "returns_autocorr":{
        "extract": lambda res: returns_from_prices(res["price_series"])[1:],
        "extract2": lambda res: returns_from_prices(res["price_series"])[:-1],
        "plot_fn": plot_scatter,
        "y_label": "r(t+1)",
        "x_label": "r(t)",
        "stat_sum": False,
        "stub": "returns_autocorr",
    },
    # add more metrics here later
}

    for metric_name, spec in metric_specs.items():
        
        paths = []
        
        items = [spec["extract"](res) for _, res in results]
        if spec["extract2"] is not None:
            items2 = [spec["extract2"](res) for _, res in results]
        else:
            items2 = None
        
        suptitle = suptitle = f"{metric_name} by {fam_cfg.target_payoff} {fam_cfg.vary}"
        
        if metric_name == "returns_autocorr":
            # per-panel autocorr
            rhos = [
                autocorr_lag(returns_from_prices(res["price_series"]))
                for _, res in results
            ]
            titles = [
                f"{fam_cfg.vary} = {v}, ρ₁ = {rho:.3f}"
                for (v, _), rho in zip(results, rhos)
            ]

        else:
            titles = [f"{fam_cfg.target_payoff} {fam_cfg.vary} = {v}" for v, _ in results]

        
        p = plot_population_grid(items=items,
                              plot_fn = spec["plot_fn"],
                              titles = titles,
                              suptitle=suptitle, 
                              save_dir=plots_dir,
                              x_label=spec["x_label"],
                              y_label=spec["y_label"],
                              stat_sum=spec["stat_sum"],
                              filename_prefix=prefix,
                              filename_stub=spec["stub"],
                              items2=items2)
        paths.append(p)

    """
   


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
        f"Market Maker: {market_maker}"
        f"Position Limit: {position_limit}"
        f"Seed: {seed}",
        f"Saved plots: {p1}, {p2}, {p3}, {p4}, {p5}, {p6}, {p7}, {p8}, {p9}",
        f"Run dir: {logger.get_dir()}",
    ])

    """
    logger.close()

def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument(
                        "--config",
                        type=str,
                        required=True,
                        help="Path to JSON config file for a family of populations",
                        )
    args = parser.parse_args()
    
    fam_cfg, game_cfg = load_config(args.config)
    plot_family_grid(fam_cfg, game_cfg)

if __name__ == "__main__":
    main()
    