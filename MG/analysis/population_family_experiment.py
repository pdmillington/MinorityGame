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

def stat_summary(prices, periods_per_year=252):
    """
    Calculate statistical measures from a price series.
    
    Parameters:
    -----------
    prices : array-like
        Time series of prices
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252 for daily data)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - avg_ret: Mean log return (annualized)
        - vol: Volatility/standard deviation (annualized)
        - skew_ret: Skewness of log returns
        - kurt_ret: Excess kurtosis of log returns
        - n_obs: Number of return observations
    """
    # Input validation
    prices = np.asarray(prices)

    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations")

    if np.any(prices <= 0):
        raise ValueError("Prices must be positive for log returns")

    # Calculate log returns
    returns = np.diff(np.log(prices))

    # Handle NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("No valid returns after removing NaN values")

    # Calculate statistics (annualized)
    stats = {
        'avg_ret': np.mean(returns) * periods_per_year,  # Annualized mean return
        'vol': np.std(returns, ddof=1) * np.sqrt(periods_per_year),  # Annualized volatility
        'skew_ret': skew(returns),  # Skewness (no annualization)
        'kurt_ret': kurtosis(returns, fisher=True),  # Excess kurtosis (no annualization)
        'n_obs': len(returns)
    }

    return stats

# ------------------------------
# Plotting helpers
# ------------------------------

def plot_attendance_series(results,
                           m_values,
                           s_values,
                           payoff_scheme_name,
                           N,
                           market_maker,
                           position_limit,
                           ):
    #Create 2D axes array
    fig, axes = plt.subplots(len(m_values),
                              len(s_values),
                              figsize=(12, 9),
                              sharex=True,
                              sharey=True,
                              squeeze=False)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            actions = results[(m, s)]["actions"]
            actions_av = np.mean(actions)
            actions_std = np.std(actions)
            ax.plot(results[(m, s)]["actions"], lw=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            stats_text = (
                f"μ: {actions_av:.2f}\n"
                f"σ: {actions_std:.2f}"
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
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Total Action A(t)")

    fig.suptitle(
        f"Attendance(t): {payoff_scheme_name}, N={N}, "
        f"MM={market_maker}, Limit={position_limit}",
        y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def plot_price_graph(results,
                     m_values,
                     s_values,
                     payoff_scheme_name,
                     N,
                     market_maker,
                     position_limit
                     ):
    """
    Build a grid of price time series plots and return the Matplotlib Figure.
    This function performs no filesystem I/O (no save).
    """
    # Always create a 2D array
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=False,
                             squeeze=False)

    # Extract prices from dictionary and plot for each m, s
    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            p = results[(m, s)]["prices"]
            ax = axes[i, j]

            # Draw price graphs
            ax.plot(p, lw=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)

            # Compute stats
            try:
                stats = stat_summary(p)  # expects keys: avg_ret, vol, skew_ret, kurt_ret
                if all(k in stats for k in ("avg_ret", "vol", "skew_ret", "kurt_ret")):
                    stats_text = (
                        f"μ: {stats['avg_ret']*100:.2f}%\n"
                        f"σ: {stats['vol']*100:.1f}%\n"
                        f"skew: {stats['skew_ret']:.2f}\n"
                        f"kurt: {stats['kurt_ret']:.2f}"
                    )
                    ax.text(
                        0.02, 0.98, stats_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        va='top',
                        bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='white',
                                 edgecolor='gray',
                                 alpha=0.8,
                                 linewidth=0.5)
                    )
            except Exception:
                # Keep plotting even if stats fail
                pass

            if i == len(m_values)-1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Price")

    fig.suptitle(
        f"Price(t): {payoff_scheme_name}, N={N}, "
        f"MM={market_maker}, Limit={position_limit}",
        y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig

def plot_returns_scatter(results,
                         m_values,
                         s_values,
                         payoff_scheme_name,
                         N,
                         market_maker,
                         position_limit
                         ):
    """
    Plot returns(t+1) vs returns(t) generated by the game's price series and
    calculates autocorrelation.
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
            prices = results[(m, s)]["prices"]
            returns = returns_from_prices(prices)
            auto_corr = autocorr_lag(returns)
            ax.scatter(returns[1:], returns[:-1], alpha=0.6)
          
            ax.set_title(f"r(t+1) vs r(t); ρ₁ = {auto_corr:.3f}")
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values) - 1:
                ax.set_xlabel("r(t+1) (basis points)")
            if j == 0:
                ax.set_ylabel("r(t) (basis points)")
    to_bps = lambda x, pos: f"{x*1e4:.0f} bp"  # 1 bp = 0.0001 = 0.01%
    for a in axes.ravel():
        a.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        a.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        a.xaxis.set_major_formatter(FuncFormatter(to_bps))
        a.yaxis.set_major_formatter(FuncFormatter(to_bps))
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.grid(True, which='major', linestyle='--', alpha=0.6)
        a.grid(True, which='minor', linestyle=':',  alpha=0.3)


    fig.suptitle(f"Return Autocorrelation: {payoff_scheme_name}, N={N}, "
                 f"MM={market_maker}, Limit={position_limit}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def plot_attendance_distribution(results,
                                 m_values,
                                 s_values,
                                 payoff_scheme_name,
                                 N,
                                 market_maker,
                                 position_limit,
                                 ):
    """
    Build a grid of attendance distribution plots and return the Matplotlib Figure.
    This function performs no filesystem I/O (no save).
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
            actions = results[(m, s)]["actions"]
            actions_av = np.mean(actions)
            actions_std = np.std(actions)
            ax.hist(actions, bins=30, alpha=0.75, edgecolor='black')
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            stats_text = (
                f"μ: {actions_av:.2f}\n"
                f"σ: {actions_std:.2f}"
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
                ax.set_xlabel("Total Action A(t)")
            if j == 0:
                ax.set_ylabel("Frequency")

    fig.suptitle(f"Attendance Frequency Distribution: {payoff_scheme_name}, "
                 f"N={N}, MM={market_maker}, Limit={position_limit}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def plot_wealth_distribution(results,
                             m_values,
                             s_values,
                             payoff_scheme_name,
                             N,
                             market_maker,
                             position_limit,
                             ):
    """
    Produces a plot of the distrubition of wealth among the agents and returns the
    matplotlib figure.  This function does not save the figure.
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
            w = results[(m, s)]["wealth"]
            w_av = np.mean(w)
            w_std = np.std(w)
            ax.hist(results[(m, s)]["wealth"], bins=30, alpha=0.75, edgecolor='black')
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            stats_text = (
                f"μ: {w_av:.1f}\n"
                f"σ: {w_std:.1f}"
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
                ax.set_xlabel("Agent Terminal Wealth")
            if j == 0:
                ax.set_ylabel("Frequency")

    fig.suptitle(f"Agent Wealth Distribution: {payoff_scheme_name}, N={N},"
                 f" MM={market_maker}, Limit={position_limit}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

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

def plot_correlation_points_wealth(results,
                     m_values,
                     s_values,
                     payoff_scheme_name,
                     N,
                     market_maker,
                     position_limit,
                     ):
    """
    Matplotlib scatter showing correlation between points and wealth for agents.
    """
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True,
                             squeeze=False)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            w = np.array(results[(m, s)]["wealth"])
            p = np.array(results[(m, s)]["points"])
            ax=axes[i,j]
            ac1 = float(np.corrcoef(p, w)[0, 1])
            ax.scatter(p, w, s=6, alpha=0.6)
            ax.set_title(f"m={m}, s={s}; ρ = {ac1:.3f}")
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values)-1:
                ax.set_xlabel("wealth")
            if j == 0:
                ax.set_ylabel("points")

    fig.suptitle(f"Points vs Wealth: {payoff_scheme_name}, N={N}, "
                 f"MM={market_maker}, Limit={position_limit}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def plot_correlation_switches_wealth(results,
                     m_values,
                     s_values,
                     payoff_scheme_name,
                     N,
                     market_maker,
                     position_limit,
                     ):
    """
    Generates scatter plot of number of switches and terminal wealth of the agents.
    """
    fig, axes = plt.subplots(len(m_values),
                             len(s_values),
                             figsize=(12, 9),
                             sharex=True,
                             sharey=True,
                             squeeze=False)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            w = np.array(results[(m, s)]["wealth"])
            sw = np.array(results[(m, s)]["strategy_switches"])
            ax=axes[i,j]
            ac2 = float(np.corrcoef(sw, w)[0, 1])
            ax.scatter(sw, w, s=6, alpha=0.6)
            ax.set_title(f"m={m}, s={s}; ρ = {ac2:.3f}")
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values)-1:
                ax.set_xlabel("Wealth")
            if j == 0:
                ax.set_ylabel("Strategy Switches")

    fig.suptitle(f"Strat switches vs Wealth: {payoff_scheme_name}, N={N}, "
                 f"MM={market_maker}, Limit={position_limit}")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    return fig

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
        "stub": "attendance",
    },
    "price_series": {
        "extract": lambda res: res["price_series"],
        "extract2": None,
        "plot_fn": plot_series,
        "y_label": "Price",
        "stub": "price",
    },
    "A_series_attend": {
        "extract": lambda res: res["A_series"],
        "extract2": None,
        "plot_fn": plot_hist,          # your callback
        "y_label": "Frequency",
        "stub": "attendance_hist",
    },
    "wealth_points":{
        "extract": lambda res: res["final_wealth"],
        "extract2": lambda res: res["final_wins"],
        "plot_fn": plot_scatter,
        "y_label": "Wealth",
        "stub": "wlth_v_wins",
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
        titles = [f"{fam_cfg.vary} = {v}" for v, _ in results]
        
        p = plot_population_grid(items=items,
                              plot_fn = spec["plot_fn"],
                              suptitle=f'{metric_name} by population variant', 
                              save_dir=plots_dir,
                              x_label="Round",
                              y_label=spec["y_label"],
                              filename_prefix=prefix,
                              filename_stub=spec["stub"],
                              items2=items2)
        paths.append(p)
    print(paths)

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
    