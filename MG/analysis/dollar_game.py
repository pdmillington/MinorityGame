#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:38:04 2025

@author: petermillington

Dollar Game Analysis Module

What it does (single-file):
- Runs a Dollar Game simulation (trend-following by default) or analyzes an existing one
- Computes diagnostics: autocorrelation of A, variance of A, per-player $ (cumulative and per-round)
- Builds a synthetic price series with Δp(t+1) = λ·A(t) + ε(t+1)
* λ defaults to Var(A), or set manually
* ε is optional Gaussian noise
- Saves plots and tables; logs params/metrics/artifacts into a timestamped run folder

Assumptions:
- Your Game implements DollarGamePayoff with delayed credit, populating:
game.actions # list of A(t)
player.dollar (cumulative) and player.dollar_per_round (per-round), if available
- If player.dollar_per_round is missing, we derive per-round ≈ diff of cumulative with a leading 0.

Edit the import paths below to match your repo.
"""
import os
import sys
import json
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -----------------------------------------------------------------------------
# Adjust PYTHONPATH for project imports
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Project imports 
from core.game import Game
from payoffs.mg import DollarGamePayoff 
from utils.logger import RunLogger, log_simulation

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def autocorr_lag1(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def build_price_series(A: np.ndarray, lam: float, noise_std: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Construct price series p given Δp(t+1)/p = lam*A(t) + ε_{t+1}. p[0]=100."""
    rng = np.random.default_rng(seed)
    T = A.size
    eps = rng.normal(0.0, noise_std, size=T) if noise_std > 0 else np.zeros(T)
    r = A / lam +eps
    p = np.zeros(T + 1)
    p[0] = 100
    p[1:] = p[0] * np.exp(np.cumsum(r))
    return p

def moving_average(arr, interval):
    arr = np.asarray(arr, dtype=float)
    if interval < 0 or interval > arr.size:
        return np.array([])
    ret = np.cumsum(arr, dtype=float)
    ret[interval:] = ret[interval:]-ret[:-interval]
    return ret[interval-1:] / interval

def wealth_from_actions(players, price, W0=100.0):
    """
    Single-period holding, 1 unit:
      position at t = a_i(t); PnL_i(t) = a_i(t)*(p[t+1]-p[t]).
    Returns:
      final_wealth: (N,) final wealth at T
      wealth_paths: (N, T) wealth per round (optional analysis)
    """
    p = np.asarray(price, dtype=float)         # length T+1
    actions = np.vstack([p_i.actions for p_i in players]).astype(float)  # (N, T)
    dP = np.diff(p)                            # (T,)
    pnl = actions * dP                         # (N, T)
    wealth_paths = W0 + np.cumsum(pnl, axis=1) # (N, T)
    final_wealth = wealth_paths[:, -1]
    return final_wealth, wealth_paths


def summarize_dollars(players) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cum_dollars, per_round_dollars) arrays of shape (N,) and (N,T) where possible.
    If per-round not available, derive from cumulative by differencing.
    """
    N = len(players)
    cum = np.array([getattr(p, "dollar", 0.0) for p in players], dtype=float)
    # Try to gather per-round; pad uneven lengths with zeros
    lists = [getattr(p, "dollar_per_round", None) for p in players]
    if all(isinstance(lst, list) and len(lst) > 0 for lst in lists):
        T = max(len(lst) for lst in lists)
        per = np.zeros((N, T), dtype=float)
        for i, lst in enumerate(lists):
            per[i, :len(lst)] = np.array(lst, dtype=float)
            return cum, per
    # Derive per-round from cumulative + length of game.actions if available
    T_guess = max(len(getattr(p, "wins_per_round", [])) for p in players) or None
    if T_guess:
        per = np.zeros((N, T_guess), dtype=float)
        # rough: assume first reward is 0, then diff cumulative evenly where lengths mismatch
        # In most runs cumulative equals sum(per_round), so this is fine as a fallback
        # Without round-by-round, we keep zeros; user can recompute if needed
        return cum, per
    return cum, np.zeros((N, 0), dtype=float)

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_action_series(results, m_values, s_values, N, rounds, save_dir):
    
    fig, axes = plt.subplots(len(m_values), len(s_values), figsize=(12,9), sharex=True, sharey=True)
    
    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            ax.plot(results[(m,s)]["actions"], lw=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values)-1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Total Buyers-Sellers")
   
    plt.suptitle("Buyers-Sellers")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits = ["action_series", "dollar_game", f"{N}_agents", f"{rounds}_rounds"]
    filename = "_".join(bits)+".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path
     
def plot_dollar_distribution(results, m_values, s_values, N, rounds, save_dir):
    
    fig, axes = plt.subplots(len(m_values), len(s_values), figsize=(12,9), sharex=True, sharey=True)
    
    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            ax = axes[i, j]
            ax.hist(results[(m,s)]["dollars"], bins=40, edgecolor="black", alpha=0.8)
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values)-1:
                ax.set_xlabel("Wealth")
            if j == 0:
                ax.set_ylabel("Frequency")
   
    plt.suptitle("WDistribution of terminal wealth")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits = ["awealth", "dollar_game", f"{N}_agents", f"{rounds}_rounds"]
    filename = "_".join(bits)+".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_success_rates(results, m_values, s_values, N, rounds, lam, savedir, interval_lengths=(50, 100, 250)):
    
    fig, axes = plt.subplots(len(m_values), len(s_values), figsize=(12,9), sharex=True, sharey=True)
    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            for interval in interval_lengths:
                ax = axes[i,j]
                mov_avg = moving_average(results[(m,s)]["average_success_rates_per_round"], interval)
                if mov_avg.size == 0:
                    continue
                x = np.arange(interval, interval + mov_avg.size)
                label = f"{interval}-round MA" if i == len(m_values)-1 and j == len(s_values)-1 else None
                ax.plot(x, mov_avg, lw=0.8, label=label)
            
            ax.set_title(f"m={m}, s={s}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == len(m_values)-1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Avg mean wins per round")
            if i == len(m_values)-1 and j == len(s_values)-1:
                ax.legend(title="Window", loc="best")
                
    plt.suptitle("Mean wins per round (moving averages)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    bits = ["mean_wins", "dollar_game", f"{N}_agents", f"{rounds}_rounds"]
    filename = "_".join(bits)+".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path
     
def plot_price_graph(results, m_values, s_values, N, rounds, lam, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(m_values), len(s_values), figsize=(12, 9), sharex=True, sharey=True)
    
    # Construct prices from the attendance data and plot for each m, s
    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            p = build_price_series(np.array(results[(m, s)]["actions"]), lam)
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
    bits = ["price_time_series", "dollar_game", f"{N}_agents", f"{rounds}_rounds"]
    filename = "_".join(bits)+".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path
    

def plot_autocorrelation(results, m_values, s_values, N, rounds, save_dir):
   # TODO: update for m, s
   ac1 = 1.0
   plt.figure(figsize=(6, 6))
   plt.scatter(results[:-1], results[1:], s=6, alpha=0.6)
   plt.title(f"A(t) vs A(t+1); ρ₁ = {ac1:.3f}")
   plt.xlabel("A(t)")
   plt.ylabel("A(t+1)")
   plt.grid(True, linestyle="--", alpha=0.6)
   plt.tight_layout()
   p_ac = os.path.join(save_dir, "A_lag1_scatter.pdf")
   plt.savefig(p_ac, format="pdf", dpi=300)
   plt.close()


# -----------------------------------------------------------------------------
# Core run + analysis
# -----------------------------------------------------------------------------
def run_dollar_game_grid (
        m_values: list = [2,5],
        s_values: list = [2,5],
        N: int = 1000,
        rounds: int = 20000,
        sign: int = +1,
        seed: Optional[int] = 12345,
        lambda_mode: str = "manual",
        lambda_value: float = 1.0,
        price_noise_std:float = 0.0,
    ) -> Dict[str, Any]:
    results = {}
    
    """
    Runs a Dollar Game and returns a dict of metrics.
    """
   
    payoff = DollarGamePayoff(sign=sign)
    
   
    # ---- Run game ----
    for m in m_values:
        for s in s_values:
            game = Game(num_players=N, memory=m, num_strategies=s, rounds=rounds, payoff_scheme=payoff)
            game.run()
            key = (m,s)
            results[key] = {
                "actions": game.actions,
                "dollars": [p.dollar for p in game.players],
                "average_success_rates_per_round": [np.mean(np.vstack([p.wins_per_round for p in game.players]), axis=0)],
                "success_rates": [np.mean(p.wins_per_round) for p in game.players]
            }
    return results
            
    
   
    
    
    """
    
    return {
        "run_dir": logger.get_dir(),
        "A": A,
        "varA": varA,
        "autocorr_lag1": ac1,
        "lambda": lam,
        "price": p,
        "cum_dollars": cum_dol,
        "series_table": series_path,
        "dollars_table": dollars_path,
        "plots": {
            "attendance": p_att,
            "dollars": p_dol,
            "price": p_price,
            "A_lag1": p_ac,
        },
    }


 
"""
# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Adjust parameters
    m_values = [7, 9, 12]
    s_values = [2, 3, 4]
    N = 501
    rounds = 50000
    sign = +1
    seed = 12345
    lambda_mode = "manual"
    lambda_value = 50 * N
    price_noise_std = 0.0
    
    
        # --- logger with seed (creates timestamped run dir) ---
    logger = RunLogger(module="DollarGame", payoff=f"Dollar(sign={sign:+d})",
                           run_id=f"N{N}_T{rounds}_m{m_values}_s{s_values}", seed=seed)
    logger.log_params({
        "m_values": m_values, "s_values": s_values, "N": N, "rounds": rounds,
        "sign": sign, "lambda_mode": lambda_mode,
        "lambda_value": lambda_value, "price_noise_std": price_noise_std,
        "seed": seed,
        })
    
    out = run_dollar_game_grid(
        m_values=m_values, s_values=s_values, N=N, rounds=rounds,
        sign=sign, # trend-following $-game
        seed=seed,
        lambda_mode=lambda_mode,
        lambda_value=lambda_value,
        price_noise_std=price_noise_std,
    )
    
    run_dir = logger.get_dir()
    
    # ---- Plots ----
    save_dir = os.path.join(run_dir, "artifacts")
    _ensure_dir(save_dir)
    
    p1 = plot_action_series(out, m_values, s_values, N, rounds, save_dir)
    p2 = plot_price_graph(out, m_values, s_values, N, rounds, lambda_value, save_dir)
    p3 = plot_dollar_distribution(out, m_values, s_values, N, rounds, save_dir)
    p4 = plot_success_rates(out, m_values, s_values, N, rounds, lambda_value, save_dir)

    """    
    # Register metrics
    logger.log_metrics({
        "varA": varA,
        "autocorr_lag1_A": ac1,
        "lambda_used": lambda_value,
        "mean_dollar": float(np.mean(cum_dol)),
        "std_dollar": float(np.std(cum_dol)),
        "min_dollar": float(np.min(cum_dol)),
        "max_dollar": float(np.max(cum_dol)),
    })
    
    # Human-readable log
    log_simulation([
        "Module: DollarGame",
        f"m={m}, s={s}, N={N}, rounds={rounds}, sign={sign:+d}",
        f"Var(A)={varA:.4g}, rho1(A)={ac1:.3f}, lambda={lam:.4g}, noise_std={price_noise_std}",
        f"Saved tables: {series_path}, {dollars_path}",
        f"Artifacts: {p_att}, {p_dol}, {p_price}, {p_ac}",
        f"Run dir: {logger.get_dir()}",
    ])
    
    logger.close()
    
    """