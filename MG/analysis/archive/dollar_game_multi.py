#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:37:08 2025

@author: petermillington

Dollar Game — Multi-Run Analysis (20 runs by default)

What this script does:
- Runs the Dollar Game R times (default R=20) with different seeds
- Uses a log-return price mapping: ln p(t+1) - ln p(t) = lam * A(t) + eps_t
  * lam defaults to 1 / (N * 50), optionally scaled by lambda_scale
  * eps_t ~ N(0, noise_std^2)
- Produces a plot of the average price path with ±1 SD envelope across runs
- Extracts per-run stats on **log returns**: mean, variance, skew, kurtosis,
  lag-1 autocorr of returns, and lag-1 autocorr of squared returns (vol clustering)
- Logs params, metrics, tables, and artifacts using your RunLogger + log_simulation

Edit project import paths as needed.
"""
import os
import sys
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Adjust PYTHONPATH for project imports (edit as needed)
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from core.game import Game
from payoffs.mg import DollarGamePayoff
from utils.logger import RunLogger, log_simulation
from concurrent.futures import ProcessPoolExecutor

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_price_series_log(A: np.ndarray, lam: float, noise_std: float = 0.0, seed: Optional[int] = None,
                           p0: float = 100.0) -> np.ndarray:
    """Price with log-returns r_t = lam*A_t + eps_t; p_{t+1} = p_t * exp(r_t). Returns length T+1."""
    rng = np.random.default_rng(seed)
    T = A.size
    eps = rng.normal(0.0, noise_std, size=T) if noise_std > 0 else np.zeros(T)
    r = lam * A + eps
    p = np.zeros(T + 1, dtype=float)
    p[0] = float(p0)
    p[1:] = p[0] * np.exp(np.cumsum(r))
    return p


def acf_lag1(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def stats_on_returns(r: np.ndarray) -> Dict[str, float]:
    """Compute stats on a 1D array of returns r."""
    if r.size == 0:
        return {k: np.nan for k in ["mean", "var", "skew", "kurt_excess", "acf1", "acf1_sq"]}
    m = float(np.mean(r))
    v = float(np.var(r))
    s = float(np.mean(((r - m) / (np.sqrt(v) + 1e-12))**3)) if v > 0 else np.nan
    k = float(np.mean(((r - m) / (np.sqrt(v) + 1e-12))**4) - 3.0) if v > 0 else np.nan
    a1 = acf_lag1(r)
    a1sq = acf_lag1(r**2)
    return {"mean": m, "var": v, "skew": s, "kurt_excess": k, "acf1": a1, "acf1_sq": a1sq}


# -----------------------------------------------------------------------------
# Core multi-run driver
# -----------------------------------------------------------------------------

def run_dollar_game_multi(
    R: int = 20,
    m: int = 7,
    s: int = 2,
    N: int = 1001,
    rounds: int = 20000,
    sign: int = +1,                 # +1 trend-following, -1 minority-flavored
    base_seed: int = 12346,
    lambda_scale: float = 1.0,      # lam = lambda_scale * (1/(N*50))
    noise_std: float = 0.0,
    p0: float = 100.0,
) -> Dict[str, Any]:
    """Run R simulations and aggregate stats/plots/logs. Returns dict of outputs."""
    payoff_name = f"Dollar(sign={sign:+d})"
    logger = RunLogger(module="DollarMultiRun", payoff=payoff_name,
                       run_id=f"N{N}_T{rounds}_m{m}_s{s}_R{R}", seed=base_seed)

    lam_base = 1.0 / (N * 50.0)
    lam = lambda_scale * lam_base

    logger.log_params({
        "R": R, "m": m, "s": s, "N": N, "rounds": rounds, "sign": sign,
        "base_seed": base_seed, "lambda_base": lam_base, "lambda_scale": lambda_scale, "lam": lam,
        "noise_std": noise_std, "p0": p0,
    })

    prices = []  # list of arrays length T+1
    returns = [] # list of arrays length T (log returns)
    acfA = []    # lag1 autocorr of A for each run

    for r_idx in range(R):
        seed = base_seed + r_idx
        payoff = DollarGamePayoff(sign=sign)

        game = Game(num_players=N, memory=m, num_strategies=s, rounds=rounds, payoff_scheme=payoff)
        game.run()
        A = np.array(game.actions, dtype=float)

        p = build_price_series_log(A, lam=lam, noise_std=noise_std, seed=seed, p0=p0)
        prices.append(p)
        ret = np.diff(np.log(p))  # should equal lam*A + eps_t
        returns.append(ret)
        acfA.append(acf_lag1(A))

        # per-run metrics row
        st = stats_on_returns(ret)
        st.update({"run": r_idx, "seed": seed, "acf1_A": float(acfA[-1])})
        logger.log_metrics(st, step=r_idx)

    # -------- Aggregate & save tables --------
    T = len(prices[0]) - 1
    P = np.vstack(prices)  # shape (R, T+1)
    Rets = np.vstack(returns)  # shape (R, T)

    # Average path + 1 SD envelope
    mean_path = P.mean(axis=0)
    std_path = P.std(axis=0)

    # Save per-run summary table
    rows = []
    for r_idx in range(R):
        st = stats_on_returns(Rets[r_idx])
        rows.append({
            "run": r_idx,
            **st,
            "acf1_A": acfA[r_idx],
        })
    df_runs = pd.DataFrame(rows)
    runs_table = logger.log_table(df_runs, "per_run_return_stats")

    # Save mean/std path table
    df_path = pd.DataFrame({
        "t": np.arange(T+1, dtype=int),
        "mean_price": mean_path,
        "std_price": std_path,
    })
    path_table = logger.log_table(df_path, "mean_std_price_path")

    # -------- Plots --------
    art_dir = os.path.join(logger.get_dir(), "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    # Price mean ± 1 SD envelope
    tgrid = np.arange(T+1)
    plt.figure(figsize=(10, 6))
    plt.plot(tgrid, mean_path, label="Mean price")
    plt.fill_between(tgrid, mean_path - std_path, mean_path + std_path, alpha=0.2, label="±1 SD")
    plt.title(f"Dollar Game Price Paths (R={R})\nlam={lam:.3e}, noise={noise_std}")
    plt.xlabel("t")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    p_envelope = os.path.join(art_dir, "price_mean_sd_envelope.pdf")
    plt.savefig(p_envelope, format="pdf", dpi=300)
    plt.close()
    
    # Price paths
    plt.figure(figsize=(10,6))
    for i in range(R):
        plt.plot(tgrid, P[i], label=f"Sim_{i}")
    plt.title("Price paths")
    plt.xlabel("t")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    price_paths = os.path.join(art_dir, "price_paths.pdf")
    plt.savefig(price_paths, format="pdf", dpi=300)
    plt.close()

    # Optional: histogram of per-run mean log-returns
    plt.figure(figsize=(7, 5))
    plt.hist(df_runs["mean"], bins=30, edgecolor="black", alpha=0.8)
    plt.title("Per-run mean log-returns")
    plt.xlabel("mean(log return)")
    plt.ylabel("frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p_hist_mean = os.path.join(art_dir, "hist_mean_log_returns.pdf")
    plt.savefig(p_hist_mean, format="pdf", dpi=300)
    plt.close()

    # -------- Human-readable summary --------
    log_simulation([
        "Module: DollarMultiRun",
        f"R={R}, m={m}, s={s}, N={N}, T={rounds}, sign={sign:+d}",
        f"lambda = {lam:.3e} (scale={lambda_scale}, base=1/(N*50))",
        f"noise_std = {noise_std}, p0={p0}",
        f"Saved tables: {runs_table}, {path_table}",
        f"Artifacts: {p_envelope}, {p_hist_mean}",
        f"Run dir: {logger.get_dir()}",
    ])

    logger.close()

    return {
        "run_dir": logger.get_dir(),
        "mean_path": mean_path,
        "std_path": std_path,
        "per_run_table": runs_table,
        "path_table": path_table,
        "plots": {"envelope": p_envelope, "hist_mean": p_hist_mean},
        "lambda": lam,
    }


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    out = run_dollar_game_multi(
        R=5,
        m=12,
        s=2,
        N=1001,
        rounds=1000,
        sign=+1,            # trend-following
        base_seed=12345,
        lambda_scale=1.0,   # lam = 1/(N*50)
        noise_std=0.0,
        p0=100.0,
    )
    print("Done. Run folder:", out["run_dir"])
