#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 12:22:25 2025

@author: petermillington

A game is run with varied memories for the agents.  Parameters for the game and 
agents are defined below.  Two plots measuring success by agent type (memory)
are produced and saved with logging details.
"""

import sys
import os
from typing import Optional, Dict, Any, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.game import Game
from payoffs.mg import BinaryMGPayoff, ScaledMGPayoff, SmallMinorityPayoff, DollarGamePayoff
from utils.logger import log_simulation, RunLogger, _ts, _ensure_dir


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def _extract_success_lists(results: Dict[int, Any]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for m, val in results.items():
        if isinstance(val, list):
            out[m] = [float(x) for x in val]
        elif isinstance(val, dict):
            sr = val.get("success_rates", [])
            out[m] = [float(x) for x in sr]
        else:
            raise TypeError(
        f"Unexpected results value for m={m}: {type(val)}. Expected list or dict with 'success_rates'."
        )
    return out

def _extract_wealth_lists(results: Dict[int, Any]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for m, val in results.items():
        if isinstance(val, list):
            out[m] = [float(x) for x in val]
        elif isinstance(val, dict):
            sr = val.get("wealth", [])
            out[m] = [float(x) for x in sr]
        else:
            raise TypeError(
        f"Unexpected results value for m={m}: {type(val)}. Expected list or dict with 'success_rates'."
        )
    return out

def plot_success_boxplot(
    results: Dict[int, Any],
    payoff_scheme_name: str,
    save_dir: str = "plots/success",
    filename_prefix: Optional[str] = None,
    ) -> str:
    """
    Plots success rates by memory.
    """
    _ensure_dir(save_dir)
    success_by_m = _extract_success_lists(results)
    m_values = sorted(success_by_m.keys())
    data = [success_by_m[m] for m in m_values]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[f"m={m}" for m in m_values])
    plt.title(f"Success Rates by Memory (Payoff={payoff_scheme_name})")
    plt.xlabel("Memory Size m")
    plt.ylabel("Success Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    bits = ["success_boxplot", payoff_scheme_name]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_wealth_boxplot(
    results: Dict[int, Any],
    payoff_scheme_name: str,
    save_dir: str = "plots/success",
    filename_prefix: Optional[str] = None,
    ) -> str:
    """
    Plots wealth by memory size
    """
    _ensure_dir(save_dir)
    wealth_by_m = _extract_wealth_lists(results)
    m_values = sorted(wealth_by_m.keys())
    data = [wealth_by_m[m] for m in m_values]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[f"m={m}" for m in m_values])
    plt.title(f"Wealth by Memory (Payoff={payoff_scheme_name})")
    plt.xlabel("Memory Size m")
    plt.ylabel("Wealth")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    bits = ["wealth_boxplot", payoff_scheme_name]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_average_success_line(
    results: Dict[int, Any],
    payoff_scheme_name: str,
    save_dir: str = "plots/success",
    filename_prefix: Optional[str] = None,
    show_std_band: bool = True,
    ) -> str:
    """
    plots average succes line.
    """
    _ensure_dir(save_dir)
    success_by_m = _extract_success_lists(results)

    m_values = sorted(success_by_m.keys())
    means = np.array([np.mean(success_by_m[m]) if success_by_m[m] else np.nan for m in m_values])
    stds = np.array([np.std(success_by_m[m]) if success_by_m[m] else np.nan for m in m_values])

    plt.figure(figsize=(10, 6))
    plt.plot(m_values, means, marker="o")
    if show_std_band:
        upper = means + stds
        lower = means - stds
        plt.fill_between(m_values, lower, upper, alpha=0.2)

    plt.title(f"Average Success Rate vs Memory (Payoff={payoff_scheme_name})")
    plt.xlabel("Memory Size m")
    plt.ylabel("Average Success Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    bits = ["success_mean_line", payoff_scheme_name]
    if filename_prefix:
        bits.insert(1, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    return path

def run_varied_memory_game(
        m_values: List[int],
        s: int,
        payoff_scheme,
        N_per_m: int=50,
        rounds: int=1000,
        lambda_value: float=100
    )->Dict[int, Dict[str, List[float]]]:
    """
    Create N_per_m players for each memory length in m_values.
    Returns a dictionary mapping m to success rates, average score 
    and strategy switches of its players.
    """
    game = Game(
        num_players=N_per_m,
        memory_list=m_values,
        num_strategies=s,
        rounds=rounds,
        payoff_scheme=payoff_scheme(),
        lambda_value=lambda_value
    )
    game.run()

    results: Dict[int, Dict[str, List[float]]] = {}
    idx = 0
    for m in m_values:
        group_size = min(N_per_m, len(game.players)-idx)
        group_players = game.players[idx: idx + group_size]

        success_rates = [np.mean(p.wins_per_round) for p in group_players]
        avg_scores = [p.points/max(1,len(p.wins_per_round)) for p in group_players]
        switches = [p.strategy_switches for p in group_players]
        wealth = [p.wealth for p in group_players]
        results[m] = {
            "success_rates": success_rates,
            "average_score": avg_scores,
            "strategy_switches": switches,
            "wealth": wealth
        }
        idx += N_per_m
    return results


def main() -> None:
    """Main program"""
    # --- Parameters for inputting ---
    payoff_cls = BinaryMGPayoff     # or ScaledMGPayoff, SmallMinorityPayoff
    m_values = list(range(2, 15))
    S = 4
    N_per_m = 175
    rounds = 40000
    seed = 12345        # set None to allow non-deterministic runs
    lambda_value = 50 * N_per_m * len(m_values)

    # --- Initialize logger (creates timestamped run folder and seeds RNGs) ---
    logger = RunLogger(module="HeteroMemory", payoff=payoff_cls.__name__, run_id=None, seed=seed)
    logger.log_params({
        "memories": m_values,
        "strategies_per_player": S,
        "agents_per_memory": N_per_m,
        "rounds": rounds,
        "seed": seed,
    })

    # --- Run simulation ---
    results = run_varied_memory_game(m_values, S, payoff_scheme=payoff_cls,
                                     N_per_m=N_per_m, rounds=rounds,
                                     lambda_value=lambda_value)

    # --- Plot & save figures (timestamped via run folder) ---
    prefix = logger.run_info["start_time"]  # fix file prefix for all files saved in case of duplication.
    box_path = plot_success_boxplot(results, payoff_scheme_name=payoff_cls.__name__,
                                    save_dir="plots/success", filename_prefix=prefix)
    wealth_path = plot_wealth_boxplot(results, payoff_scheme_name=payoff_cls.__name__,
                                    save_dir="plots/success", filename_prefix=prefix)
    mean_path = plot_average_success_line(results, payoff_scheme_name=payoff_cls.__name__,
                                          save_dir="plots/success", filename_prefix=prefix)

    # --- Register artifacts & any summary tables ---
    logger.log_artifact(box_path)
    logger.log_artifact(mean_path)

    # Save per-player metrics as a tidy table for quick analysis
    rows = []
    for m, metrics in results.items():
        n = len(metrics["success_rates"]) if metrics["success_rates"] else 0
        for i in range(n):
            rows.append({
                "memory": m,
                "success_rate": metrics["success_rates"][i],
                "average_score": metrics["average_score"][i],
                "strategy_switches": metrics["strategy_switches"][i],
            })
        df_players = pd.DataFrame(rows, columns=["memory",
                                                 "success_rate",
                                                 "average_score",
                                                 "strategy_switches"])
        table_path = logger.log_table(df_players, name="per_player_metrics")

    # Optionally, log a couple of high-level metrics
    # (example: average success rate by memory)
    avg_by_m = {int(m): float(np.mean(v["success_rates"])) for m, v in results.items()}
    logger.save_dict("avg_success_by_memory", avg_by_m)

    # --- Human-readable summary (append-only file) ---
    log_simulation([
    f"Module: HeteroMemory",
    f"Payoff: {payoff_cls.__name__}",
    f"Memories: {m_values}",
    f"Strategies per player: {S}",
    f"Agents per memory: {N_per_m}",
    f"Rounds: {rounds}",
    f"Seed: {seed}",
    f"Saved plot: {box_path}",
    f"Saved plot: {mean_path}",
    f"Saved plot: {wealth_path}"
    f"Run dir: {logger.get_dir()}",
    f"Per-player CSV: {table_path}",
    ])

    logger.close()
    print(f"Done. Run folder: {logger.get_dir()}")

if __name__ == "__main__":
    main()
