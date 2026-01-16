#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:41:18 2025

@author: petermillington
"""

import sys
import os
from datetime import datetime
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from core.game import Game
from core.game_config import GameConfig
from utils.logger import log_simulation

@dataclass
class PhaseDiagramConfig:
    payoff_key: str
    m_values: Iterable[int] = range(3,11)
    num_players: int = 301
    num_strategies: int = 2
    position_limit: int = None        
    noise_players: int = 0
    noise_allow_no_action: bool = False
    noise_position_limit: int = None
    rounds: int = 10_000
    num_games: int = 20
    market_maker: bool | None = None
    save_path: str = "plots/phase/phase_diagram_loglog.pdf"
    save_path2: str = "plots/phase/return_var_loglog.pdf"
    save_path3: str = "plots/phase/return_mean_loglog.pdf"
    log_path: str = "logs/simulation_log.txt"

def load_config(path: str) -> PhaseDiagramConfig:
    with open(path, "r") as f:
        data =json.load(f)
        config_data = {k: v for k, v in data.items() if not k.startswith('_')}
    return PhaseDiagramConfig(**config_data)


def simulate_single_game(args):
    """
    Returns a dict with keys:
        'sigma2' = variance of attendance
        'var prices' = variance of prices
        'mean returns' = mean of returns
        'var returns' = variance of returns
        
    Parameters
    ----------
    args : tuple
        memory m, payoff class, number of players, number of strategies, number of rounds,
        market_maker (None or True), position_limit.

    """
    m, cfg = args

    population_spec = {
        "total": cfg.num_players + cfg.noise_players,
        "cohorts":[
            {
                "count": cfg.num_players,
                "agent_type": "strategic",
                "memory": m,
                "payoff": cfg.payoff_key,
                "strategies": cfg.num_strategies,
                "position_limit": cfg.position_limit,
                }
            ]
        }

    if cfg.noise_players> 0:
        population_spec["cohorts"].append(
            {
                "agent_type": "noise",
                "count": cfg.noise_players,
                "allow_no_action": cfg.noise_allow_no_action,
                "position_limit": cfg.noise_position_limit})

    cfg_game = GameConfig(
        rounds=cfg.rounds,
        lambda_=1/(cfg.num_players * 50),
        mm=None,
        price=100,
        record_agent_series=True
        )

    game = Game(
        population_spec=population_spec,
        cfg=cfg_game,
    )
    results = game.run()

    attendance = results["Attendance"]
    prices = results["Prices"]

    prices_safe = np.where(prices <= 0, np.nan, prices)
    log_prices = np.log(prices_safe)
    returns = np.diff(log_prices)           # length rounds

    sigma2 = np.nanvar(attendance)
    var_prices = np.nanvar(prices)
    mean_returns = np.nanmean(returns)
    var_returns = np.nanvar(returns)

    return {"sigma2":sigma2, "var_prices":var_prices, "mean_returns":mean_returns,
            "var_returns":var_returns}

def run_phase_diagram(cfg: PhaseDiagramConfig):
    """
    Run num_games with specified number of rounds for each value of m in m_values
    The outputs are various graphs of the sigma ** 2 of the attendance, the variance
    of the price returns etc.
    """
    alphas = []
    normalized_vols = []
    metadata = []
    return_mean = []
    return_var = []

    with ProcessPoolExecutor() as executor:
        for m in cfg.m_values:
            args_list = [(m, cfg) for _ in range(cfg.num_games)]
            results = list(executor.map(simulate_single_game, args_list))

            sigmas = np.array([r["sigma2"] for r in results], dtype=float)
            r_mean = np.array([r["mean_returns"] for r in results], dtype=float)
            r_var = np.array([r["var_returns"] for r in results], dtype=float)

            sigma2_mean = np.mean(sigmas)
            r_var_means = np.mean(r_var)
            r_mean_ave = np.mean(r_mean)
            alpha = 2 ** m / cfg.num_players
            alphas.append(alpha)
            normalized_vols.append(sigma2_mean / cfg.num_players)
            return_var.append(r_var_means)
            return_mean.append(r_mean_ave)
            metadata.append(
                f"m={m}, alpha={alpha:.5f}, "
                f"sigma^2/N={sigma2_mean / cfg.num_players:.5f}, "
                f"m_maker={cfg.market_maker}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = cfg.save_path.replace(".pdf", f"_{timestamp}.pdf")
    save_path2 = cfg.save_path2.replace(".pdf", f"_{timestamp}.pdf")
    save_path3 = cfg.save_path3.replace(".pdf", f"_{timestamp}.pdf")

    label = cfg.payoff_key

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(alphas, normalized_vols, 'o-', label=label)
    plt.xlabel(r"$\alpha = 2^m / N$")
    plt.ylabel(r"$\sigma^2 / N$")
    plt.title(r"Phase Diagram (Log-Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.loglog(alphas, return_var, 'o-', label=label)
    plt.xlabel(r"$\alpha = 2^m / N$")
    plt.ylabel(r"$\sigma^2_r$")
    plt.title(r"(Log-Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    plt.savefig(save_path2, format="pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.loglog(alphas, return_mean, 'o-', label=label)
    plt.xlabel(r"$\alpha = 2^m / N$")
    plt.ylabel(r"$\sigma^2_r$")
    plt.title(r"(Log-Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    plt.savefig(save_path3, format="pdf")
    plt.close()

    # Logging
    metadata.insert(0, f"[{timestamp}] Phase Diagram with {label}")
    metadata.extend(
        [
            f"Players: {cfg.num_players}",
            f"Strategies: {cfg.num_strategies}",
            f"Rounds: {cfg.rounds}",
            f"Repetitions: {cfg.num_games}",
            f"Market Maker: {cfg.market_maker}",
            f"Position Limit: {cfg.position_limit}",
            f"Plot Saved: {save_path}, {save_path2}, {save_path3}"
        ]
    )
    metadata.append("="*60)

    log_simulation(metadata, cfg.log_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file for the phase diagram experiment.",
        )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_phase_diagram(cfg)
    
if __name__ == "__main__":
    main()
