#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:41:18 2025

@author: petermillington
"""

import sys
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import Game
from payoffs.mg import BinaryMGPayoff, ScaledMGPayoff, DollarGamePayoff
from utils.logger import log_simulation

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
        memory, payoff type, number of players, number of strategies, number of rounds,
        market_maker (None or True).

    Returns
    -------
    sigma2 : list of floats
        variance of attendance for a list of games.
    var prices : list of floats
        variance of prices
    mean returns : list of floats
        average returns.
    var returns: 

    """
    m, payoff_class, num_players, num_strategies, rounds, market_maker, position_limit = args

    payoff = payoff_class()

    game = Game(
        num_players=num_players,
        memory=m,
        num_strategies=num_strategies,
        rounds=rounds,
        payoff_scheme=payoff,
        lambda_value=num_players*30,
        market_maker=market_maker,
        position_limit=position_limit
    )
    game.run()
    sigma2 = np.var(game.actions)
    var_prices = np.var(game.prices)
    mean_returns = np.mean(game.returns)
    var_returns = np.var(game.returns)

    return {"sigma2":sigma2, "var_prices":var_prices, "mean_returns":mean_returns,
            "var_returns":var_returns}

def run_phase_diagram(
    payoff_class,
    m_values=range(3, 11),
    num_players=301,
    num_strategies=2,
    position_limit=10,
    rounds=5000,
    num_games=20,
    market_maker=None,
    save_path="plots/phase/phase_diagram_loglog.pdf",
    save_path2="plots/phase/return_var_loglog.pdf",
    save_path3="plots/phase/return_mean_loglog.pdf",
    log_path="logs/simulation_log.txt"
):
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
        for m in m_values:
            args_list = [(m,
                          payoff_class,
                          num_players,
                          num_strategies,
                          rounds,
                          market_maker,
                          position_limit
                          )
                         for _ in range(num_games)]
            results = list(executor.map(simulate_single_game, args_list))

            sigmas = np.array([r["sigma2"] for r in results], dtype=float)
            r_mean = np.array([r["mean_returns"] for r in results], dtype=float)
            r_var = np.array([r["var_returns"] for r in results], dtype=float)

            sigma2_mean = np.mean(sigmas)
            r_var_means = np.mean(r_var)
            r_mean_ave = np.mean(r_mean)
            alpha = 2 ** m / num_players
            alphas.append(alpha)
            normalized_vols.append(sigma2_mean / num_players)
            return_var.append(r_var_means)
            return_mean.append(r_mean_ave)
            metadata.append(f"m={m}, alpha={alpha:.5f}, sigma^2/N={sigma2_mean / num_players:.5f}, m_maker={market_maker}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_path.replace(".pdf", f"_{timestamp}.pdf")
    save_path2 = save_path2.replace(".pdf", f"_{timestamp}.pdf")
    save_path3 = save_path3.replace(".pdf", f"_{timestamp}.pdf")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(alphas, normalized_vols, 'o-', label=payoff_class.__name__)
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
    plt.loglog(alphas, return_var, 'o-', label=payoff_class.__name__)
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
    plt.loglog(alphas, return_mean, 'o-', label=payoff_class.__name__)
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
    metadata.insert(0, f"[{timestamp}] Phase Diagram with {payoff_class.__name__}")
    metadata.extend([f"Players: {num_players}",
    f"Strategies: {num_strategies}",
    f"Rounds: {rounds}",
    f"Repetitions: {num_games}",
    f"Market Maker: {market_maker}",
    f"Position Limit: {position_limit}",
    f"Plot Saved: {save_path}, {save_path2}, {save_path3}"])
    metadata.append("="*60)

    log_simulation(metadata, log_path)

if __name__ == "__main__":
    run_phase_diagram(BinaryMGPayoff)
