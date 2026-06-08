#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:55:04 2025

@author: petermillington
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from core.game import Game
from payoffs.mg import BinaryMGPayoff, ScaledMGPayoff, DollarGamePayoff
from utils.logger import log_simulation

# Helper function to calculate return and variances

def r_var_from_wealth(wealth_per_round):
    w = np.asarray(wealth_per_round, dtype=float)
    dw_per_round = w[1:] - w[:-1]
    av_r = np.mean(dw_per_round)
    var_r = np.var(dw_per_round)
    return av_r, var_r

def simulate_single_game(args):
    """
    Returns a dict with keys:
        'risk' = variance of changes in wealth
        'return' = average change in wealth

        


    Parameters
    ----------
    args : tuple
        memory, payoff type, number of players, number of strategies, number of rounds.

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
    
    m, payoff_class, num_players, num_strategies, rounds = args
    
    payoff = payoff_class()
    
    game = Game(
        num_players=num_players,
        memory=m,
        num_strategies=num_strategies,
        rounds=rounds,
        payoff_scheme=payoff,
        lambda_value=num_players*30,
        market_maker=True,
    )
    game.run()
    wealth_per_round = game.mm.wealth_per_round
    r_avg, var = r_var_from_wealth(wealth_per_round)
    
    return {"return":r_avg, "var_wealth":var}

# Main function
def run_wealth_distrib(
        payoff_class,
        m_values=range(7,13),
        num_players=501,
        num_strategies=6,
        rounds=500,
        num_games=1000,
        market_maker=True,
        save_path="plots/mm_risk/avg_return.pdf",
        save_path2="plots/mm_risk/risk.pdf",
        log_path="logs/simulation_log.txt"
        ):
    
    alphas= []
    metadata = []
    ret_mean = []
    ret_mean_lower = []
    ret_mean_higher = []
    risk_mean = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_path.replace(".pdf", f"_{timestamp}.pdf")
    save_path2 = save_path2.replace(".pdf", f"_{timestamp}.pdf")
    
    with ProcessPoolExecutor() as executor:
        for m in m_values:
            args_list = [(m, payoff_class, num_players, num_strategies, rounds) for _ in range(num_games)]
            results = list(executor.map(simulate_single_game, args_list))
            
            returns = np.array([r["return"] for r in results], dtype=float)
            risk = np.array([r["var_wealth"] for r in results], dtype=float)
            mean_ret = np.mean(returns)
            mean_risk = np.mean(risk)
            ret_mean.append(mean_ret)
            risk_mean.append(mean_risk)
            alpha = 2 ** m / num_strategies
            alphas.append(alpha)
            print(np.shape(alphas))
            print(np.shape(ret_mean))
            metadata.append(f"m={m}, s={num_strategies}, alpha={alpha:.5f}, mean return={mean_ret:.5f}")
        
            
    
    
    # Plotting - returns
    plt.figure(figsize=(8, 6))
    plt.semilogx(alphas, ret_mean, 'o-', label=payoff_class.__name__)
    #plt.fill_between(alphas, np.array(ret_mean) - np.sqrt(np.array(risk_mean)), np.array(ret_mean) + np.sqrt(np.array(risk_mean)), alpha=0.25, label=r"±$\sigma$")
    plt.xlabel(r"$\alpha = 2^m / s$")
    plt.ylabel(r"$\overline{r}$")
    plt.title(r"Mean return (Lin-Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf")
    plt.close()
    
    # Plotting - risk
    plt.figure(figsize=(8, 6))
    plt.loglogx(alphas, risk_mean, 'o-', label=payoff_class.__name__)
    plt.xlabel(r"$\alpha = 2^m / s$")
    plt.ylabel(r"$Risk$")
    plt.title(r"Risk (Log-Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path2, format="pdf")
    plt.close()
    
    # Logging


    metadata.insert(0, f"[{timestamp}] MM Wealth Distribution {payoff_class.__name__}")
    metadata.extend([f"Players: {num_players}",
    f"Strategies: {num_strategies}",
    f"Rounds: {rounds}",
    f"Repetitions: {num_games}",
    f"Plot Saved: {save_path}, {save_path2}"])
    metadata.append("="*60)
    
    log_simulation(metadata, log_path)
  
if __name__ == "__main__":
    run_wealth_distrib(DollarGamePayoff)    