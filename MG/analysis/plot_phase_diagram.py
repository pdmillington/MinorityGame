#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:41:18 2025

@author: petermillington
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from core.game import Game
from payoffs.mg import BinaryMGPayoff, ScaledMGPayoff
from utils.logger import log_simulation
from concurrent.futures import ProcessPoolExecutor

def compute_volatility(game):
    actions = np.array(game.actions)
    return np.var(actions)

def simulate_single_game(args):
    m, payoff_class, num_players, num_strategies, rounds = args
    game = Game(
        num_players=num_players,
        memory=m,
        num_strategies=num_strategies,
        rounds=rounds,
        payoff_scheme=payoff_class()
    )
    game.run()
    return np.var(game.actions)

def run_phase_diagram(
    payoff_class,
    m_values=range(2, 15),
    num_players=301,
    num_strategies=2,
    rounds=20000,
    num_games=20,
    save_path="plots/phase/phase_diagram_loglog.pdf",
    log_path="logs/simulation_log.txt"
):
    alphas = []
    normalized_vols = []
    metadata = []
    
    with ProcessPoolExecutor() as executor:
        for m in m_values:
            args_list = [(m, payoff_class, num_players, num_strategies, rounds) for _ in range(num_games)]
            sigmas = list(executor.map(simulate_single_game, args_list))
            
            sigma2 = np.mean(sigmas)
            alpha = 2 ** m / num_players
            alphas.append(alpha)
            normalized_vols.append(sigma2 / num_players)
            metadata.append(f"m={m}, alpha={alpha:.5f}, sigma^2/N={sigma2 / num_players:.5f}")
  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_path.replace(".pdf", f"_{timestamp}.pdf")
    
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

    # Logging


    metadata.insert(0, f"[{timestamp}] Phase Diagram with {payoff_class.__name__}")
    metadata.extend([f"Players: {num_players}",
    f"Strategies: {num_strategies}",
    f"Rounds: {rounds}",
    f"Repetitions: {num_games}",
    f"Plot Saved: {save_path}"])
    metadata.append("="*60)
    
    log_simulation(metadata, log_path)

if __name__ == "__main__":
    run_phase_diagram(ScaledMGPayoff)