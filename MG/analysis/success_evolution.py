#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 18:19:23 2025

@author: petermillington
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from core.game import Game
from payoffs.mg import BinaryMGPayoff,  ScaledMGPayoff, DollarGamePayoff

def plot_success_evolution(game, interval_lengths, save_path):
    """
    Takes output from game (usually lengthy) and plots moving averages of success
    rates.
    """
    avg_success_rates = []
    rounds = len(game.players[0].wins_per_round)

    for interval in interval_lengths:
        segment_means = []
        for p in game.players:
            wins = np.array(p.wins_per_round)
            # ignore incomplete final interval
            full_intervals = rounds // interval
            wins = wins[:full_intervals * interval]
            reshaped = wins.reshape(-1, interval)
            interval_means = reshaped.mean(axis=1)
            segment_means.append(interval_means)

        avg_over_players = np.mean(segment_means, axis=0)
        avg_success_rates.append((interval, avg_over_players))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    for interval, rates in avg_success_rates:
        ax.plot(np.arange(len(rates)) * interval, rates, label=f"Δt={interval}")

    ax.set_title("Average Success Rate over Time")
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg. Success Rate")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.close()

if __name__ == "__main__":

    payoff = DollarGamePayoff  # or ScaledMGPayoff

    # Success evolution demo run
    long_game = Game(
        num_players=501,
        memory=12,
        num_strategies=5,
        rounds=20000,
        payoff_scheme=payoff()
    )
    long_game.run()
    intervals = [500, 1000, 2000, 5000]
    plot_success_evolution(long_game, intervals, save_path="plots/success/success_evolution.pdf")
