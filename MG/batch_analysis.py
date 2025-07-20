#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 19:43:46 2025

@author: petermillington
"""

import numpy as np
from multiprocessing import Pool
from core.game import Game
from payoffs.mg import BinaryMGPayoff, ScaledMGPayoff


def run_game_once(N, m, s, rounds, payoff_class):
    game = Game(
        num_players=N,
        memory=m,
        num_strategies=s,
        rounds=rounds,
        payoff_scheme=payoff_class()
    )
    game.run()
    return np.var(game.actions) / N


def random_baseline_volatility(N, rounds):
    A = [sum(np.random.choice([-1, 1], size=N)) for _ in range(rounds)]
    return np.var(A) / N


def sweep_alpha_vs_volatility(memory_range, N, s, rounds, repeats, payoff_class):
    alphas = []
    vols = []
    for m in memory_range:
        with Pool() as pool:
            results = pool.starmap(run_game_once, [(N, m, s, rounds, payoff_class) for _ in range(repeats)])
        sigma2_over_N = np.mean(results)
        alpha = 2 ** m / N
        alphas.append(alpha)
        vols.append(sigma2_over_N)
    return alphas, vols


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    memory_range = range(2, 11)
    N = 101
    s = 2
    rounds = 1000
    repeats = 20

    # Run both payoff types
    alphas_bin, vols_bin = sweep_alpha_vs_volatility(memory_range, N, s, rounds, repeats, BinaryMGPayoff)
    alphas_scaled, vols_scaled = sweep_alpha_vs_volatility(memory_range, N, s, rounds, repeats, ScaledMGPayoff)

    baseline = random_baseline_volatility(N, rounds)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas_bin, vols_bin, marker='o', label='Binary Payoff')
    plt.plot(alphas_scaled, vols_scaled, marker='s', label='Proportional Payoff')
    plt.axhline(y=baseline, color='gray', linestyle='--', label='Random Baseline (±1)')
    plt.xlabel(r'$\alpha = 2^m / N$')
    plt.ylabel(r'$\sigma^2 / N$')
    plt.title('Volatility vs. $\alpha$ in the Minority Game')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()