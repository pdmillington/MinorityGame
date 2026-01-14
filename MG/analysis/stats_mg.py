#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:18:24 2025

@author: petermillington
"""

import numpy as np
from core.game import Game
from payoffs.mg import BinaryMGPayoff

def compute_volatility(actions):
    """Compute variance of total action A(t)."""
    return np.var(actions)

def compute_scaled_volatility(actions, N):
    """Compute σ² / N."""
    return np.var(actions) / N

def compute_alpha(memory, N):
    return 2 ** memory / N

def compute_win_rate(player):
    """Compute percentage of rounds player gained points."""
    total = len(player.actions)
    wins = sum(1 for i in range(1, total) if player.points > 0)
    return wins / total if total > 0 else 0

def run_game_once(N, m, s, rounds):
    game = Game(
        num_players=N,
        memory=m,
        num_strategies=s,
        rounds=rounds,
        payoff_scheme=BinaryMGPayoff()
    )
    game.run()
    return np.var(game.actions) / N  # σ²/N
