#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:57:58 2025

@author: petermillington
"""

from core.player import Player
import numpy as np

class Game:
    def __init__(self, num_players, memory, num_strategies, rounds, payoff_scheme):
        self.players = [Player(memory, num_strategies) for _ in range(num_players)]
        self.memory = memory
        self.rounds = rounds
        self.payoff_scheme = payoff_scheme
        self.history = list(np.random.choice([-1, 1], size=memory))
        self.actions = []

    def play_round(self):
        current_actions = [p.choose_action(self.history) for p in self.players]
        A_t = sum(current_actions)
        self.actions.append(A_t)

        for player in self.players:
            self.payoff_scheme.update(player, current_actions, A_t, self.history)

        # Update global history (minority encoded as 0/1)
        minority_action = -1 if A_t > 0 else 1
        self.history = (self.history + [minority_action])[-self.memory:]

    def run(self):
        for _ in range(self.rounds):
            self.play_round()