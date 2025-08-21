#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:46:56 2025

@author: petermillington
"""
import numpy as np

class Player:
    def __init__(self, memory, num_strategies):
        self.memory = memory
        self.num_strategies = num_strategies
        self.strategies = np.random.choice([-1, 1], size=(num_strategies, 2 ** memory))
        self.scores = np.zeros(num_strategies)
        self.actions = []
        self.wins_per_round = []
        self.points = 0
        self.position = 0
        self.cash = 0
        self.strategy_switches = 0
        self.strategy = None

    def choose_action(self, full_history):
        # take only the last m bits
        h = full_history[-self.memory:]
        index = int(''.join(['1' if h == 1 else '0' for h in h]), 2)
        best_strat_idx = np.flatnonzero(self.scores == self.scores.max())
        chosen_idx = np.random.choice(best_strat_idx)
        if chosen_idx != self.strategy:
            self.strategy_switches += 1
        self.strategy = chosen_idx
        action = self.strategies[chosen_idx, index]
        self.actions.append(action)
        return action


    def update_score(self, strat_index, reward):
        self.scores[strat_index] += reward