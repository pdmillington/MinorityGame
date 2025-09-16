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
        self.dollar_per_round = []
        self.index_history = []
        self.points = 0
        self.position = 0             # start flat; or +1/-1 if you prefer
        self.position_history = []
        self.order_history = []
        self.dollar = 0
        self.cash = 0
        self.strategy_switches = 0
        self.strategy = None

    def choose_action(self, full_history):
        """
        Decide an action (±1) based on the last `memory` bits of full_history.
        - For MG: this is the action that will be summed into A.
        - For $-game with uses_orders=True: this is the DESIRED POSITION; Game will
          turn it into an order = desired - current position.
        """
        h = full_history[-self.memory:]
        index = int(''.join(['1' if bit == 1 else '0' for bit in h]), 2)
        self.index_history.append(index)
        best_strat_idx = np.flatnonzero(self.scores == self.scores.max())
        chosen_idx = np.random.choice(best_strat_idx)
        
        # 3) Track switches & strategy history
        if getattr(self, "strategy", None) is not None and chosen_idx != self.strategy:
            self.strategy_switches += 1
        self.strategy = chosen_idx
        if hasattr(self, "strategy_history"):
            self.strategy_history.append(chosen_idx)
        
        # 4) Action / desired position from the chosen strategy at this index (±1)
        action = int(self.strategies[chosen_idx, index])
        self.actions.append(action)
        
        
        return action

    

    

    def update_score(self, strat_index, reward):
        self.scores[strat_index] += reward