#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 21:12:36 2025

@author: petermillington
"""

from payoffs.base import PayoffScheme
import numpy as np

class MGPayoff(PayoffScheme):
    def get_reward(self, a_i, total_action):
        raise NotImplementedError("Must implement get_reward in subclass.")

    def update(self, player, all_actions, total_action, history):
        if len(player.actions) == 0:
            return
        a_i = player.actions[-1]
        reward = self.get_reward(a_i, total_action)
        player.points += reward

        # Convert ±1 history to binary string index
        history_bits = ['1' if h == 1 else '0' for h in history]
        index = int(''.join(history_bits), 2)

        # Strategy reward: apply reward directly to matching strategies
        for i in range(player.num_strategies):
            if player.strategies[i, index] == a_i:
                player.scores[i] += reward

class ScaledMGPayoff(MGPayoff):
    def get_reward(self, a_i, total_action):
        return -a_i * total_action

class BinaryMGPayoff(MGPayoff):
    def get_reward(self, a_i, total_action):
        return -a_i * np.sign(total_action)