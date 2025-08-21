#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 21:12:36 2025

@author: petermillington
"""

from payoffs.base import PayoffScheme
import numpy as np

class MGPayoff(PayoffScheme):
    def get_reward(self, a_i, total_action, N=None):
        raise NotImplementedError("Must implement get_reward in subclass.")

    def update(self, player, all_actions, total_action, history):
        if len(player.actions) == 0:
            return
        a_i = player.actions[-1]
        N=len(all_actions)
        reward = self.get_reward(a_i, total_action, N)
        player.points += reward

        # Convert ±1 history to binary string index
        h = history[-player.memory:]
        history_bits = ['1' if x == 1 else '0' for x in h]
        index = int(''.join(history_bits), 2)

        # Strategy reward: apply reward directly to matching strategies
        for i in range(player.num_strategies):
            hypothetical_action = player.strategies[i, index]
            hypothetical_reward = self.get_reward(hypothetical_action, total_action, N)
            player.scores[i] += hypothetical_reward

class BinaryMGPayoff(MGPayoff):
    def get_reward(self, a_i, total_action, N=None):
        return -a_i * np.sign(total_action)

class ScaledMGPayoff(MGPayoff):
    def get_reward(self, a_i, total_action, N=None):
        return -a_i * total_action
    
class SmallMinorityPayoff(MGPayoff):
    def get_reward(self, a_i, total_action, N=None):
        return -a_i * np.sign(total_action) * (N / 2 + abs(total_action))
    
class AssymetricMinorityPayoff(MGPayoff):
    def get_reward(self, a_i, total_action, N=None):
        return max(-a_i * np.sign(total_action) * (N / 2 + abs(total_action)),0)

    
class DollarGamePayoff(PayoffScheme):
    def update(self, player, all_actions, total_action, history):
        if len(player.actions) < 2:
            return
        a_i_prev = player.actions[-2]
        reward = a_i_prev * total_action
        player.dollar += reward
        player.dollar_per_round.append(player.dollar)
        
        # Update strategy scores based on previous action (optional logic)
        h = history[-player.memory:]
        history_bits = ['1' if x == 1 else '0' for x in h]
        index = int(''.join(history_bits), 2)
        for i in range(player.num_strategies):
            if player.strategies[i, index] == a_i_prev:
                player.scores[i] += reward



    