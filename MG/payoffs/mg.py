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

    def update(self, player, all_actions, total_action, history, price):
        if len(player.actions) == 0:
            return
        a_i = player.actions[-1]
        N=len(all_actions)
        reward = self.get_reward(a_i, total_action, N)
        player.points += reward
        player.cash += -a_i * price
        player.cash_per_round.append(-a_i * price)
        player.wealth_per_round.append(player.position * price + player.cash)
        player.wealth = player.wealth_per_round[-1]

        # Convert ±1 history to binary string index
        h = history[-player.memory:]
        history_bits = ['1' if x == 1 else '0' for x in h]
        index = int(''.join(history_bits), 2)

        # Strategy reward: apply reward directly to matching strategies
        for i in range(player.num_strategies):
            hypothetical_action = player.strategies[i, index]
            hypothetical_reward = self.get_reward(hypothetical_action, total_action, N)
            player.scores[i] += hypothetical_reward
            
        #MG style win for current round t
        minority_action = -1 if total_action > 0 else 1
        player.wins_per_round.append(1 if a_i == minority_action else 0)

class BinaryMGPayoff(MGPayoff):
    uses_orders = False
    expects_delta_price = False
    def get_reward(self, a_i, total_action, N=None):
        return -a_i * np.sign(total_action)

class ScaledMGPayoff(MGPayoff):
    uses_orders = False
    expects_delta_price = False
    def get_reward(self, a_i, total_action, N=None):
        return -a_i * total_action
    
class SmallMinorityPayoff(MGPayoff):
    uses_orders = False
    expects_delta_price = False
    def get_reward(self, a_i, total_action, N=None):
        return -a_i * np.sign(total_action) * (N/2 + abs(total_action))
    
class AssymetricMinorityPayoff(MGPayoff):
    uses_orders = False
    expects_delta_price = False
    def get_reward(self, a_i, total_action, N=None):
        return max(-a_i * np.sign(total_action) * (N / 2 + abs(total_action)),0)

    
class DollarGamePayoff(PayoffScheme):
    uses_orders = True
    expects_delta_price = True
    def __init__(self, sign=+1, lambda_value=1.0):
        # sign=+1 → trend-following r(t-1) = + a_i(t-1)*A(t)
        # sign=-1 → minority-flavored r(t-1) = - a_i(t-1)*A(t)
        self.sign = 1 if sign >= 0 else -1
        self.lambda_value = float(lambda_value)

    def _prev_index(self, history, m):
        """
        Build the index for t-1 using the m-bit window that was used to decide at t-1.
        Assuming `history` at this call already contains the minority bit for t-1, but not for t.
        Then the correct slice for t-1's lookup is history[-m-1 : -1].
        """
        if len(history) < m + 1:
            return None
        window = history[-m-1:-1]
        idx = int(''.join('1' if bit == 1 else '0' for bit in window), 2)
        return idx

    def update(self, player, all_actions, total_action, history, price, delta_price=None):
        # Need at least two chosen actions to pay t-1 with A(t)
        if len(player.actions) < 2:
            # keep alignment of per-round series
            a_i = player.actions[-1]
            player.cash_per_round.append(-a_i * price)
            player.cash += -a_i * price
            player.wealth_per_round.append(player.position * price + player.cash)
            player.wealth = player.wealth_per_round[-1]
            return

        # Pay chosen action from previous round with current total_action A(t)
        a_prev = player.actions[-2]
        a_i = player.order_history[-1]
        reward = self.sign * a_prev * total_action / self.lambda_value

        
        player.points += reward
        player.cash_per_round.append(-a_i * price)
        player.cash += -a_i * price
        player.wealth_per_round.append(player.position * price + player.cash)
        player.wealth = player.wealth_per_round[-1]
        
        sA = 1 if total_action > 0 else -1
        win = 1 if (a_prev == self.sign * sA) else 0
        
        player.wins_per_round.append(win)
        
        # virtual update at t-1 using the exact index you stored at decision time
        if hasattr(player, "index_history") and len(player.index_history) >= 2:
            prev_idx = player.index_history[-2]
            virt_actions = player.strategies[:, prev_idx]
            player.scores += self.sign * virt_actions * (self.lambda_value * total_action)
        


    