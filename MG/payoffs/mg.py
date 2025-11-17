#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 21:12:36 2025

@author: petermillington
"""

from payoffs.base import PayoffScheme
import numpy as np

class MGPayoff(PayoffScheme):
    """General MGPayoff class, checks implementation of functions in the subclasses."""

    def get_reward(self, a_i, total_action, N=None, lambda_value=None):
        """Calculates reward for given subclass"""
        raise NotImplementedError("Must implement get_reward in subclass.")

    def get_reward_vector(self, a_vec, total_action, N=None, lambda_value=None):
        """Vectorised reward calculation for given subclass."""
        raise NotImplementedError("Must implement get_reward_vector in subclass.")

    def get_win(self, a_i, total_action):
        """Allows player to calculate a win for each round depending on the payoff"""
        raise NotImplementedError("Must implement get_win in subclass.")

class BinaryMGPayoff(MGPayoff):
    """
    Classic MG payoff: +1 if in Minority Group, -1 otherwise.  Immediate because
    rewards are calculated with curent decision data.
    """
    mode = "immediate"
    expects_delta_price = False

    def get_reward(self, a_i, total_action, N=None, lambda_value=None):
        return -a_i * np.sign(total_action)

    def get_reward_vector(self, a_vec, total_action, N=None, lambda_value=None):
        return -a_vec * np.sign(total_action)

    def get_win(self, a_i, total_action):
        minority=-1 if total_action > 0 else 1
        return a_i == minority

class ScaledMGPayoff(MGPayoff):
    """
    Extension of the classic payoff, but where the reward is scaled by the size
    of the total action.  So larger negative reward for those in the majority 
    when the majority is larger, larger positive reward for minority members 
    when the minority is smaller.
    """
    mode = "immediate"
    expects_delta_price = False

    def get_reward(self, a_i, total_action, N=None, lambda_value=None):
        return -a_i * total_action

    def get_reward_vector(self, a_vec, total_action, N=None, lambda_value=None):
        return -a_vec * total_action

    def get_win(self, a_i, total_action):
        minority=-1 if total_action > 0 else 1
        return a_i == minority

class SmallMinorityPayoff(MGPayoff):
    """
    Specific payout that rewards a small minority and generates a discontinuity
    in reward at the mid point
    """
    mode = "immediate"
    expects_delta_price = False

    def get_reward(self, a_i, total_action, N=None, lambda_value=None):
        return -a_i * np.sign(total_action) * (N/2 + abs(total_action))

    def get_reward_vector(self, a_vec, total_action, N=None, lambda_value=None):
        return -a_vec * np.sign(total_action) * (N/2 + abs(total_action))

    def get_win(self, a_i, total_action):
        minority = -1 if total_action > 0 else 1
        return a_i == minority

class AssymetricMinorityPayoff(MGPayoff):
    """
    Same as SmallMinority, but only allows a positive reward
    """
    mode = "immediate"
    expects_delta_price = False

    def get_reward(self, a_i, total_action, N=None, lambda_value=None):
        scale = N / 2 + np.abs(total_action)
        return max(-a_i * np.sign(total_action) * scale, 0)

    def get_reward_vector(self, a_vec, total_action, N=None, lambda_value=None):
        scale = N / 2 + np.abs(total_action)
        return np.maximum(-a_vec * np.sign(total_action) * scale, 0)

    def get_win(self, a_i, total_action):
        minority = -1 if total_action > 0 else 1
        return a_i == minority

class DollarGamePayoff(PayoffScheme):
    """
    Mode is delayed because the decision at round t-1 is evaluated at t.  Success
    is when decision is in the majority.  The idea is that price movement is positively
    correlated with majority direction, so optimal trader purchases at t-1 when 
    majority purchases at t and vice-versa.
    """
    mode = "delayed"
    expects_delta_price = True

    def __init__(self, sign=+1, lambda_value=1.0):
        # sign=+1 → trend-following r(t-1) = + a_i(t-1)*A(t)
        # sign=-1 → minority-flavored r(t-1) = - a_i(t-1)*A(t)
        self.sign = 1 if sign >= 0 else -1
        self.lambda_value = float(lambda_value)

    def get_reward(self, a, total_action, N, lambda_value):
        return self.sign * a * total_action / lambda_value

    def get_reward_vector(self, a_vec, total_action, N, lambda_value):
        return self.sign * a_vec * total_action / lambda_value

    def get_win(self, a_i, total_action):
        majority = -1 if total_action < 0 else 1
        return a_i == majority

PAYOFF_REGISTRY = {
    "BinaryMG": BinaryMGPayoff(),
    "ScaledMG": ScaledMGPayoff(),
    "SmallMinority": SmallMinorityPayoff(),
    "DollarGame": DollarGamePayoff()}
