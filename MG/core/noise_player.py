#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:51:24 2025

@author: petermillington
"""

from typing import List, Optional
import numpy as np
from core.base_agent import BaseAgent

class NoisePlayer(BaseAgent):
    """
    Noise player is an agent that picks actions randomly.
    """
    def __init__(self,
                 position_limit: Optional[int] = None,
                 initial_cash: float = 0.0,
                 allow_no_action: bool = False,
                 leverage_limit: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = None,
                 ):
        """
        Initialize noise player
        Parameters
        ----------
        position_limit: int, optional
            maximum allowed position
        initial_cash: float
            Starting cash amount, 0.0 default
        leverage_limit
            leverage limit, not used currently
        rng: Optional
            random number generator
        seed: int, optional
            RNG seed
        """
        super().__init__(position_limit, initial_cash, rng, seed)

        self.memory = 1
        self.num_strategies = 0
        self.wins = 0
        self.points = 0.0
        self.strategy = -1
        self.possible_action = [-1, 0, +1] if allow_no_action else [-1, +1]

        self._action: Optional[int] = None

    def choose_action(self, history: List[int]) -> int:
        """
        Choose action randomly.
        history is ignored by the Noise Trader
        Returns an integer
        """
        desired_action = self.rng.choice(self.possible_action)

        action = self._enforce_position_limit(desired_action)

        self._action = action

        return action

    def update(self,
               N,
               flow,
               price: float,
               lambda_value: float) -> None:
        """
        Update inventory, cash, wealth etc.
        """
        if self._action is not None:
            self._apply_trade(self._action, price)
            self._action = None
