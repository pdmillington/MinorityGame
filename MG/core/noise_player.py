#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:51:24 2025

@author: petermillington
"""

from typing import List, Optional, Dict, Any
import numpy as np

class NoisePlayer:
    """
    Noise player is an agent that picks actions randomly.
    """
    def __init__(self,
                 position_limit: Optional[int] = None,
                 initial_cash: Optional[float] = None,
                 leverage_limit: Optional[float] = None,
                 rng: np.random.Generator | None = None,
                 seed: int | None = None,
                 allow_no_action: bool = False
                 ):
        self.memory = 1
        self.num_strategies = 0
        if position_limit in (None, ):
            self.position_limit = None
        else:
            self.position_limit = int(position_limit)
        self.initial_cash = initial_cash
        self.leverage_limit = leverage_limit
        self.rng = rng if rng is not None else(
            np.random.default_rng(seed) if seed is not None else None)
        self.wins = 0
        init_cash = float(initial_cash or 0.0)
        self.cash = init_cash
        self.points = 0.0
        self.wealth = init_cash
        self.position = 0
        self.strategy = -1
        if allow_no_action:
            self.possible_action = [-1, 0, +1]
        else:
            self.possible_action = [-1, +1]
        
        
    def _apply_trade(self, action: int, price:float) -> None:
        """
        Update position, cash and wealth according to history.
        Single source of current position, wealth etc.
        """
        self.position += action
        self.cash += -action * price
        self.wealth = self.position * price + self.cash
    
    def choose_action(self, history) -> int:
        
        if self.rng is not None:
            action = self.rng.choice(self.possible_action)
        else:
            action = np.random.choice(self.possible_action)
        
        if self.position_limit is not None:
            if abs(self.position + action) > self.position_limit:
                action = 0
        
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
        a_i = self._action
        
        self._apply_trade(a_i, price)
        
        return