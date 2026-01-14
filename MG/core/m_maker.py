#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:55:59 2025

@author: petermillington
"""

from typing import List
import numpy as np

class MarketMaker:
    """
    Zero Intelligence market maker, acting as a sink for the aggregate flow.
    The MarketMaker class implies that historical actions impact the price formation.
    The market maker has a calculated position, cash and wealth amount dependent on 
    the actions of the remaining agents.
    
    Attributes
    position: int
        Current inventory (negative of cumulative agent flow)
    cash: float
        cash accumulated from trading with agents
    wealth: float
        market maker pl
    position_history: List[int]
        Position after each round
    cash_history: Lit[float]
        Cash per round
    wealth_history: List[float]
        Wealth at each round
    """
    def __init__(self, initial_position: int = 0, initial_cash: float = 0.0):
        self.position = initial_position
        self.cash = initial_cash
        self.wealth = initial_cash

        self.position_history: List[int] = []
        self.cash_history: List[float] = []
        self.wealth_history: List[float] = []

    def update(self, price: float, flow: int) -> None:
        """
        Update MM state after actions.  MM action is opposed to the aggregate actions
        """
        self.position -= flow
        self.cash += flow * price
        self.wealth = self.cash + self.position * price

        self.position_history.append(self.position)
        self.cash_history.append(self.cash)

        self.wealth_history.append(self.wealth)

    def reset(self) -> None:
        """Reset MM to initial state"""
        self.position = 0
        self.cash = 0.0
        self.wealth = 0.0
        self.position_history = []
        self.cash_history = []
        self.wealth_history = []

    def get_pnl(self) -> float:
        """Get current PL"""
        return self.wealth

    def get_average_position(self) -> float:
        """"Calc average absolute position"""
        if not self.position_history:
            return 0
        return np.mean(np.abs(self.position_history))

    def get_max_position(self) -> int:
        """Maximum absolute inventory over history"""
        if not self.position_history:
            return 0
        return int(np.max(np.abs(self.position_history)))

    @property
    def current_exposure(self) -> int:
        """"Current absolute position"""
        return abs(self.position)

    def __repr__(self) -> str:
        return (f"MarketMaker(position={self.position}, "
                f"cash={self.cash:.2f}, wealth={self.wealth:.2f})")
