#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:26:15 2026

@author: petermillington

Base interface for all agent types to ensure consistency.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the game.
    
    All agents must implement:
    - choose_action(): decision-making
    - update(): post-round updates
    
    All agents track:
    - position, cash, wealth
    - position_limit enforcement
    """
    
    def __init__(self,
                 position_limit: Optional[int] = None,
                 initial_cash: float = 0.0,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = None):
        """
        Initialize base agent attributes.
        
        Parameters
        ----------
        position_limit : int, optional
            Maximum absolute position allowed
        initial_cash : float
            Starting cash balance
        rng : np.random.Generator, optional
            Random number generator instance
        seed : int, optional
            Seed for RNG if rng not provided
        """
        # Standardize position limit handling
        self.position_limit = int(position_limit) if position_limit not in (None, 0) else None
        
        # Financial state
        self.cash = float(initial_cash)
        self.position = 0
        self.wealth = float(initial_cash)
        
        # RNG
        self.rng = rng if rng is not None else (
            np.random.default_rng(seed) if seed is not None else 
            np.random.default_rng())
        
        # Tracking
        self.cohort_id: Optional[int] = None
        
    @abstractmethod
    def choose_action(self, history: List[int]) -> int:
        """
        Choose action based on game history.
        
        Parameters
        ----------
        history : List[int]
            Game history (last bits for strategic agents)
            
        Returns
        -------
        int
            Action in {-1, 0, +1}
        """
        pass
    
    @abstractmethod
    def update(self, 
               N: int,
               flow: int, 
               price: float,
               lambda_value: float) -> None:
        """
        Update agent state after round completion.
        
        Parameters
        ----------
        N : int
            Total number of agents
        flow : int
            Aggregate market flow
        price : float
            Current price
        lambda_value : float
            Market impact parameter
        """
        pass
    
    def _apply_trade(self, action: int, price: float) -> None:
        """
        Execute trade and update position/cash/wealth.
        
        This is the single source of truth for position changes.
        
        Parameters
        ----------
        action : int
            Trade quantity (signed)
        price : float
            Execution price
        """
        self.position += action
        self.cash -= action * price
        self.wealth = self.position * price + self.cash
    
    def _enforce_position_limit(self, desired_action: int) -> int:
        """
        Enforce position limit by clipping action if necessary.
        
        Parameters
        ----------
        desired_action : int
            Desired action before limit check
            
        Returns
        -------
        int
            Feasible action (possibly clipped to 0)
        """
        if self.position_limit is None:
            return desired_action
        
        if abs(self.position + desired_action) > self.position_limit:
            return 0
        
        return desired_action
    
    @property
    def agent_type(self) -> str:
        """Return agent type for identification."""
        return self.__class__.__name__


class StrategicAgent(BaseAgent):
    """
    Base class for strategic agents (MG, DG, etc.) with strategy banks.
    
    Adds:
    - Strategy management
    - Scoring system
    - Payoff scheme integration
    """
    
    def __init__(self,
                 memory: int,
                 num_strategies: int,
                 payoff: Any,
                 position_limit: Optional[int] = None,
                 initial_cash: float = 0.0,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = None):
        """
        Initialize strategic agent.
        
        Parameters
        ----------
        memory : int
            Number of history bits to use
        num_strategies : int
            Size of strategy bank
        payoff : Any
            Payoff scheme object with get_reward() method
        """
        super().__init__(position_limit, initial_cash, rng, seed)
        
        self.memory = memory
        self.num_strategies = num_strategies
        self.payoff = payoff
        
        # Initialize strategy bank
        self.strategies = self.rng.choice(
            [-1, 1], 
            size=(num_strategies, 2 ** memory))
        
        # Strategy tracking
        self.scores = np.zeros(num_strategies)
        self.points = 0.0
        self.strategy: Optional[int] = None
        self.strategy_switches = 0
        self.wins = 0
        
        # Settlement buffers for delayed payoff
        self._pending: Optional[Dict[str, Any]] = None
        self._prev: Optional[Dict[str, Any]] = None
    
    def choose_action(self, history: List[int]) -> int:
        """
        Choose action using best-performing strategy.
        
        Parameters
        ----------
        history : List[int]
            Global game history
            
        Returns
        -------
        int
            Action in {-1, 0, +1} after position limit enforcement
        """
        # Get relevant history
        h = history[-self.memory:]
        index = int(''.join(['1' if bit == 1 else '0' for bit in h]), 2)
        
        # Get virtual actions from all strategies
        virt_actions = self.strategies[:, index]
        
        # Select best strategy (random tie-breaking)
        best_indices = np.flatnonzero(self.scores == self.scores.max())
        chosen_idx = self.rng.choice(best_indices)
        
        # Track strategy selection
        if self.strategy is not None and chosen_idx != self.strategy:
            self.strategy_switches += 1
        self.strategy = chosen_idx
        
        # Get desired action
        desired_action = int(self.strategies[chosen_idx, index])
        
        # Enforce position limit
        action = self._enforce_position_limit(desired_action)
        
        # Store for settlement
        self._prev = self._pending
        self._pending = {
            "idx": index,
            "virt_actions": virt_actions,
            "chosen_action": action,
            "chosen_idx": chosen_idx
        }
        
        return action
    
    def update(self,
               N: int,
               flow: int,
               price: float,
               lambda_value: float) -> None:
        """
        Update position/wealth and credit strategy scores.
        
        Parameters
        ----------
        N : int
            Number of players
        flow : int
            Aggregate flow
        price : float
            Current price
        lambda_value : float
            Market impact parameter
        """
        # Execute trade from current round
        action = self._pending["chosen_action"]
        self._apply_trade(action, price)
        
        # Determine which round to settle (immediate vs delayed payoff)
        mode = getattr(self.payoff, "mode", "immediate")
        buf = self._pending if mode == "immediate" else self._prev
        
        if buf is None:
            # Nothing to settle (e.g., first round with delayed payoff)
            return
        
        # Get rewards
        a_used = buf["chosen_action"]
        virt = buf["virt_actions"]
        
        reward = self.payoff.get_reward(a_used, flow, N, lambda_value)
        self.points += reward
        
        virt_rewards = self.payoff.get_reward_vector(virt, flow, N, lambda_value)
        self.scores += virt_rewards
        
        self.wins = self.payoff.get_win(a_used, flow)


# Type alias for convenience
Player = StrategicAgent