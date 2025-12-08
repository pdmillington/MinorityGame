#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:46:56 2025

@author: petermillington
"""
from typing import List, Optional, Dict, Any
import numpy as np

class Player:
    """
    Minority Game / $-Game agent with simple strategy selection and basic
    PnL/portfolio accounting.

    The player holds a bank of binary strategies (±1) indexed by the last
    `memory` bits of the global history. Each round they:
      1) Pick an index from the recent history.
      2) Choose among the best-scoring strategies (ties broken randomly).
      3) Produce an action (±1) that is either added to market flow (MG) or
         interpreted as a desired position (in certain $-game variants).
      4) Update position, cash, wealth, and virtual/real strategy scores.

    Attributes
    ----------
    memory : int
        Number of past bits used to index strategies (history length).
    num_strategies : int
        Number of strategies held by the player.
    strategies : np.ndarray
        Shape (num_strategies, 2**memory). Entries are in {-1, +1}.
    scores : np.ndarray
        Shape (num_strategies,). Virtual performance score per strategy.
    points : float
        Accumulated payoff points (from `payoff_scheme.get_reward`).
    position : int
        Current inventory/position (updates by realized action).
    cash : float
        Current cash balance (trade cash flows only).
    strategy_switches : int
        Count of times the chosen strategy index changes.
    strategy : Optional[int]
        Currently selected strategy index.
    wealth : float
        Mark-to-market wealth = position * price + cash (last computed).
    position_limit : Optional[int]
        If set, absolute cap on position; actions that would breach are clipped to 0.
    initial_cash : Optional[float]
        If provided, can be used by caller to seed `cash` externally.
    leverage_limit : Optional[float]
        Reserved for external leverage checks (not enforced here).
    _pending : Optional[Dict[str, Any]]
        Current-round buffer for settlement (idx, virt_actions, chosen_action, chosen_idx).
    _prev : Optional[Dict[str, Any]]
        Previous-round buffer used when payoff settlement is delayed.

    Notes
    -----
    - This class assumes `payoff_scheme` exposes:
        * attribute: `mode` in {"immediate", "delayed"} (default "immediate")
        * methods: `get_reward(a_used, flow, N, lambda_value)` -> float
                   `get_reward_vector(virt, flow, N, lambda_value)` -> np.ndarray
    - Position limits are enforced by zeroing the action when the next position
      would exceed `position_limit`.
    """
    def __init__(self,
                 memory: int,
                 num_strategies: int,
                 payoff: Optional[str] = None,
                 position_limit: Optional[int] = None,
                 initial_cash: Optional[float] = None,
                 leverage_limit: Optional[float] = None,
                 rng: np.random.Generator | None = None,
                 seed: int | None = None
                 ):
        self.memory = memory
        self.num_strategies = num_strategies
        self.payoff = payoff
        if position_limit in (None, 0):
            # Treat both None and 0 as "no limit"
            self.position_limit = None
        else:
            self.position_limit = int(position_limit)
        self.initial_cash = initial_cash
        self.leverage_limit = leverage_limit

        self.rng = rng if rng is not None else (
            np.random.default_rng(seed) if seed is not None else None)

        if self.rng:
            self.strategies = self.rng.choice([-1, 1], size=(num_strategies, 2 ** memory))
        else:
            self.strategies = np.random.choice([-1, 1], size=(num_strategies, 2 ** memory))
        self.scores = np.zeros(num_strategies)

        self.wins = 0

        self.strategy: Optional[int] = None
        self.strategy_switches: int = 0

        init_cash = float(initial_cash or 0.0)
        self.cash = init_cash
        self.points = 0.0
        self.wealth = init_cash
        self.position = 0

        self._pending: Dict[str, Any] = None        #holds data for current round
        self._prev: Dict[str, Any] = None           #holds data for the previous round


    # Helper functions
    def _apply_trade(self, action: int, price: float) -> None:
        """
        Update position, cash and wealth according to history.
        Single source of current position, wealth etc.
        """
        self.position += action
        self.cash += - action * price
        self.wealth = self.position * price + self.cash


    def choose_action(self, full_history: List[int])-> int:
        """
        Choose an action based on the last `memory` bits of `full_history`.

        For standard MG: the returned value is the signed action (±1) that
        contributes to the aggregate flow.
        For $-game variants that interpret actions as desired positions:
        external code may translate action -> order = desired - current position.

        Parameters
        ----------
        full_history : List[int]
            Global binary history with entries in {0, 1} or {-1, +1}.
            Only the last `memory` entries are used.

        Returns
        -------
        int
            The realized action after enforcing position limit (in {-1, 0, +1}).

        Side Effects
        ------------
        - Tracks `strategy_switches` and updates `strategy`.
        - Fills `_pending` with the round’s settlement info.
        """
        h = full_history[-self.memory:]
        index = int(''.join(['1' if bit == 1 else '0' for bit in h]), 2)
        virt_actions = self.strategies[:, index]
        best_strat_idx = np.flatnonzero(self.scores == self.scores.max())
        if self.rng:
            chosen_idx = self.rng.choice(best_strat_idx)
        else:
            chosen_idx = np.random.choice(best_strat_idx)

        # Track switches & strategy history
        self.strategy = chosen_idx

        # Action / desired position from the chosen strategy at this index (±1)
        action = int(self.strategies[chosen_idx, index])

        if self.position_limit is not None:
            if abs(self.position + action) > self.position_limit:
                action = 0

        # Keep current round decision data and data for last round
        self._prev =  self._pending
        self._pending = {
            "idx": index,
            "virt_actions": virt_actions,
            "chosen_action": action,
            "chosen_idx": chosen_idx
            }

        return action

    def update(self,
               N,
               flow,
               price: float,
               lambda_value: float) -> None:
        """
        Settle the current (or previous) round, update inventory and PnL, and
        credit real and virtual strategy scores according to `payoff_scheme`.
        
        Parameters
        ----------
        N : int
            Number of players (for payoff normalization in some schemes).
        flow : int
            Aggregate market flow (e.g., sum of player actions).
        payoff_scheme : Any
            Object with `mode` and reward APIs (see class docstring).
        price : float
            Current price used for cash/wealth updates.
        history : List[int]
            Global history (unused here but included for symmetry/extensibility).
        lambda_value : float
            Market impact / intensity parameter used in rewards.
        
        Notes
        -----
        - Inventory and cash are updated using the *current* round action
          (from `_pending`), regardless of settlement mode.
        - Rewards are applied to `_pending` (immediate) or `_prev` (delayed).
        """

        a_i = self._pending["chosen_action"]

        self._apply_trade(a_i, price)

        mode = getattr(self.payoff, "mode", "immediate")

        # Pick which buffer to credit
        buf = self._pending if mode == "immediate" else self._prev
        if buf is None:
            # nothing to settle this step (e.g., first step for delayed)
            return

        a_used = buf["chosen_action"]
        virt = buf["virt_actions"]  # (S,)
        #print(f"a: {a_used}, flow: {flow}, N: {N}, lamba: {lambda_value}")
        #print(f"{self.payoff}")
        reward = self.payoff.get_reward(a_used, flow, N, lambda_value)
        self.points += reward

        virt_rewards = self.payoff.get_reward_vector(virt, flow, N, lambda_value)
        self.scores += virt_rewards
        self.wins = self.payoff.get_win(a_used, flow)

        return
