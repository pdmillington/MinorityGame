#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:12:48 2025

@author: petermillington
"""
import numpy as np
from typing import Dict, Any, Optional

class StatsRecorder:
    def __init__(self, N, rounds, k_max,
                 record_agent_series=True,
                 record_strategies=False):
        """
        N      : number of agents
        rounds : number of rounds in the game
        k_max  : max number of strategies across all agents
        record_agent_series : if False, will only store final stats + global series
        record_strategies : if True, extract strategy tables at finalize
        """
        self.N = N
        self.rounds = rounds
        self.k_max = k_max
        self.record_agent_series = record_agent_series
        self.record_strategies = record_strategies

        # --- game-level series ---
        self.attendance = np.zeros(rounds, dtype=np.int64)
        self.prices = np.zeros(rounds + 1, dtype=float)  # include initial price at t=0

        # add more global series later: returns, volatility proxies, etc.

        # --- agent-level per-round series (optional, can be big) ---
        if record_agent_series:
            # Position and wins from fast arrays
            # Wealth and cash reconstructed in finalize
            self.position = np.zeros((rounds + 1, N), dtype=float)
            self.wins_per_round = np.zeros((rounds + 1, N), dtype=np.int32)
            self.points_per_round = np.zeros((rounds + 1, N), dtype=float)
            self.best_strategy = -np.ones((rounds + 1, N), dtype=np.int16)

            self.wealth = np.zeros((rounds + 1, N), dtype=float)
            self.cash = np.zeros((rounds + 1, N), dtype=float)
        else:
            self.position = None
            self.wins_per_round = None
            self.points_per_round = None
            self.best_strategy = None
            self.wealth = None
            self.cash = None

        # final summaries (filled at end)
        self.final_wealth = None
        self.final_wins = None
        self.strategy_switches = None
        self.final_points = None
        self.final_position = None
        self.final_cash = None

    def record_initial_state(self, price: float, players,
                             position_arr: Optional[np.ndarray] = None,
                             wins_arr: Optional[np.ndarray] = None,
                             points_arr: Optional[np.ndarray] = None,
                             initial_cash_arr: Optional[np.ndarray] = None):
        """
        Call once at t=0 before the first round.
        Fast path: pass fast arrays from Game
        Slow path: falls back to reading from player objects
        """
        self.prices[0] = price
        if self.record_agent_series:
            if position_arr is not None:
                self.position[0, :] = position_arr
                self.wins_per_round[0, :] = wins_arr if wins_arr is not None else 0
                self.points_per_round[0, :] = points_arr if points_arr is not None else 0.0
                if initial_cash_arr is not None:
                    self.cash[0, :] = initial_cash_arr
                self.wealth[0, :] = self.cash[0, :]
            else:
                # Slow path
                for i, p in enumerate(players):
                    self.position[0, i] = p.position
                    self.wins_per_round[0, i] = p.wins
                    self.points_per_round[0, i] = p.points
                    self.cash[0, i] = p.cash
                    self.wealth[0, i] = p.wealth


    def record_round(self, t: int, price: float, A: int, players,
                     position_arr: Optional[np.ndarray] = None,
                     wins_arr: Optional[np.ndarray] = None,
                     points_arr: Optional[np.ndarray] = None,
                     best_strategy_arr: Optional[np.ndarray] = None):
        """
        Record after round t has been played.
        
        Fast path: pass arrays from Game fast arrays — no per-agent Python loop.
        Slow path: falls back to reading from player objects (original behaviour).
 
        Parameters
        ----------
        t               : 0-based round index
        price           : current price (scalar)
        A               : aggregate flow (scalar)
        players         : list of agent objects (used in slow path only)
        position_arr    : (N,) int32  current positions  [fast path]
        wins_arr        : (N,) int32  cumulative wins     [fast path]
        points_arr      : (N,) float  cumulative points   [fast path]
        best_strategy_arr: (N,) int16 chosen strategy idx [fast path]
        """
        self.attendance[t] = A
        self.prices[t + 1] = price

        if not self.record_agent_series:
            return

        if position_arr is not None:
            # Fast path
            self.position[t + 1, :] = position_arr
            self.wins_per_round[t + 1, :]  = wins_arr
            self.points_per_round[t + 1,:] = points_arr
            if best_strategy_arr is not None:
                self.best_strategy[t + 1, :] = best_strategy_arr
        else:
            # Slow path
            for i, p in enumerate(players):
                self.position[t + 1, i] = p.position
                self.wins_per_round[t + 1, i] = p.wins
                self.points_per_round[t + 1, i] = p.points
                self.best_strategy[t + 1, i] = p.strategy
                self.wealth[t + 1, i] = p.wealth
                self.cash[t + 1, i] = p.cash

    # Wealth reconstruction
    def _reconstruct_wealth_cash(self, initial_cash_arr: Optional[np.ndarray] = None):
        """
        Reconstruct per agent wealth and cash time series from position and prices
        
        Called by finalized when fast path was used (wealth not recorded per round).
        
        Cash update:  cash[t] = cash[t-1] - delta_q[t] * price[t]
        Wealth:       wealth[t] = cash[t] + position[t] * price[t]
 
        Parameters
        ----------
        initial_cash_arr : (N,) float, optional
            Starting cash per agent. Defaults to zero.
        """
        N = self.N
        rounds = self.rounds

        if initial_cash_arr is not None:
            self.cash[0, :] = initial_cash_arr
        # else cash has already been set or is equal to zero

        prices = self.prices
        pos = self.position

        for t in range(1, rounds + 1):
            delta_q = pos[t, :] - pos[t - 1, :]
            self.cash[t, :] = self.cash[t - 1, :] - delta_q * prices[t]
            self.wealth[t, :] = self.cash[t, :] + pos[t, :] * prices[t]

    def _extract_strategies(self, players) -> Dict[str, Any]:
        """
        Extract strategy data from players at end of game.
        
        Returns dict with:
        - strategies: array of strategy tables
        - best_strategy_ids: which strategy each agent used
        - memory: int, memory parameter m
        """
        # Determine memory size from first strategic agent
        memory = None
        for p in players:
            if hasattr(p, 'strategies') and hasattr(p, 'memory'):
                memory = p.memory
                break
        
        if memory is None:
            # No strategic agents, return empty
            return {
                "strategies": None,
                "best_strategy_ids": None,
                "memory": None
            }
        memory_size = 2 ** memory
        strategies = np.zeros((self.N, memory_size), dtype=np.int8)  # ±1 fits in int8
        best_ids = np.zeros(self.N, dtype=np.int16)
        
        # Extract best strategy for each agent
        for i, p in enumerate(players):
            if hasattr(p, 'strategies') and hasattr(p, 'strategy'):
                # Get the best strategy
                best_idx = p.strategy if p.strategy is not None else 0
                strategies[i, :] = p.strategies[best_idx, :]
                best_ids[i] = best_idx
        
        return {
            "strategies": strategies,  # Shape: (N, 2^m)
            "best_strategy_ids": best_ids,
            "memory": memory
        }

    def finalize(self, players,
                 fast_path: bool = False,
                 initial_cash_arr: Optional[np.ndarray] = None):
        """
        Compute final summaries after the last round.
        
        Parameters
        ----------
        players          : list of agent objects
        fast_path        : if True, reconstruct wealth/cash from position+prices
                           (used when Game ran with vectorised play_round)
        initial_cash_arr : (N,) float, starting cash per agent [fast path only]
        """
        if fast_path and self.record_agent_series:
            self._reconstruct_wealth_cash(initial_cash_arr)
        
        if self.wealth is not None:
            self.final_wealth = self.wealth[-1, :].astype(float)
        else:
            self.final_wealth = np.array([p.wealth for p in players], dtype=float)

        if self.wins_per_round is not None:
            self.final_wins = self.wins_per_round[-1, :].astype(np.int32)
        else:
            self.final_wins = np.array([p.wins for p in players], dtype=np.int32)

        if self.points_per_round is not None:
            self.final_points = self.points_per_round[-1, :].astype(float)
        else:
            self.final_points = np.array([p.points for p in players], dtype=float)

        if self.position is not None:
            self.final_position = self.position[-1,:].astype(np.int32)
        else:
            self.final_position = np.array([p.position for p in players], dtype=np.int32)

        if self.cash is not None:
            self.final_cash = self.cash[-1, :].astype(float)
        else:
            self.final_cash = np.array([p.cash for p in players], dtype=float)

        if self.record_agent_series and self.best_strategy is not None:
            # number of times the chosen strategy changed between t and t+1
            # strategy_switches[i] = count of switches for agent i
            # ignore -1 sentinel at t=0
            bs = self.best_strategy
            # shape (rounds, N): differences between consecutive rounds
            diff = bs[1:, :] - bs[:-1, :]
            # treat transitions involving -1 as no switch (or mask them out)
            valid = (bs[1:, :] >= 0) & (bs[:-1, :] >= 0)
            switches = (diff != 0) & valid
            self.strategy_switches = switches.sum(axis=0).astype(np.int32)
        else:
            # Extract directly from players if attribute exists
            if hasattr(players[0], 'strategy_switches'):
                self.strategy_switches = np.array(
                    [p.strategy_switches for p in players], dtype=np.int32
                    )
            else:
                self.strategy_switches = None
                
        if self.record_strategies:
            extracted = self._extract_strategies(players)
        else:
            extracted = {
                "strategies": None,
                "best_strategy_ids": None,
                "memory": None
                }

        # Return everything bundled in a dict for convenience
        return {
            "Attendance": self.attendance,
            "Prices": self.prices,
            "wealth": self.wealth,
            "cash": self.cash,
            "position": self.position,
            "wins": self.wins_per_round,
            "points": self.points_per_round,
            "best_strategy": self.best_strategy,
            "final_wealth": self.final_wealth,
            "final_wins": self.final_wins,
            "final_points": self.final_points,
            "final_cash": self.final_cash,
            "final_position": self.final_position,
            "strategy_switches": self.strategy_switches,
            "strategies": extracted["strategies"],
            "best_strategy_ids": extracted["best_strategy_ids"],
            "memory": extracted['memory']
        }

