#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:12:48 2025

@author: petermillington
"""
import numpy as np

class StatsRecorder:
    def __init__(self, N, rounds, k_max, record_agent_series=True):
        """
        N      : number of agents
        rounds : number of rounds in the game
        k_max  : max number of strategies across all agents
        record_agent_series : if False, will only store final stats + global series
        """
        self.N = N
        self.rounds = rounds
        self.k_max = k_max
        self.record_agent_series = record_agent_series

        # --- game-level series ---
        self.A_series = np.zeros(rounds, dtype=np.int64)
        self.price_series = np.zeros(rounds + 1, dtype=float)  # include initial price at t=0

        # you can add more global series later: returns, volatility proxies, etc.

        # --- agent-level per-round series (optional, can be big) ---
        if record_agent_series:
            self.wealth = np.zeros((rounds + 1, N), dtype=float)
            self.position = np.zeros((rounds + 1, N), dtype=float)
            self.wins_per_round = np.zeros((rounds + 1, N), dtype=np.int32)
            self.points_per_round = np.zeros((rounds + 1, N), dtype=float)
            self.cash_per_round = np.zeros((rounds + 1, N), dtype=float)

            # store which strategy was actually used each round
            # (or -1 for "not defined / before start")
            self.best_strategy = -np.ones((rounds + 1, N), dtype=np.int16)
        else:
            self.wealth = self.position = self.wins_per_round = self.best_strategy = None

        # final summaries (filled at end)
        self.final_wealth = None
        self.final_wins = None
        self.strategy_switches = None
        self.final_points = None
        self.final_position = None
        self.final_cash = None

    def record_initial_state(self, price, players):
        """Call once at t=0 before the first round."""
        self.price_series[0] = price
        if self.record_agent_series:
            for i, p in enumerate(players):
                self.wealth[0, i] = p.wealth
                self.position[0, i] = p.position
                self.wins_per_round[0, i] = p.wins
                self.points_per_round[0, i] = p.points
                self.cash_per_round[0, i] = p.cash

                # if you have a notion of 'current strategy index', log it; else leave -1

    def record_round(self, t, price, A, players):
        """
        Record after round t has been played and players updated.
        t is 0-based index for rounds (0..rounds-1).
        """
        self.A_series[t] = A
        self.price_series[t + 1] = price

        if self.record_agent_series:
            for i, p in enumerate(players):
                self.wealth[t + 1, i] = p.wealth
                self.position[t + 1, i] = p.position
                self.wins_per_round[t + 1, i] = p.wins
                self.points_per_round[t + 1, i] = p.points
                self.cash_per_round[t + 1, i] = p.cash

                # assume Player exposes 'current_strategy_index' for the chosen strategy this round
                self.best_strategy[t + 1, i] = p.strategy

    def finalize(self, players):
        """Compute final summaries after the last round."""
        self.final_wealth = self.wealth[-1, :].astype(float)
        self.final_wins = np.sum(self.wins_per_round, axis=0).astype(np.int32)
        self.final_points = np.array([p.points for p in players], dtype=float)
        self.final_position = np.array([p.position for p in players], dtype=np.int32)
        self.final_cash = np.array([p.cash for p in players], dtype=float)

        if self.record_agent_series:
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

        # Return everything bundled in a dict for convenience
        return {
            "A_series": self.A_series,
            "price_series": self.price_series,
            "wealth": self.wealth,
            "position": self.position,
            "wins": self.wins_per_round,
            "best_strategy": self.best_strategy,
            "final_wealth": self.final_wealth,
            "final_wins": self.final_wins,
            "final_points": self.final_points,
            "strategy_switches": self.strategy_switches,
        }
