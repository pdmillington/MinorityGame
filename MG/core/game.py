#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:57:58 2025

@author: petermillington
"""

from typing import List, Dict, Any, Optional
import numpy as np
from core.base_agent import BaseAgent, StrategicAgent
from core.noise_player import NoisePlayer
from core.m_maker import MarketMaker
from core.players_factory import build_players
from core.game_config import GameConfig
from data.stats_recorder import StatsRecorder


SAFE_MAX = 1e200
SAFE_MIN = - SAFE_MAX

class Game:
    """
    Main Game Engine for Minority Game and Dollar Game
    Coordinates agents, market maker and price dynamics
    
    Responsibilities:
        - initialize population from specification
        - Run game rounds
        - update global states (history, price)
        - coordinate agent and MM updates
        - record statistics

    Parameters
    ----------
    population_spec : dict
        Specification for population composition
    cgf : GameConfig
        Game configuration(rounds, lambda, MM, etc.)
    agent_class_map : dict, optional
        Mapping from agent_type str to class
    Returns
    -------
    results: dict
        Dictionary of results

    """
    def __init__(self,
                 population_spec: Dict[str, Any],
                 cfg: GameConfig,
                 agent_class_map: Optional[Dict[str, type]] = None):
        """
        Initialize game with population and configuration
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Default agent types
        if agent_class_map is None:
            agent_class_map = {
                "strategic": StrategicAgent,
                "noise": NoisePlayer
            }
        # Build population
        self.players, self.meta, self.cohort_id = build_players(
            population_spec,
            agent_class_map=agent_class_map,
            rng=self.rng,
            shuffle=True,
            per_player_seeds=False,
            master_seed=cfg.seed
            )
        # Game state
        self.n = len(self.players)
        self.rounds = cfg.rounds
        self.lambda_value = cfg.lambda_ or (30 * self.n)
        self.price = self.cfg.price
        # History
        self.max_memory = max(p.memory for p in self.players)
        self.history=list(self.rng.choice([-1, +1], size=self.max_memory))
        # Aggregate flow
        self.A = 0
        # Market Maker
        self.mm = MarketMaker() if cfg.mm else None
        # Statistics recorder
        k_max = max(p.num_strategies for p in self.players
                    if hasattr(p, 'num_strategies'))
        self.stats = StatsRecorder(
            N = self.n,
            rounds = self.rounds,
            k_max = k_max,
            record_agent_series=cfg.record_agent_series,
            )
        
        # Store results
        self.results: Optional[Dict[str, Any]] = None

    def _reset_for_run(self) -> None:
        """
        Resets all states for a fresh simulation run.
        Automatically called by run() if there are multiple runs.
        """
        #Global state
        self.history = list(self.rng.choice([-1, 1], size=self.max_memory))
        self.price = self.cfg.price
        self.A = 0

        if self.mm:
            self.mm.reset()
        # Reset for all agents
        for p in self.players:
            if isinstance(p, StrategicAgent):
                p.strategies = self.rng.choice(
                    [-1, 1],
                    size=(p.num_strategies, 2 ** p.memory))
                p.scores = np.zeros(p.num_strategies)
                p.strategy = None
                p.strategy_switches = 0
                p._pending = None
                p._prev = None

            p.position = 0
            p.cash = p.initial_cash if hasattr(p, 'initial_cash') else 0.0
            p.wealth = p.cash
            p.points = 0
            p.wins = 0

    def _clip_or_nan(self, x:float) -> float:
        """Ensures that prices do not blow up in large numbers of rounds"""
        if np.isnan(x) or np.isinf(x):
            return np.nan
        if x > SAFE_MAX: return SAFE_MAX
        if x < SAFE_MIN: return SAFE_MIN
        return float(x)

    def play_round(self):
        """
        Play a single round of a game.
        1. Agents choose actions
        2. Calc aggregate flow
        3. Update price
        4. Update market maker if present
        5. Update all agents
        6. Update global history
        """
        # step 1 and step 2
        actions = [p.choose_action(self.history) for p in self.players]
        self.A = sum(actions)          # order flow A_t

        # Random tie break if aggregate flow is zero
        if self.A == 0:
            self.A = self.rng.choice([-1,+1])

        # Price impact
        mm_position = self.mm.position if self.mm else 0
        r_t =  self.A * self.lambda_value - mm_position * (self.lambda_value)
        self.price = self._clip_or_nan(self.price * np.exp(r_t))

        if self.mm:
            self.mm.update(self.price, self.A)

        # Player update
        for p in self.players:
            p.update(self.n,
                     self.A,
                     self.price,
                     self.lambda_value)

        # Update “public info” bit for the MG history. Define it on the same 'flow'
        minority_action = -1 if self.A > 0 else 1
        self.history = (self.history + [minority_action])[-self.max_memory:]


    def run(self, reset: bool = False) -> Dict[str, Any]:
        """
        Run a simulation and reset for a new simulation run,
        
        Parameter
        reset: bool
            If True, resets all states before running
        
        Returns
        dict
            Results dictionary with price series, agent stats etc.
            
        """
        if reset:
            self._reset_for_run()
        # Record initial state
        self.stats.record_initial_state(self.price, self.players)

        # Run all rounds
        for t in range(self.rounds):
            self.play_round()
            self.stats.record_round(t, self.price, self.A, self.players)

        # Finalize stats
        results = self.stats.finalize(self.players)

        # Add metadata
        results["cohort_ids"] = np.array(self.cohort_id, dtype=int)
        results["cohorts"] = self.meta["cohorts"]
        results["config"] = self.cfg

        self.results = results
        return results

    @property
    def is_stable(self, max_price: float= 10.0, min_price: float = 0.1) -> bool:
        """
        Check on stability of prices
        Params
        max_price: float
            Upper limit for price stability
        min_price: float
            Lower limit for price stability
            
        Returns
            True if prices are within limits
        """
        if self.results is None:
            raise ValueError("No results available, run simulation!")

        prices = self.results.get("prices")
        if prices is None:
            return False

        return (prices.max() < max_price and
                prices.min() > min_price and
                not np.any(np.isnan(prices)))

    def time_to_explosion(self, threshold: float = 1000.0) -> int:
        """
        Calculate number of rounds until price explosion
        Parameter
            threshold: price threshold for explosion
        Returns
            Number of rounds until explosion (or total number of rounds if no explosion)
        """
        if self.results is None:
            raise ValueError("No results available.  Please run simulation!")

        prices = self.results.get("prices")
        if prices is None:
            return 0
        
        for t, price in enumerate(prices):
            if price > threshold or price < 1 / threshold:
                return t
        return len(prices)