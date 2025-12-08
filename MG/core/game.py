#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:57:58 2025

@author: petermillington
"""

from core.player import Player
from core.m_maker import MarketMaker
from core.population_factory import PopulationFactory
from core.players_factory import build_players
from core.game_config import GameConfig
from data.stats_recorder import StatsRecorder
import numpy as np

SAFE_MAX = 1e200
SAFE_MIN = - SAFE_MAX

class Game:
    """
    Either pass (num_players, memory) for homogeneous or 
    pass memory_list=[m1, m2, ....mN]for heterogeneous players
    

    Parameters
    ----------
    num_players : TYPE, optional
        DESCRIPTION. The default is None.
    memory : TYPE, optional
        DESCRIPTION. The default is None.
    memory_list : TYPE, optional
        DESCRIPTION. A list must be entered if a game is being
        played with heterogeneous players
    num_strategies : TYPE, optional
        DESCRIPTION. The default is 2.
    rounds : TYPE, optional
        DESCRIPTION. The default is 1000.
    payoff_scheme : TYPE, optional
        DESCRIPTION. The default is None.
    price init:  TYPE, 
    Returns
    -------
    None.

    """
    def __init__(self, population_spec, cfg, PlayerClass):

        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.players, self.meta, self.cohort_id = build_players(
            population_spec, 
            PlayerClass,
            rng=self.rng,
            shuffle=True,
            per_player_seeds=False,
            master_seed=cfg.seed
            )
        self.n = len(self.players)
        self.rounds = cfg.rounds
        self.lambda_value = cfg.lambda_
        self.price = self.cfg.price
        self.max_memory = max(p.memory for p in self.players)
        self.history=list(np.random.choice([-1, +1], size=self.max_memory))
        self.A = 0
        
        k_max = max(p.num_strategies for p in self.players)
        self.stats = StatsRecorder(
            N = self.n,
            rounds = self.rounds,
            k_max = k_max,
            record_agent_series=True,
            )

        if self.lambda_value is None:
            self.lambda_value = 30 * self.n   # sets standard lambda for all games
        self.market_maker = cfg.mm
        self.mm = None
        
        self.mm= MarketMaker() if self.market_maker is not None else None

    def _reset_for_run(self):
        self.history = list(np.random.choice([-1, 1], size=self.max_memory))
        self.price = self.cfg.price
        self.returns = []
        self.A = 0
        if self.mm:
            self.mm.position = 0
            self.mm.position_per_round = []
            self.mm.cash = 0
            self.mm.cash_per_round = []
            self.mm.wealth = 0
            self.mm.wealth_per_round = []

        for p in self.players:
            p.strategies = np.random.choice([-1, 1], size=(p.num_strategies, 2 ** p.memory))
            p.scores = np.zeros(p.num_strategies)
            p.points = 0
            p.pos_sumition_per_round = []
            p.position_per_round.append(0)
            p.order_per_round = []
            p.cash_per_round = []
            p.cash_per_round.append(0)
            p.strategy_switches = 0
            p.strategy = None
            p.wealth_per_round = []
            p.wealth_per_round.append(0)

    def clip_or_nan(self, x:float) -> float:
        """Ensures that prices do not blow up in large numbers of rounds"""
        if np.isnan(x) or np.isinf(x):
            return np.nan
        if x > SAFE_MAX: return SAFE_MAX
        if x < SAFE_MIN: return SAFE_MIN
        return float(x)

    def play_round(self):
        """Play a single round of a game."""


        actions = [p.choose_action(self.history) for p in self.players]

        self.A = sum(actions)          # order flow A_t

        if self.A == 0:
            self.A = np.random.choice([-1,+1])

        if self.mm:
            pos_sum = self.mm.position
        else:
            pos_sum = 0

        r_t =  self.A * self.lambda_value - pos_sum * (self.lambda_value)
        self.price = self.clip_or_nan(self.price * np.exp(r_t))

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


    def run(self):
        """Resets data for each run of n rounds and plays another round"""
        #self._reset_for_run()
        self.stats.record_initial_state(self.price, self.players)
        for t in range(self.rounds):
            self.play_round()
            self.stats.record_round(t, self.price, self.A, self.players)
            
        results = self.stats.finalize(self.players)
        
        results["cohort_ids"] = np.array(self.cohort_id, dtype=int)
        results["cohorts"] = self.meta["cohorts"]
        
        self.results = results
        
        return results
