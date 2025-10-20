#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:57:58 2025

@author: petermillington
"""

from core.player import Player
from core.m_maker import MarketMaker
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
    def __init__(self,
                 num_players=None,
                 memory=None,
                 memory_list=None,
                 num_strategies=2,
                 rounds=1000,
                 payoff_scheme=None,
                 price_init=100,
                 lambda_value=None,
                 market_maker=None,
                 position_limit=None):

        self.num_players = num_players
        self.memory_list = memory_list
        self.memory = memory if memory_list is None else max(memory_list)
        self.num_strategies = num_strategies
        self.rounds = rounds
        self.payoff_scheme = payoff_scheme
        self.history = list(np.random.choice([-1, 1], size=self.memory))
        self.actions = []
        self.players = []
        self.prices = []
        self.price_init = price_init
        self.prices= [self.price_init]
        self.returns = []
        if lambda_value is not None:
            self.lambda_value = lambda_value
        else:
            self.lambda_value = 30 * num_players   # sets standard lambda for all games
        self.market_maker = market_maker
        self.mm = None

        if memory_list is not None:
            # For each value of memory in memory_list, create num_players agents
            for m in memory_list:
                for _ in range(num_players):
                    self.players.append(Player(memory=m,
                                               num_strategies=num_strategies,
                                               position_limit=position_limit)
                                        )
            if len(self.players) % 2 == 0:
                self.players.append(Player(memory=memory_list[0],
                                           num_strategies=num_strategies,
                                           position_limit=position_limit)
                                    )
        else:
            self.players = [Player(memory, num_strategies,
                                   position_limit=position_limit)
                            for _ in range(num_players)]
        self.num_players = len(self.players)
        self.mm= MarketMaker() if market_maker is not None else None

    def _reset_for_run(self):
        self.history = list(np.random.choice([-1, 1], size=self.memory))
        self.prices = [self.price_init]
        self.returns = []
        self.actions = []
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
            p.actions = []
            p.wins_per_round = []
            p.dollar_per_round = []
            p.index_history = []
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

        flow = sum(actions)          # order flow A_t

        if flow == 0:
            flow = np.random.choice([-1,+1])

        self.actions.append(flow)

        eps_t = 0.0  # or np.random.normal(0, self.noise_std)

        if self.mm:
            pos_sum = self.mm.position
        else:
            pos_sum = 0

        r_t =  flow / self.lambda_value - pos_sum / (self.lambda_value) + eps_t
        p_next = self.clip_or_nan(self.prices[-1] * np.exp(r_t))

        self.prices.append(p_next)
        self.returns.append(r_t)
        if self.mm:
            self.mm.update(p_next, flow)

        # Player update
        for p in self.players:
            p.update(self.num_players,
                     flow,
                     self.payoff_scheme,
                     self.prices[-1],
                     self.lambda_value)

        # Update “public info” bit for the MG history. Define it on the same 'flow'
        minority_action = -1 if flow > 0 else 1
        self.history = (self.history + [minority_action])[-self.memory:]

    def run(self):
        """Resets data for each run of n rounds and plays another round"""
        self._reset_for_run()
        for _ in range(self.rounds):
            self.play_round()
