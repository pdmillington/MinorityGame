#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:57:58 2025

@author: petermillington
"""

from core.player import Player
from core.m_maker import MarketMaker
import numpy as np

class Game:
    def __init__(self, 
                 num_players=None, 
                 memory=None,
                 memory_list=None,
                 num_strategies=2, 
                 rounds=1000, 
                 payoff_scheme=None,
                 price_init=100,
                 lambda_value=None,
                 market_maker=None):
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

        Returns
        -------
        None.

        """
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
        self.lambda_value = lambda_value
        self.market_maker = market_maker
        self.mm = None
        
        if memory_list is not None:
            # For each value of memory in memory_list, create num_players agents
            for m in memory_list:
                for _ in range(num_players):
                    self.players.append(Player(memory=m, num_strategies=num_strategies))
            if len(self.players) % 2 == 0:
                self.players.append(Player(memory=memory_list[0], num_strategies=num_strategies))
        
        else:
            
                self.players = [Player(memory, num_strategies) for _ in range(num_players)]
        
        self.num_players = len(self.players)
        
        self.mm= MarketMaker() if market_maker is not None else None
        
    def _reset_for_run(self):
        self.history = list(np.random.choice([-1, 1], size=self.memory))
        self.prices = [self.price_init]
        self.returns = []
        self.action = []
        if self.mm:
            self.mm.position = 0
            self.mm.position_by_round = []
            
        for p in self.players:
            p.position = 0
            p.position_history = []
            p.order_history = []
            
            # p.reset_strategy_scores()

    def play_round(self):
        payoff = self.payoff_scheme
        uses_orders = getattr(payoff, "uses_orders", False)
        expects_delta_price = getattr(payoff, "expects_delta_price", False)
        
        desired = [p.choose_action(self.history) for p in self.players]
        
        # Interpret 'desired' as desired positions; compute orders = Δpos
        orders=[]
        for p, d in zip(self.players, desired):
            if uses_orders:
                order = int(d - p.position)   # {-2,-1,0,1,2}
            else:
                order = d
   
            p.position += order
            p.position_history.append(p.position)
            p.order_history.append(order)
            orders.append(order)
        flow = sum(orders)          # order flow A_t
        if flow == 0:
            flow = np.random.choice([-1,+1]) 
        all_actions = orders              # pass to payoff as "actions" for consistency
            
            
        self.actions.append(flow)
        self.returns = []
        
        eps_t = 0.0  # or np.random.normal(0, self.noise_std)
        total_flow = flow - self.mm.position
        r_t =  flow / self.lambda_value - self.mm.position / (3 * self.lambda_value) + eps_t
        p_next = self.prices[-1] * np.exp(r_t)
        delta_price = p_next - self.prices[-1]
        self.prices.append(p_next)
        self.returns.append(r_t)
        self.mm.position -= flow
        self.mm.position_by_round.append(-flow)
        
        

        # 3) Payoffs update (MG: immediate; Dollar: delayed)
        for p in self.players:
            # MG update: (player, all_actions, flow, history_slice)
            # Dollar update expects delta_price as kw; MG will ignore unknown kw
            if expects_delta_price:
                self.payoff_scheme.update(p, all_actions, total_flow, self.history[-p.memory:], self.prices[-1], delta_price=delta_price)
            else:
                self.payoff_scheme.update(p, all_actions, total_flow, self.history[-p.memory:], self.prices[-1])


        # 5) Update “public info” bit for the MG history. Define it on the same 'flow'
        minority_action = -1 if total_flow > 0 else 1
        self.history = (self.history + [minority_action])[-self.memory:]
        

    def run(self):
        self._reset_for_run()
        for _ in range(self.rounds):
            self.play_round()