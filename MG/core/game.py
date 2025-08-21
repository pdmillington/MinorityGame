#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:57:58 2025

@author: petermillington
"""

from core.player import Player
import numpy as np

class Game:
    def __init__(self, 
                 num_players=None, 
                 memory=None,
                 memory_list=None,
                 num_strategies=2, 
                 rounds=1000, 
                 payoff_scheme=None,):
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
        

    def play_round(self):
        actions = [p.choose_action(self.history[-p.memory:]) for p in self.players]
        A_t = sum(actions)
        self.actions.append(A_t)

        minority_action = -1 if A_t > 0 else 1
        
        for p, a in zip(self.players, actions):
            self.payoff_scheme.update(p, actions, A_t, self.history[-p.memory:])
            p.wins_per_round.append(1 if a == minority_action else 0)

        # Update global history (minority encoded as 0/1)
        self.history = (self.history + [minority_action])[-self.memory:]

    def run(self):
        for _ in range(self.rounds):
            self.play_round()