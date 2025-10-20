#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:55:59 2025

@author: petermillington
"""

import numpy as np

SAFE_MAX = 1e200
SAFE_MIN = - SAFE_MAX

class MarketMaker:
    """
    The MarketMaker class implies that historical actions impact the price formation.
    The market maker has a calculated position, cash and wealth amount dependent on 
    the actions of the remaining agents.
    """
    def __init__(self):
        self.position = 0
        self.position_per_round = []
        self.cash = 0
        self.cash_per_round =[]
        self.wealth = 0
        self.wealth_per_round=[]

    def update(self, price, flow):
        self.position -= flow
        self.position_per_round.append(self.position)
        self.cash += flow * price
        self.cash_per_round.append(self.cash)
        self.wealth = self.cash + self.position * price
        self.wealth_per_round.append(self.wealth)
