#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:55:59 2025

@author: petermillington
"""

import numpy as np

class MarketMaker:
    def __init__(self):
        self.position = 0
        self.position_by_round = []
        self.cash = 0
        self.cash_per_round =[]
        self.wealth = 0
        self.wealth_per_round=[]