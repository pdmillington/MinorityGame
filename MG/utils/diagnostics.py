#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 12:09:36 2025

@author: petermillington
"""

from dataclasses import dataclass

@dataclass
class RoundRecord:
    t: int
    flow: int
    price: float
    r_t: float 
    mm_pos: int 
    stuck: bool 
    
class Trace:
    def __init__(self, enabled=False, max_rows=10,000):
        self.enabled = enabled
        self.rows = []
        self.max_rows = max_rows
    def log(self, row: RoundRecord):
        if self.enabled and len(self.rows)<self.max_rows:
            self.rows.append(row)