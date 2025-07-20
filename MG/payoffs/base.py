#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 21:11:03 2025

@author: petermillington
"""

class PayoffScheme:
    def update(self, player, all_actions, total_action, history):
        raise NotImplementedError("Must be implemented by subclass.")