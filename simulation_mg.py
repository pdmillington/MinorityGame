#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:21:39 2025

@author: petermillington
"""

from MinorityGame2 import MinorityGame  # Replace 'your_module' with the actual module containing MinorityGame

def run_game(args):
    n, M, rounds, S = args
    game = MinorityGame(n, M, rounds, S, False, False)
    game.run()
    return game.get_variance_normalized()