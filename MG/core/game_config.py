#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:21:51 2025

@author: petermillington
"""
# game_config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class MarketMakerConfig:
    """
    Future development: market maker statistics could be added here in case 
    we want to define that more carefully with bid-offer spds etc.
    """
    base_spread: float = 0.0
    inv_aversion: float = 0.0
    quote_depth: float = 1.0

@dataclass(frozen=True)
class GameConfig:
    """
    Configuration of global non-player variables for the game
    Rounds: number of rounds to play
    lambda_ is the lambda value for use in price calculations
    mm: currently None or True, but could be extended to MarketMakerConfig
    seed: for rng
    record_every: cadence for stats of players
    panel_size: when large numbers of players, rounds use panel to reduce data
    """
    rounds: int
    lambda_: float = None              # impact / price update parameter
    mm: bool = None                    # None
    price: float = 100                 # Initial Price for game
    seed: int | None = None
    record_every: int = 1              # cohort stats cadence
    panel_size: int = 200
    record_agent_series: bool = True
