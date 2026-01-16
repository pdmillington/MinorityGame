#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 18:19:23 2025

@author: petermillington
"""

import sys
import os
import json
from dataclasses import dataclass
import argparse
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
from core.game import Game
from core.game_config import GameConfig
from utils.logger import log_simulation

@dataclass
class SuccessEvolutionConfig:
    """
    Dataclass for success evolution, allowing set up of JSON file to input 
    configuration details for this experiment.
    """
    payoff_key: str
    m_value: int = 5
    num_players: int = 301
    num_strategies: int = 2
    position_limit: int = 0     #0 means no limit
    rounds: int = 10_000
    intervals: Iterable[int] = None
    market_maker: bool | None = None
    price: float = 100
    record_agent_series: bool = False
    save_path: str = "plots/success/success_evolution.pdf"

def load_config(path: str) -> SuccessEvolutionConfig:
    with open(path, 'r') as f:
        data = json.load(f)
        config_data = {k: v for k, v in data.items() if not k.startswith('_')}
    return SuccessEvolutionConfig(**config_data)

def plot_success_evolution(cfg):
    """
    Takes output from game (usually lengthy) and plots moving averages of success
    rates.
    """
    avg_success_rates = []
    rounds = cfg.rounds

    population_spec = {
        "total": cfg.num_players,
        "cohorts": [
            {
                "count": cfg.num_players,
                "memory": cfg.m_value,
                "payoff": cfg.payoff_key,
                "strategies": cfg.num_strategies,
                "position_limit": cfg.position_limit,
                }
            ]
        }

    cfg_game = GameConfig(
        rounds=cfg.rounds,
        lambda_=1/(cfg.num_players * 50),
        mm=cfg.market_maker,
        price=cfg.price,
        record_agent_series=cfg.record_agent_series
        )

    # Success evolution demo run
    long_game = Game(
        population_spec=population_spec,
        cfg=cfg_game
    )

    results = long_game.run()
    wins = np.array(results["wins"])

    intervals = cfg.intervals

    for interval in intervals:
        # ignore incomplete final interval
        full_intervals = rounds // interval
        wins = wins[:full_intervals * interval]
        avg_wins = np.mean(wins, axis=1)
        reshaped = avg_wins.reshape(-1, interval)
        interval_means = reshaped.mean(axis=1)
        avg_success_rates.append((interval, interval_means))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    for interval, rates in avg_success_rates:
        ax.plot(np.arange(len(rates)) * interval, rates, label=f"Δt={interval}")

    ax.set_title("Average Success Rate over Time")
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg. Success Rate")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.save_path, format="pdf", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file for the success evolution experiment")

    args = parser.parse_args()

    cfg = load_config(args.config)
    plot_success_evolution(cfg)

if __name__ == "__main__":
    main()
