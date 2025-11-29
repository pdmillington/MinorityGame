#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 12:22:25 2025

@author: petermillington

A game is run with varied memories for the agents.  Parameters for the game and 
agents are defined below.  Two plots measuring success by agent type (memory)
are produced and saved with logging details.
"""

import sys
import os
import json
import argparse
from typing import Optional, Dict, Any, List, Iterable
from dataclasses import dataclass, field
import itertools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.game import Game
from core.player import Player
from core.game_config import GameConfig
from analysis.cohort_utils import group_vector_by_cohort, cohort_labels_from_meta, group_timeseries_mean_by_cohort
from utils.logger import log_simulation, RunLogger, _ts, _ensure_dir

@dataclass
class CohortConfig:
    count: int
    memory: int
    strategies: int
    payoff: str
    position_limit: int = 0

@dataclass
class SuccessBoxplotCohortConfig:
    """
    Dataclass for plotting success by cohort for heterogeneous memory or strategies
    allowing JSON file input of experiment configuration.
    """
    payoffs: Iterable[str]|str
    num_players_per_cohort: int
    m_values: Iterable[int]|int
    s_values: Iterable[int]|int
    position_limit: int = 0
    rounds: int = 1_000
    market_maker: bool | None = None
    price: float = 100
    record_agent_series: bool = True
    save_dir: str = "plots/success"
    mode: str = "cartesian"   #or "zip"
    cohorts: List[CohortConfig] = field(default_factory=list)
    seed: str = 1234


# Helper functions
def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    elif x is None:
        return []
    else:
        return[x]
    
def load_config(path: str) -> SuccessBoxplotCohortConfig:
    with open(path, "r") as f:
        data =json.load(f)

    cohort_dicts = data.get("cohorts") or []
    cohorts = [CohortConfig(**c) for c in cohort_dicts]
    data["cohorts"] = cohorts
    return SuccessBoxplotCohortConfig(**data)

def make_hetero_population_spec(
    m_values,
    num_players_per_cohort,
    payoffs,
    s_values,
    position_limit=0
):
    cohorts = []
    
    m_values = ensure_list(m_values)
    s_values = ensure_list(s_values)
    payoffs = ensure_list(payoffs)
    
    all_cohorts = list(itertools.product(m_values, s_values, payoffs))
    
    if num_players_per_cohort % 2 == 0:
        add_players = 1
    else:
        add_players = 0
    
    for idx, (m, s, p) in enumerate(all_cohorts):
        
        cohorts.append({
            "count": num_players_per_cohort + (add_players if idx == 0 else 0),
            "memory": m,
            "strategies": s,
            "payoff": p,
            "position_limit": position_limit
        })
        
    
    return {
        "total": num_players_per_cohort * len(all_cohorts) + add_players,
        "cohorts": cohorts
    }

# Top level builder of population
def build_population_spec(pop_cfg: SuccessBoxplotCohortConfig) -> dict:
    if pop_cfg.cohorts:
        cohorts = []
        for c in pop_cfg.cohorts:
            cohorts.append({
                "count":c.count,
                "memory": c.memory,
                "payoff": c.payoff,
                "strategies": c.strategies,
                "position_limit": c.position_limit,
                })
        total = sum(c["count"] for c in cohorts)
        return {"total": total, "cohorts": cohorts}
    if pop_cfg.mode == "cartesian":
        return make_hetero_population_spec(
            m_values=pop_cfg.m_values,
            num_players_per_cohort=pop_cfg.num_players_per_cohort,
            payoffs=pop_cfg.payoffs,
            s_values=pop_cfg.s_values,
            position_limit=pop_cfg.position_limit)
# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_metric_boxplot_by_cohort(
    metric_by_cohort: dict,
    labels: dict,
    title: str,
    ylabel: str,
    save_dir="plots/success",
    filename_prefix=None,
    filename_stub="metric_boxplot"
):
    _ensure_dir(save_dir)

    cids = sorted(metric_by_cohort.keys())
    data = [metric_by_cohort[cid] for cid in cids]
    tick_labels = [labels.get(cid, f"cohort {cid}") for cid in cids]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=tick_labels)
    plt.title(title)
    plt.xlabel("Cohort")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    bits = [filename_stub]
    if filename_prefix: bits.insert(0, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_metric_mean_line_by_cohort(
    metric_by_cohort: dict,
    labels: dict,
    title: str,
    ylabel: str,
    save_dir="plots/success",
    filename_prefix=None,
    show_std_band=True,
    filename_stub="metric_mean_line"
):
    _ensure_dir(save_dir)

    cids = sorted(metric_by_cohort.keys())
    means = np.array([np.mean(metric_by_cohort[cid]) for cid in cids], float)
    stds  = np.array([np.std(metric_by_cohort[cid])  for cid in cids], float)

    x = np.arange(len(cids))
    xticks = [labels.get(cid, f"cohort {cid}") for cid in cids]

    plt.figure(figsize=(10, 6))
    plt.plot(x, means, marker="o")
    if show_std_band:
        plt.fill_between(x, means - stds, means + stds, alpha=0.2)

    plt.xticks(x, xticks, rotation=30, ha="right")
    plt.title(title)
    plt.xlabel("Cohort")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    bits = [filename_stub]
    if filename_prefix: bits.insert(0, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    return path

def plot_switch_vs_points_corr():
    pass

def run_cohort_game(
    population_spec: dict,
    cfg: GameConfig,
    PlayerClass=Player
):
    """
    Run a cohort-based game and return cohort-grouped metrics.
    """
    game = Game(population_spec=population_spec, cfg=cfg, PlayerClass=PlayerClass)
    results = game.run()

    # cohort ids and cohort labels
    cohort_ids = results.get("cohort_ids", game.cohort_id)
    labels = cohort_labels_from_meta(game.meta)

    # --- per-agent finals from results ---
    final_wealth = results["final_wealth"]
    final_wins   = results["final_wins"]

    # success rate per agent (cumulative wins / rounds)
    rounds = cfg.rounds
    success_rate = final_wins / max(1, rounds)

    # strategy switches per agent (already derived in recorder)
    switches = results.get("strategy_switches", None)

    # group finals by cohort
    success_by_cohort = group_vector_by_cohort(success_rate, cohort_ids)
    wealth_by_cohort  = group_vector_by_cohort(final_wealth, cohort_ids)
    switches_by_cohort = (
        group_vector_by_cohort(switches, cohort_ids) if switches is not None else None
    )

    # OPTIONAL: time series per cohort if you recorded agent series
    wealth_ts_by_cohort = None
    if results.get("wealth", None) is not None:
        wealth_ts_by_cohort = group_timeseries_mean_by_cohort(results["wealth"], cohort_ids)

    return {
        "results_raw": results,
        "cohort_ids": cohort_ids,
        "labels": labels,
        "success_by_cohort": success_by_cohort,
        "wealth_by_cohort": wealth_by_cohort,
        "switches_by_cohort": switches_by_cohort,
        "wealth_ts_by_cohort": wealth_ts_by_cohort,
    }



def run_success_boxplot(cfg: SuccessBoxplotCohortConfig):


    population_spec = build_population_spec(cfg)

    cfg_game = GameConfig(
        rounds=cfg.rounds,
        lambda_=1/(population_spec["total"]*50),
        mm=None,
        price=cfg.price,
        seed=cfg.seed,
        record_agent_series=True
    )
    
    if len(cfg.payoffs)==1:
        payoffs = cfg.payoffs[0]
    else:
        payoffs = "Mixed"

    logger = RunLogger(module="HeteroCohorts", payoff=payoffs, run_id=None, seed=cfg.seed)
    logger.log_params({
        "population_spec": population_spec,
        "rounds": cfg.rounds,
        "seed": cfg.seed,
    })

    out = run_cohort_game(population_spec, cfg_game, PlayerClass=Player)

    prefix = logger.run_info["start_time"]
    labels = out["labels"]

    box_path = plot_metric_boxplot_by_cohort(
        out["success_by_cohort"],
        labels,
        title=f"Success by Cohort",
        ylabel="Success rate",
        save_dir="plots/success",
        filename_prefix=prefix,
        filename_stub="success_boxplot"
    )

    wealth_path = plot_metric_boxplot_by_cohort(
        out["wealth_by_cohort"],
        labels,
        title=f"Wealth by Cohort",
        ylabel="Final wealth",
        save_dir="plots/success",
        filename_prefix=prefix,
        filename_stub="wealth_boxplot"
    )

    mean_path = plot_metric_mean_line_by_cohort(
        out["success_by_cohort"],
        labels,
        title=f"Average Success by Cohort",
        ylabel="Average success rate",
        save_dir="plots/success",
        filename_prefix=prefix,
        filename_stub="success_mean_line"
    )

    logger.log_artifact(box_path)
    logger.log_artifact(wealth_path)
    logger.log_artifact(mean_path)

    # tidy per-agent table (now includes cohort_id)
    results_raw = out["results_raw"]
    cohort_ids = out["cohort_ids"]

    rows = []
    for i, cid in enumerate(cohort_ids):
        rows.append({
            "agent": i,
            "cohort_id": int(cid),
            "cohort_label": labels.get(int(cid), f"cohort {cid}"),
            "final_wealth": float(results_raw["final_wealth"][i]),
            "final_wins": int(results_raw["final_wins"][i]),
            "success_rate": float(results_raw["final_wins"][i] / cfg.rounds),
            "strategy_switches": (
                int(results_raw["strategy_switches"][i])
                if results_raw.get("strategy_switches") is not None else None
            )
        })
    df_players = pd.DataFrame(rows)
    table_path = logger.log_table(df_players, name="per_player_metrics")

    """
    log_simulation([
        f"Module: HeteroCohorts",
        f"Payoff: {payoffs}",
        f"Cohorts: {population_spec['cohorts']}",
        f"Rounds: {rounds}",
        f"Seed: {seed}",
        f"Saved plot: {box_path}",
        f"Saved plot: {mean_path}",
        f"Saved plot: {wealth_path}",
        f"Run dir: {logger.get_dir()}",
        f"Per-player CSV: {table_path}",
    ])
    """
    logger.close()
    print(f"Done. Run folder: {logger.get_dir()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file for the success boxplot experiment.",
        )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_success_boxplot(cfg)


if __name__ == "__main__":
    main()
