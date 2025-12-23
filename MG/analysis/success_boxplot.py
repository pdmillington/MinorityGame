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
import pandas as pd
import argparse
from typing import Optional, Dict, Any, List, Iterable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from core.game import Game
from core.player import Player
from core.game_config import GameConfig
from analysis.cohort_utils import group_vector_by_cohort, cohort_labels_from_meta, group_timeseries_mean_by_cohort
from analysis.plot_utils import (
    plot_metric_boxplot_by_cohort,
    plot_metric_mean_line_by_cohort,
    plot_price_graph)
from analysis.population_spec import build_population_spec, CohortConfig, PopulationConfig
from utils.logger import log_simulation, RunLogger, _ts



# Helper functions
def load_config(path: str) -> PopulationConfig:
    with open(path, "r") as f:
        data =json.load(f)

    cohort_dicts = data.get("cohorts") or []
    cohorts = [CohortConfig(**c) for c in cohort_dicts]
    data["cohorts"] = cohorts
    return PopulationConfig(**data)

def plot_switch_vs_points_corr():
    pass

def run_cohort_game(
    population_spec: dict,
    cfg: GameConfig,
):
    """
    Run a cohort-based game and return cohort-grouped metrics.
    """
    game = Game(population_spec=population_spec, cfg=cfg)
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



def run_success_boxplot(cfg: PopulationConfig):


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

    out = run_cohort_game(population_spec, cfg_game)

    prefix = logger.run_info["start_time"]
    labels = out["labels"]

    box_path = plot_metric_boxplot_by_cohort(
        out["success_by_cohort"],
        labels,
        title="Success by Cohort",
        ylabel="Success rate",
        save_dir="plots/success",
        filename_prefix=prefix,
        filename_stub="success_boxplot"
    )

    wealth_path = plot_metric_boxplot_by_cohort(
        out["wealth_by_cohort"],
        labels,
        title="Wealth by Cohort",
        ylabel="Final wealth",
        save_dir="plots/success",
        filename_prefix=prefix,
        filename_stub="wealth_boxplot"
    )

    mean_path = plot_metric_mean_line_by_cohort(
        out["success_by_cohort"],
        labels,
        title="Average Success by Cohort",
        ylabel="Average success rate",
        save_dir="plots/success",
        filename_prefix=prefix,
        filename_stub="success_mean_line"
    )

    price_path = plot_price_graph(
        out["results_raw"]["Prices"],
        title="Price Plot",
        ylabel="Price",
        save_dir="plots/success",
        filename_prefix=prefix,
        filename_stub="price"
    )

    logger.log_artifact(box_path)
    logger.log_artifact(wealth_path)
    logger.log_artifact(mean_path)
    logger.log_artifact(price_path)

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
