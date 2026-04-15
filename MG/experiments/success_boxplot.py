#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 12:22:25 2025

@author: petermillington

Success Boxplot Experiment

Runs a game with heterogeneous cohorts and produces a consolidated report
including success/wealth analysis and information metrics.

Output: Single run directory containing:
    - experiment_report.pdf (consolidated)
    - figures/ (individual PDFs for publication)
    - per_player_metrics.csv
    - params.json, run_info.json
"""

import json
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.game import Game
from core.game_config import GameConfig
from analysis.cohort_utils import (
    group_vector_by_cohort,
    cohort_labels_from_meta,
    group_timeseries_mean_by_cohort,
    summarize_population,
    format_population_summary,
    )
from analysis.plot_utils import (
    create_price_figure,
    create_series_figure,
    create_boxplot_figure,
    create_mean_line_figure,
    create_scatter_figure,
    )
from analysis.population_spec import build_population_spec, CohortConfig, PopulationConfig
from analysis.information_metrics import (
    compute_history_statistics,
    create_information_summary_figure,
    create_information_figure,
    create_rolling_mi_figure,
    )
from analysis.report_builder import ReportBuilder
from utils.logger import RunLogger

# Helper functions
def load_config(path: str) -> PopulationConfig:
    """Load experiment configuration from JSON file"""
    with open(path, "r") as f:
        data =json.load(f)
        config_data = {k: v for k, v in data.items() if not k.startswith('_')}

    cohort_dicts = config_data.get("cohorts") or []
    cohorts = [CohortConfig(**c) for c in cohort_dicts]
    config_data["cohorts"] = cohorts
    return PopulationConfig(**config_data)


#Core experiment logic
def run_cohort_game(
    population_spec: dict,
    cfg: GameConfig,
) -> Dict[str, Any]:
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

    # OPTIONAL: time series per cohort if agent series recorded
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
    """
    Main experiment runner.

    Produces consolidated report with:
    - Population configuration summary
    - Price and attendance series
    - Success and wealth boxplots by cohort
    - Information metrics analysis
    """
    population_spec = build_population_spec(cfg)

    cfg_game = GameConfig(
        rounds=cfg.rounds,
        lambda_=1/(population_spec["total"]*50),
        mm=None,
        price=cfg.price,
        seed=cfg.seed,
        record_agent_series=True
    )

    if population_spec['cohorts']:
        mem = [cohort['memory'] for cohort in population_spec['cohorts']]
        payoffs = [cohort['payoff'] for cohort in population_spec['cohorts']]
    else:
        mem = cfg.m_values
        payoffs = cfg.payoffs

    if isinstance(payoffs, list) and len(payoffs)==1:
        payoff_label = payoffs[0]
    elif isinstance(payoffs, str):
        payoff_label = payoffs
    else:
        payoff_label = "Mixed"

    #Initialize logger
    logger = RunLogger(
        module="HeteroCohorts",
        payoff=payoff_label,
        run_id=None,
        seed=cfg.seed
        )
    logger.log_params({
        "population_spec": population_spec,
        "rounds": cfg.rounds,
        "lambda": cfg_game.lambda_,
        "seed": cfg.seed,
    })

    #Run simulation
    out = run_cohort_game(population_spec, cfg_game)
    results = out['results_raw']
    labels = out['labels']

    # Compute information metrics before building report
    base_m = mem[0] if mem else 4
    if isinstance(mem, (list, tuple)) and len(mem) > 0:
        base_m = max(3, min(mem))

    information_results = []

    for offset in range(3):  # Analyze m, m+1, m+2
        k = base_m + offset
        info = compute_history_statistics(results['Attendance'], window=k)
        information_results.append(info)

    #Inaccessible information analysis
    mi_at_m = information_results[0]['mutual_information']
    mi_at_m1 = information_results[1]['mutual_information']
    mi_delta = mi_at_m1 - mi_at_m
    
    #Log metrics
    logger.log_metrics({
        "mutual_information_at_m": mi_at_m,
        "mutual_information_at_m1": mi_at_m1,
        "inaccessible_information": mi_delta,
        })
    
    #Build consolidated report
    report = ReportBuilder(logger.get_dir())
    
    #Page 1: Population configuration
    pop_summary = format_population_summary(summarize_population(population_spec))
    config_text = (
        f"Experiment: Success Boxplot Analysis\n"
        f"{'=' * 40}\n\n"
        f"Rounds: {cfg.rounds}\n"
        f"Lambda: {cfg_game.lambda_:6f}\n"
        f"Market Maker: {cfg.market_maker}\n"
        f"Seed: {cfg.seed}\n\n"
        f"{'=' * 40}\n"
        f"POPULATION STRUCTURE\n"
        f"{'=' * 40}\n\n"
        f"{pop_summary}")
    report.add_text_page(config_text, title="Experiment Configuration")

    #Price Series
    fig_price = create_price_figure(results["Prices"], title="Price Series")
    report.add_figure(fig_price, "price_series", close_after=True)
    
    # Attendance series
    fig_att = create_series_figure(
        results["Attendance"], 
        title="Attendance Series",
        xlabel="Round",
        ylabel="Attendance A(t)"
    )
    report.add_figure(fig_att, "attendance_series", close_after=True)

    # Success boxplot by cohort
    fig_success_box = create_boxplot_figure(
        out["success_by_cohort"],
        labels,
        title="Success Rate by Cohort",
        ylabel="Success Rate"
    )
    report.add_figure(fig_success_box, "success_boxplot", close_after=True)

    # Wealth boxplot by cohort
    fig_wealth_box = create_boxplot_figure(
        out["wealth_by_cohort"],
        labels,
        title="Final Wealth by Cohort",
        ylabel="Wealth"
    )
    report.add_figure(fig_wealth_box, "wealth_boxplot", close_after=True)

    # Mean success line plot
    fig_success_line = create_mean_line_figure(
        out["success_by_cohort"],
        labels,
        title="Mean Success Rate by Cohort",
        ylabel="Success Rate"
    )
    report.add_figure(fig_success_line, "success_mean_line", close_after=True)

    # Switches vs wealth scatter (if available)
    if out["switches_by_cohort"] is not None:
        all_switches = np.concatenate([out["switches_by_cohort"][cid] for cid in out["switches_by_cohort"]])
        all_wealth = np.concatenate([out["wealth_by_cohort"][cid] for cid in out["wealth_by_cohort"]])
        fig_scatter = create_scatter_figure(
            all_switches, all_wealth,
            title="Strategy Switches vs Final Wealth",
            xlabel="Number of Switches",
            ylabel="Final Wealth"
        )
        report.add_figure(fig_scatter, "switches_vs_wealth", close_after=True)

    #Information metrics analysis
    fig_info_summary = create_information_summary_figure(
        information_results,
        base_m,
        mi_at_m,
        mi_at_m1)
    report.add_figure(fig_info_summary, "Information Summary", close_after=True)
    
    # Rolling MI at k = base_m
    fig_rolling_base = create_rolling_mi_figure(
        results["Attendance"],
        k=base_m,
        roll_window=2000,
        step=100,
        title=f'Rolling MI (k={base_m}, window=2000 rounds) - structure at core agent memory',
        )
    report.add_figure(fig_rolling_base, "rolling_mi_base", close_after=True)
    
    # Rolling MI at k = base_m +1
    fig_rolling = create_rolling_mi_figure(
        results["Attendance"],
        k=base_m + 1,
        roll_window=2000,
        step=100,
        title=f'Rolling MI (k={base_m+1}, window=2000 rounds) - structure beyond core agent memory',
        )
    report.add_figure(fig_rolling, "rolling_mi", close_after=True)
    
    for info in information_results:
        k = info['window']
        fig_info = create_information_figure(info, window=k)   
        report.add_figure(fig_info, f"information_{k}", close_after=True)
    
    
    report_path = report.build()
    
    # tidy per-agent table (now includes cohort_id)
    cohort_ids = out["cohort_ids"]
    rows = []
    
    for i, cid in enumerate(cohort_ids):
        rows.append({
            "agent": i,
            "cohort_id": int(cid),
            "cohort_label": labels.get(int(cid), f"cohort {cid}"),
            "final_wealth": float(results["final_wealth"][i]),
            "final_wins": int(results["final_wins"][i]),
            "success_rate": float(results["final_wins"][i] / cfg.rounds),
            "strategy_switches": (
                int(results["strategy_switches"][i])
                if results.get("strategy_switches") is not None else None
            )
        })
    df_players = pd.DataFrame(rows)
    table_path = logger.log_table(df_players, name="per_player_metrics")
    
    logger.close()
    print(f"Done. Run folder: {logger.get_dir()}")
    print(f"Report: {report_path}")
    print(f"Tables: {table_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run heterogeneous cohort analysis with information metrics"
        )
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
