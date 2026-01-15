#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:43:46 2025

@author: petermillington

Plot Attendance Analysis Module
- Runs a generalised MG (dollar game to be checked) for various values of m and s.
- Outputs plots of attendance time series, attendance distribution, points distribution
and success rate distribution.
- Reduced logging capability 
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join
                                (os.path.dirname(__file__), '..')))  #ensure modules are accessible
import json
import argparse

import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from core.game import Game
from core.game_config import GameConfig
from analysis.population_spec import build_population_variant, CohortConfig, PopulationConfig, PopulationFamilyConfig
from analysis.plot_utils import make_population_grid, plot_series, plot_hist, plot_scatter, plot_box
from analysis.cohort_utils import group_vector_by_cohort
from utils.logger import RunLogger, log_simulation  # Moved logging utility to reusable utils
from typing import Optional, Iterable, Dict, Any, List, Tuple

def load_config(path:str) -> PopulationFamilyConfig:
    with open(path, "r") as f:
        data = json.load(f)

    fam_cfg = PopulationFamilyConfig(**data)
    game_cfg = GameConfig(
        rounds=fam_cfg.rounds,
        lambda_=fam_cfg.lambda_,
        mm=fam_cfg.market_maker,
        price=fam_cfg.price,
        record_agent_series=fam_cfg.record_agent_series,
        )
    return fam_cfg, game_cfg


# Helper to calculate risk and return from wealth data.
def returns_from_prices(prices):
    """
    Return vector from a given vector of prices
    """
    prices = np.asarray(prices)

    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations")

    if np.any(prices <= 0):
        raise ValueError("Prices must be positive for log returns")

    returns = np.diff(np.log(prices))

    returns = returns[~np.isnan(returns)]
    return returns

def autocorr_lag(r: np.ndarray, lag: int = 1) -> float:
    """
    Generates scalar autocorrelation value for a given lag
    """
    if r.size <= lag:
        return np.nan
    a = r[lag:]
    b = r[:-lag]
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt((a0 @ a0) * (b0 @ b0))
    return float((a0 @ b0) / denom) if denom > 0 else np.nan

def risk_from_wealth(wealth):
    """
    Calculates average return and risk for a wealth series
    """
    daily_change = np.diff(wealth)
    avg_return = np.mean(daily_change)
    risk = np.var(daily_change)
    return avg_return, risk

# Main

def run_population_family(family_cfg, game_cfg):
    """
    Runs games for various cohorts.  A dictionary captures the output for these 
    games. 
    """
    populations = [build_population_variant(family_cfg, v) for v in family_cfg.values]
    results = []

    for v, pop_spec in zip(family_cfg.values, populations):
        game = Game(
            population_spec=pop_spec,
            cfg=game_cfg
        )
        res = game.run()
        results.append((v, res))

    return results, populations

def make_family_legend_page(fam_cfg: PopulationFamilyConfig, populations):
    """
    Intro text page to describe the different families
    """
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    lines = []
    COHORT_KEY_ORDER = ['count', 'payoff', 'memory', 'strategies', 'position_limit']

    lines.append(f"Rounds: {fam_cfg.rounds}")
    lines.append(f"Market Maker: {fam_cfg.market_maker}")
    lines.append(f"Lambda Value: {fam_cfg.lambda_:.3f}")

    for idx, population in enumerate(populations):
        line = f"Population {idx+1} Total count = {population['total']}"
        lines.append(line)
        for idy, cohort in enumerate(population['cohorts']):
            parts = []
            for key in COHORT_KEY_ORDER:
                if key in cohort:
                    parts.append(f"{key} = {cohort[key]}")

            lines.append(f" Cohort{idy + 1}: "+", ".join(parts))

    text = "\n".join(lines) if lines else "Family configuration not documented."

    ax.text(
        0.5, 0.5,
        text,
        ha="center",
        va="center",
        wrap=True,
    )

    fig.suptitle("Population Families", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def build_family_report(fam_cfg: PopulationFamilyConfig,
                        game_cfg: GameConfig,
                        metric_specs: dict
                        ) -> str:
    """
    Run the family, generate plots, write into a single PDF:
    - page 1 descriptions of families
    - remaining pages: one metric plot per page
    """
    logger = RunLogger(module="PopulationFamily", run_id =None, seed=fam_cfg.seed)
    logger.log_config(fam_cfg)

    results, families = run_population_family(fam_cfg, game_cfg)

    plots_dir = logger.subdir("plots")
    prefix = logger.run_info["start_time"]

    pdf_path = os.path.join(plots_dir, f"{prefix}_family_report.pdf")
    print("Saving family report to:", pdf_path)

    with PdfPages(pdf_path) as pdf:
        #Intro pages with Family Descriptions
        fig = make_family_legend_page(fam_cfg, families)
        pdf.savefig(fig)
        plt.close(fig)

        #Generate metric plots, 1 page per group of plots
        for metric_name, spec in metric_specs.items():
            items = [spec["extract"](res) for _, res in results]
            if spec["extract2"] is not None:
                items2 = [spec["extract2"](res) for _, res in results]
            else:
                items2 = None
            
            if fam_cfg.target_payoff:
                target = fam_cfg.target_payoff
            else:
                target = ""
            suptitle = f"{metric_name} by {target} {fam_cfg.vary}"

            titles = [f"Population Family {idx + 1}" for idx, _ in enumerate(results)]

            fig = make_population_grid(
                items=items,
                items2=items2,
                plot_fn=spec["plot_fn"],
                titles=titles,
                suptitle=suptitle,
                x_label=spec["x_label"],
                y_label=spec["y_label"],
                stat_sum=spec["stat_sum"]
                )
            pdf.savefig(fig)
            plt.close(fig)

    logger.close()

    return pdf_path

def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument(
                        "--config",
                        type=str,
                        required=True,
                        help="Path to JSON config file for a family of populations",
                        )
    args = parser.parse_args()

    fam_cfg, game_cfg = load_config(args.config)

    metric_specs = {
    "Attendance": {
        "extract": lambda res: res["Attendance"],
        "extract2": None,
        "plot_fn": plot_series,          # your callback
        "y_label": "Attendance A_t",
        "x_label": "Round",
        "stat_sum": True,
    },
    "Prices": {
        "extract": lambda res: res["Prices"],
        "extract2": None,
        "plot_fn": plot_series,
        "y_label": "Price",
        "x_label": "Time/Round",
        "stat_sum": True,
    },
    "Attendance distribution": {
        "extract": lambda res: res["Attendance"],
        "extract2": None,
        "plot_fn": plot_hist,          # your callback
        "y_label": "Frequency",
        "x_label": "Attendance",
        "stat_sum": True,
    },
    "Succes rates": {
        "extract": lambda res: res["final_wins"]/game_cfg.rounds,
        "extract2": None,
        "plot_fn": plot_hist,
        "y_label": "Frequency",
        "x_label": "Success Rate",
        "stat_sum": True,
    },
    "points": {
        "extract": lambda res: res["final_points"],
        "extract2": None,
        "plot_fn": plot_hist,
        "y_label": "Frequency",
        "x_label": "Points",
        "stat_sum": True,
    },
    "Wealth vs wins":{
        "extract": lambda res: res["final_wins"]/game_cfg.rounds,
        "extract2": lambda res: res["final_wealth"],
        "plot_fn": plot_scatter,
        "x_label": "Wealth",
        "y_label": "Win frequency",
        "stat_sum": True,
    },
    "Wealth vs Switches":{
        "extract": lambda res: res["strategy_switches"],
        "extract2": lambda res: res["final_wealth"],
        "plot_fn": plot_scatter,
        "y_label": "Num Switches",
        "x_label": "Wealth",
        "stat_sum": True,
    },
    "Autocorrelation of returns":{
        "extract": lambda res: returns_from_prices(res["Prices"])[1:],
        "extract2": lambda res: returns_from_prices(res["Prices"])[:-1],
        "plot_fn": plot_scatter,
        "y_label": "r(t+1)",
        "x_label": "r(t)",
        "stat_sum": True,
    },
    "Risk Return":{
        "extract": lambda res: [risk_from_wealth(w)[0] for w in res["wealth"].T],
        "extract2": lambda res: [risk_from_wealth(w)[1] for w in res["wealth"].T],
        "plot_fn": plot_scatter,
        "y_label": "return",
        "x_label": "risk",
        "stat_sum": False,
        },
    "wealth_by_cohort":{
        "extract": lambda res: group_vector_by_cohort(
            res["final_wealth"], 
            res["cohort_ids"]),
        "extract2": None,
        "plot_fn": plot_box,
        "y_label": "Wealth",
        "x_label": "Cohort",
        "stat_sum": False,
        },
    "success_by_cohort":{
        "extract": lambda res: group_vector_by_cohort(
            res["final_wins"]/game_cfg.rounds, 
            res["cohort_ids"]),
        "extract2": None,
        "plot_fn": plot_box,
        "y_label": "Success Rate",
        "x_label": "Cohort",
        "stat_sum": False,
        }
    # add more metrics here later
}

    path = build_family_report(fam_cfg, game_cfg, metric_specs)
    print(path)

if __name__ == "__main__":
    main()
    