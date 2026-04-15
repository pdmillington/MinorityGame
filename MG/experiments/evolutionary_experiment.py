#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:24:28 2026

@author: petermillington

Runs multiple independent evolutionary simulations and aggregates statistics
across runs

Each run is a full EvolutionaryGame - num_generations generations of rounds_
per_generation rounds each with strat evolution driven by poverty ranking 
across the whole population.

Outputs pdf report with:
        - Experiment summary
        - Price series with generation boundaries
        - Attendance variance per generation
        - Return variance per generation
        - Wealth threshold per generation
        - Replacement rate per cohort per generation (mean across runs)
        - Mean wealth per cohort per generation (mean across runs)
        - Wealth distribution by cohort - final generation, averaged across runs
"""

import os

import json
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

from core.game_config import GameConfig
from core.evolutionary_game import EvolutionaryGame
from utils.logger import RunLogger, log_simulation

# Config

@dataclass
class EvolutionaryConfig:
    """
    Full config for evolutionary experiment
    
    Parameters
    ----------
    population_spec : dict
        Full population spec supporting mixed payoff cohorts
        score_lambda may be set independently
    rounds_per_generation : int
        Number of rounds in each generation
    num_runs : int
        Number of independent runs
    poverty_percentile : float
        Fraction of population eligible for replacement
    replacement_rate : float
        Fraction of eligible actually to be replaced
    reset_wealth : bool
        True = bankruptcy so wealth, position, cash reset to zero for replaced agents
        False = strategy swap only
    lambda_ : float
        Market impact parameter.  None = auto (1 / N*50)
    market_maker : bool or None
        Whether to include a market maker.
    seed : int or None
        Master seed for reproducibility.
    record_agent_series : bool
        Whether to record per-agent time series. Expensive for large populations.
        Recommended False for multi-run experiments.
    save_dir : str
        Directory for output PDF.
    log_path : str
        Path for simulation log.
    price : float
        Initial price. Persists across generations.
    """
    population_spec: dict
    rounds_per_generation : int
    num_generations: int
    num_runs: int
    poverty_percentile: float
    replacement_rate: float
    reset_wealth: bool
    lambda_: Optional[float] = None
    market_maker: Optional[bool] = None
    seed: Optional[int] = None
    record_agent_series: bool = False
    save_dir: str = "plots/evolutionary"
    log_path: str = "logs/simulation_log_txt"
    price: float = 100.0

def load_config(path: str) -> EvolutionaryConfig:
    with open(path, "r") as f:
        data = json.load(f)
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return EvolutionaryConfig(**data)

# Single run
def simulate_single_run(args: tuple) -> dict:
    """
    Run one full evolutionary simulation for num_generations generations
    Called in parallel with ProcessPoolExecutor

    Returns
    -------
    dict with keys
        'price_series'       : full concatenated price series across all generations
        'generation_boundaries' : round indices where each generation starts
        'gen_stats'          : list of per-generation summary dicts
        'replacement_stats'  : list of per-generation replacement dicts
        'cohort_ids'         : list of unique cohort ids in population order
    """
    cfg, run_seed = args
    
    game_cfg = GameConfig(
        rounds=cfg.rounds_per_generation,
        lambda_=cfg.lambda_,
        mm=cfg.market_maker,
        price=cfg.price,
        record_agent_series=cfg.record_agent_series,
        seed=run_seed,
        )

    game = EvolutionaryGame(
        population_spec=cfg.population_spec,
        cfg=game_cfg,
        rounds_per_generation=cfg.rounds_per_generation,
        num_generations=cfg.num_generations,
        poverty_percentile=cfg.poverty_percentile,
        replacement_rate=cfg.replacement_rate,
        reset_wealth=cfg.reset_wealth,
        )

    evolutionary_results = game.run_evolutionary()

    # Build price series across generations
    price_segments = []
    for gen_result in evolutionary_results["generation_results"]:
        prices = gen_result["Prices"]
        price_segments.append(prices)

    # Generation boundaries remove the repeated price at each boundary
    generation_boundaries = [
        g * cfg.rounds_per_generation
        for g in range(cfg.num_generations)
        ]

    full_price = price_segments[0].copy()
    for seg in price_segments[1:]:
        full_price = np.concatenate([full_price, seg[1:]])

    # Per generation stat summary
    gen_stats = []
    for gen_result in evolutionary_results["generation_results"]:
        gen = gen_result["generation"]
        attendance = gen_result["Attendance"]
        prices = gen_result["Prices"]

        prices_safe = np.where(prices<=0, np.nan, prices)
        log_prices = np.log(prices_safe)
        returns = np.diff(log_prices)

      

        gen_stats.append({
            "generation": gen,
            "sigma2_attendance": float(np.nanvar(attendance)),
            "var_returns": float(np.nanvar(returns)),
            "mean_returns": float(np.nanmean(returns)),
            "mean_wealth": float(np.nanmean(gen_result["final_wealth"])),
            "std_wealth": float(np.nanstd(gen_result["final_wealth"])),
            "final_wealth": gen_result["final_wealth"],
            "cohort_ids": (gen_result.get("cohort_ids").tolist()
                           if gen_result.get("cohort_ids") is not None else []),
            })

    # replacement stats
    replacement_stats = []
    for rec in evolutionary_results["generation_records"]:
        replacement_stats.append({
            "generation": rec.generation,
            "wealth_threshold": rec.wealth_threshold,
            "n_eligible": len(rec.eligible_indices),
            "n_replaced": len(rec.replaced_indices),
            "cohort_replacement_counts": rec.cohort_replacement_counts,
            "cohort_replacement_rates": rec.cohort_replacement_rates,
            "cohort_mean_wealth_replaced": rec.cohort_mean_wealth_replaced,
            "cohort_mean_wealth_surviving": rec.cohort_mean_wealth_surviving,
            })

    # unique cohort ids in population order
    cohort_ids = sorted(set(
        p.cohort_id for p in game.players if p.cohort_id is not None))

    return {
        "price_series": full_price,
        "generation_boundaries": generation_boundaries,
        "gen_stats": gen_stats,
        "replacement_stats": replacement_stats,
        "cohort_ids": cohort_ids,
        }

# Aggregation helpers

def _collect_gen_stats(all_run_results: list, key: str) -> np.ndarray:
    """
    Collect a scalar per generation statistic across runs

    Returns
    -------
    Array of shape (num_runs, num_generations).

    """
    num_runs =  len(all_run_results)
    num_gen = len(all_run_results[0]["gen_stats"])
    out = np.full((num_runs, num_gen), np.nan)
    for r, run in enumerate(all_run_results):
        for g, gs in enumerate(run["gen_stats"]):
            out[r, g] = gs[key]
    return out

def _collect_replacement_stat(all_run_results: list, key: str,
                              cohort_id: int) -> np.ndarray:
    """
    Collect a per cohort replacement stat across runs and generations.

    Returns
    -------
    Array of shape (num_runs, num_generations - 1).

    """
    num_runs = len(all_run_results)
    num_rec = len(all_run_results[0]['replacement_stats'])
    out = np.full((num_runs, num_rec), np.nan)
    for r, run in enumerate(all_run_results):
        for g, rs in enumerate(run["replacement_stats"]):
            val = rs[key].get(cohort_id, np.nan)
            out[r, g] = val if val is not None else np.nan
    return out

# Plot helpers
def _plot_with_band(ax, x, mean, std, label, color):
    """Plot mean line with 1 std shaded band"""
    ax.plot(x, mean, label=label, color=color)
    if np.any(~np.isnan(std)) and np.nanmax(std) > 0:
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

def _add_generation_boundaries(ax, boundaries, rounds_per_gen, color='gray',
                               alpha=0.4, linestyle="--"):
    """Add vertical dashed line at generation boundaries"""
    for b in boundaries[1:]:
        ax.axvline(x=b, color=color, alpha=alpha, linestyle=linestyle,
                   linewidth=0.8)

# Report pages

def _page_config_summary(cfg: EvolutionaryConfig) -> plt.Figure:
    """Text page summarising experiment setup"""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    cohorts = cfg.population_spec.get("cohorts", [])
    cohort_lines = []
    for i, c in enumerate(cohorts):
        parts = [f"{k}={v}" for k, v in c.items() if k != "agent_type"]
        cohort_lines.append(f"   Cohort {i}: " + ",".join(parts))

    lines = [
        "Evolutionary Experiment Configuration",
        "-" * 45,
        f"Rounds per generation : {cfg.rounds_per_generation}",
        f"Generations           : {cfg.num_generations}",
        f"Runs                  : {cfg.num_runs}",
        f"Poverty percentile    : {cfg.poverty_percentile:.1%}",
        f"Replacement rate      : {cfg.replacement_rate:.1%}",
        f"Reset wealth          : {cfg.reset_wealth}",
        f"Market maker          : {cfg.market_maker}",
        f"Lambda                : {cfg.lambda_}",
        f"Seed                  : {cfg.seed}",
        f"Initial Price         : {cfg.price}",
        "",
        f"Population  (total = {cfg.population_spec.get('total', '?')})",
        "-" * 45,
        ] + cohort_lines

    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            fontfamily="monospace")
    fig.suptitle("Experiment Configuration", fontsize=12)
    fig.tight_layout()
    return fig

def _page_price_series(all_run_results: list,
                       cfg: EvolutionaryConfig) -> plt.Figure:
    """
    Full continuous prices series for all runs overlaid on a single chart
    """
    fig, ax = plt.subplots(figsize=(11.69, 5.83))
    
    colors = plt.cm.tab10.colors
    for i, run in enumerate(all_run_results):
        prices = run["price_series"]
        x = np.arange(len(prices))
        ax.plot(x, prices,
                linewidth=0.6,
                alpha=0.6,
                color=colors[i % len(colors)],
                label=f"Run {i}")

    boundaries = all_run_results[0]["generation_boundaries"]
    for b in boundaries[1:]:
        ax.axvline(x=b, color="gray", alpha=0.4,
                   linestyle="--", linewidth=0.8)

    ax.set_xlabel("Round", fontsize=9)
    ax.set_ylabel("Price", fontsize=9)
    ax.set_title("Price Series - All runs\n"
                 "(dashed lines = generation boundaries)", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    return fig

def _page_market_dynamics(all_run_results: list,
                          cfg: EvolutionaryConfig) -> plt.Figure:
    """Attendance and return variance per generation"""
    sigma2 = _collect_gen_stats(all_run_results, "sigma2_attendance")
    var_ret = _collect_gen_stats(all_run_results, "var_returns")
    generations = np.arange(cfg.num_generations)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

    _plot_with_band(ax1, generations,
                    sigma2.mean(axis=0), sigma2.std(axis=0),
                    "σ²(attendance)", "steelblue")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Attendance variance σ²")
    ax1.set_title("Attendance Variance per Generation")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    _plot_with_band(ax2, generations,
                    var_ret.mean(axis=0), var_ret.std(axis=0),
                    "σ²(returns)", "darkorange")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Return variance")
    ax2.set_title("Return Variance per Generation")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Market Dynamics Across Generations "
                 f"(mean ± 1 std, n={cfg.num_runs} runs)", fontsize=11)
    fig.tight_layout()
    return fig

def _page_replacement_dynamics(all_run_results: list,
                               cfg: EvolutionaryConfig) -> plt.Figure:
    """Replacement rate and wealth threshold per cohort/generation"""
    cohort_ids = all_run_results[0]["cohort_ids"]
    n_records = cfg.num_generations - 1
    gen_x = np.arange(n_records)

    # Wealth threshold across runs
    thresholds = np.array([
        [rs["wealth_threshold"] for rs in run["replacement_stats"]]
        for run in all_run_results
        ])

    colors = plt.cm.tab10.colors

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

    # replacement rate per cohort
    for idx, c_id in enumerate(cohort_ids):
        rates = _collect_replacement_stat(
            all_run_results, "cohort_replacement_rates", c_id)
        _plot_with_band(ax1, gen_x,
                        rates.mean(axis=0),
                        rates.std(axis=0),
                        f"Cohort {c_id}",
                        colors[idx % len(colors)])

    ax1.axhline(y=cfg.poverty_percentile * cfg.replacement_rate,
                color='black', linestyle=":", linewidth=0.8,
                label="Expected rate (p × r)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Replacement rate")
    ax1.set_title("Replacement rate per cohort")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # wealth threshold
    _plot_with_band(ax2, gen_x,
                    thresholds.mean(axis=0),
                    thresholds.std(axis=0),
                    "Poverty threshold", "purple")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Wealth")
    ax2.set_title("Poverty Threshold per Generation")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Evolutionary Pressure across Generations "
                 f"(mean ± 1 std, n={cfg.num_runs} runs)", fontsize=11)
    fig.tight_layout()
    return fig

def _page_cohort_wealth(all_run_results: list,
                        cfg: EvolutionaryConfig) -> plt.Figure:
    """Mean wealth per cohort and generation, averaged"""
    #Debugging
    cohort_ids = all_run_results[0]["cohort_ids"]
    print("cohort_ids for plotting:", cohort_ids)
    gs0 = all_run_results[0]["gen_stats"][0]
    cids_arr = np.array(gs0["cohort_ids"])
    print("cohort_ids array dtype:", cids_arr.dtype)
    print("unique values in cohort_ids arr:", np.unique(cids_arr))
    print("type of cohort_ids[0]:", type(cohort_ids[0]))
    for c_id in cohort_ids:
        mask = cids_arr == c_id
        w=np.array(gs0["final_wealth"][mask])
        print(f"cohort {c_id}: mean={np.mean(w):.4f}, std={np.std(w):.4f}, min={np.min(w):.4f}, max={np.max(w):.4f}")
    #Debugging
    generations = np.arange(cfg.num_generations)
    colors = plt.cm.tab10.colors
    
    fig, ax = plt.subplots(figsize=(8.27, 5.83))

    for idx, c_id in enumerate(cohort_ids):
        c_id = int(c_id)
        #debug
        gs = all_run_results[0]["gen_stats"][0]
        cids = np.array(gs["cohort_ids"], dtype=int)
        fw = np.array(gs["final_wealth"])
        mask = cids == c_id
        print(f"c_id={c_id}, mask sum={mask.sum()}, mean wealth={np.mean(fw[mask]):.4f}")
        wealth_by_run_gen = np.array([
            [float(np.mean(np.array(gs["final_wealth"])[
                np.array(gs["cohort_ids"], dtype=int) == c_id]
                           ))
                for gs in run["gen_stats"]
                ]
            for run in all_run_results
            ])
        
        _plot_with_band(ax, generations,
                        np.nanmean(wealth_by_run_gen, axis=0),
                        np.nanstd(wealth_by_run_gen, axis=0, ddof=0),
                        f"Cohort {c_id}",
                        colors[idx % len (colors)])

    ax.axhline(y=0, color="black", linewidth=0.6, linestyle="-")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Wealth")
    ax.set_title("Mean Wealth per Cohort Across Generations\n"
                 f"(mean ± 1 std, n={cfg.num_runs} runs)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

def _page_final_wealth_boxplot(all_run_results: list,
                               cfg: EvolutionaryConfig) -> plt.Figure:
    """Wealth distribution by cohort from the final generation pooled across runs"""
    cohort_ids = all_run_results[0]["cohort_ids"]

    # collect final generation wealth per cohort
    cohort_wealth_pooled = {c_id: [] for c_id in cohort_ids}
    for run in all_run_results:
        final_gs = run["gen_stats"][-1]
        wealth = np.array(final_gs["final_wealth"])
        cids = np.array(final_gs["cohort_ids"])
        for c_id in cohort_ids:
            c_id = int(c_id)
            mask = cids == c_id
            cohort_wealth_pooled[c_id].extend(wealth[mask].tolist())

    fig, ax = plt.subplots(figsize=(8.27, 5.83))
    data = [cohort_wealth_pooled[c_id] for c_id in cohort_ids]
    labels = [f"Cohort {c_id}" for c_id in cohort_ids]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False)
    colors = plt.cm.tab10.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0, color="black", linewidth=0.6, linestyle="-")
    ax.set_ylabel("Wealth")
    ax.set_title("Final Generation Wealth Distribution by Cohort\n"
                 f"(pooled across {cfg.num_runs} runs)")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()
    return fig

# Main experiment runner

def run_evolutionary_experiment(cfg: EvolutionaryConfig):
    """Run num_runs independent evolutionary sims in parallel, produce pdf report"""
    rng = np.random.default_rng(cfg.seed)
    run_seeds = [int(rng.integers(0, 2**31)) for _ in range(cfg.num_runs)]

    print(f"Running {cfg.num_runs} evolutionary runs "
          f"({cfg.num_generations} generations x"
          f"{cfg.rounds_per_generation} rounds each...")

    with ProcessPoolExecutor() as executor:
        args_list = [(cfg, seed) for seed in run_seeds]
        all_run_results = list(executor.map(simulate_single_run, args_list))
    
    print("Runs complete, building report .. ")

    # PDF report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg.save_dir, exist_ok=True)
    pdf_path = os.path.join(cfg.save_dir, f"{timestamp}_evolutionary_report.pdf")

    with PdfPages(pdf_path) as pdf:
        
        # page 1
        fig = _page_config_summary(cfg)
        pdf.savefig(fig)
        plt.close(fig)
        
        # page 2
        fig = _page_price_series(all_run_results, cfg)
        pdf.savefig(fig)
        plt.close(fig)
 
        # Page 3: market dynamics (attendance var, return var)
        fig = _page_market_dynamics(all_run_results, cfg)
        pdf.savefig(fig)
        plt.close(fig)
 
        # Page 4: replacement dynamics
        fig = _page_replacement_dynamics(all_run_results, cfg)
        pdf.savefig(fig)
        plt.close(fig)
 
        # Page 5: cohort wealth across generations
        fig = _page_cohort_wealth(all_run_results, cfg)
        pdf.savefig(fig)
        plt.close(fig)
 
        # Page 6: final generation wealth boxplot
        fig = _page_final_wealth_boxplot(all_run_results, cfg)
        pdf.savefig(fig)
        plt.close(fig)
 
    # --- log ---
    metadata = [
        f"[{timestamp}] Evolutionary Experiment",
        f"Generations          : {cfg.num_generations}",
        f"Rounds per generation: {cfg.rounds_per_generation}",
        f"Runs                 : {cfg.num_runs}",
        f"Poverty percentile   : {cfg.poverty_percentile}",
        f"Replacement rate     : {cfg.replacement_rate}",
        f"Reset wealth         : {cfg.reset_wealth}",
        f"Report saved         : {pdf_path}",
        "=" * 60,
    ]
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    log_simulation(metadata, cfg.log_path)
 
    print(f"Report saved to: {pdf_path}")
    return pdf_path

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to JSON config file for evolutionary experiment"
                        )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_evolutionary_experiment(cfg)

if __name__ == "__main__":
    main()



