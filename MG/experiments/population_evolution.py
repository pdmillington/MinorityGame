#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:45:40 2026

@author: petermillington
"""

import json
from dataclasses import dataclass
import argparse
from typing import List, Any, Dict

import numpy as np

from core.game import Game
from core.game_config import GameConfig
from analysis.cohort_utils import group_vector_by_cohort, cohort_labels_from_meta
from analysis.plot_utils import create_price_figure, create_boxplot_figure
from analysis.information_metrics import create_rolling_mi_figure, create_information_summary_figure
from analysis.report_builder import ReportBuilder
from utils.logger import log_simulation, RunLogger

@dataclass
class PopulationEvolutionConfig:
    """
    Dataclass for population evolution to study impact of adding higher m
    agents to a simulation
    """
    burn_in:           int         # num rounds before introducing new agents
    total_rounds:      int         # total rounds including burn in
    base_m:            int         # memory of base agent
    base_s:            int         # s_value for base 
    base_n:            int         # population size for the base
    payoff:            str         # payoff  
    proportion_new:    int         # proportion of higher m agents introduced
    mode:              str         # modes of introduction: ['sudden', 'gradual', 'quantum']
    gradual_window:    int         # rounds over which the gradual intro spreads
    quantum_interval:  int         # rounds between quantum batches
    quantum_number:    int         # number of quantum interventions
    roll_window:       int         # rollling MI window
    step:              int         # rolling MI step
    lambda_:           float = None
    market_maker:      bool | None = None
    price:             float = 100 
    record_agent_series: bool = True
    seed:              int = 1234

def load_config(path:str) -> PopulationEvolutionConfig:
    with open(path, "r") as f:
        data = json.load(f)
        config_data = {k: v for k, v in data.items() if not k.startswith('_')}

    pop_cfg = PopulationEvolutionConfig(**config_data)
    game_cfg = GameConfig(
        rounds=pop_cfg.total_rounds,
        lambda_=pop_cfg.lambda_,
        mm=pop_cfg.market_maker,
        price=pop_cfg.price,
        record_agent_series=pop_cfg.record_agent_series,
        )
    return pop_cfg, game_cfg

def _build_summary_text(
        pop_cfg: PopulationEvolutionConfig,
        n_new: int,
        activation_list: List[int],
) -> str:
    """Format experiment summary for the report cover page."""

    total_n = pop_cfg.base_n + n_new

    # Mode-specific schedule description
    if pop_cfg.mode == 'sudden':
        schedule_desc = (
            f"  All {n_new} agents activated simultaneously at round {pop_cfg.burn_in}"
        )
    elif pop_cfg.mode == 'gradual':
        per_round = n_new // pop_cfg.gradual_window
        schedule_desc = (
            f"  {per_round} agents activated per round over {pop_cfg.gradual_window} rounds\n"
            f"  Introduction window: rounds {pop_cfg.burn_in} — "
            f"{pop_cfg.burn_in + pop_cfg.gradual_window}"
        )
    elif pop_cfg.mode == 'quantum':
        per_batch = int(pop_cfg.proportion_new * pop_cfg.base_n / 100)
        schedule_desc = (
            f"  {per_batch} agents per batch, every {pop_cfg.quantum_interval} rounds\n"
            f"  {pop_cfg.quantum_number} batches, total {n_new} agents introduced\n"
            f"  First batch at round {pop_cfg.burn_in}, "
            f"last at round {pop_cfg.burn_in + (pop_cfg.quantum_number - 1) * pop_cfg.quantum_interval}"
        )

    lines = [
        "EXPERIMENT: Population Evolution",
        "=" * 50,
        "",
        "GAME PARAMETERS",
        "-" * 50,
        f"  Total rounds:        {pop_cfg.total_rounds:,}",
        f"  Burn-in:             {pop_cfg.burn_in:,}",
        f"  Post-intervention:   {pop_cfg.total_rounds - pop_cfg.burn_in:,}",
        f"  Lambda:              {pop_cfg.lambda_}",
        f"  Market maker:        {pop_cfg.market_maker}",
        f"  Seed:                {pop_cfg.seed}",
        f"  Payoff:              {pop_cfg.payoff}",
        "",
        "POPULATION STRUCTURE",
        "-" * 50,
        f"  Total agents (N):    {total_n}  {'[ODD ✓]' if total_n % 2 == 1 else '[EVEN ✗]'}",
        "",
        f"  Cohort 0 — Base agents",
        f"    n = {pop_cfg.base_n}, m = {pop_cfg.base_m}, S = {pop_cfg.base_s}",
        f"    Active from round 0 (all rounds)",
        "",
        f"  Cohort 1 — Introduced agents",
        f"    n = {n_new} ({pop_cfg.proportion_new}% of base), "
        f"m = {pop_cfg.base_m + 1}, S = {pop_cfg.base_s}",
        f"    Mode: {pop_cfg.mode.upper()}",
        schedule_desc,
        "",
        "ROLLING MI PARAMETERS",
        "-" * 50,
        f"  Window:              {pop_cfg.roll_window:,} rounds",
        f"  Step:                {pop_cfg.step} rounds",
        f"  k (base):            {pop_cfg.base_m}",
        f"  k (introduced):      {pop_cfg.base_m + 1}",
        "",
        "ACTIVATION SCHEDULE",
        "-" * 50,
        f"  First activation:    round {min(activation_list)}",
        f"  Last activation:     round {max(activation_list)}",
        f"  Unique rounds:       {len(set(activation_list))}",
    ]

    return "\n".join(lines)

def build_activation_schedule(
        config: PopulationEvolutionConfig,
        ) -> List[int]:
    """
    Returns list of active_from values for new agents
    
    sudden: all = burn_in
    gradual: evenly spaced across burn_in and gradual_window
    quantum: batches of new agents every quantum_interval rounds
    """
    cfg = config
    new_agents = int(cfg.proportion_new * cfg.base_n / 100 )
    
    activation_list = []
    
    if cfg.mode == 'sudden':
        activation_list = [cfg.burn_in] * new_agents
    elif cfg.mode == 'gradual':
        agents_per_round = new_agents//cfg.gradual_window
        remainder = new_agents % cfg.gradual_window
        for r in range(cfg.gradual_window):
            dose = agents_per_round + (1 if r < remainder else 0)
            activation_list += [cfg.burn_in + r] * dose
    elif cfg.mode == 'quantum':
        for i in range(cfg.quantum_number):
            activation_list += [cfg.burn_in + i * cfg.quantum_interval] * new_agents
            
    return activation_list

def run_evolution_experiment(
        cfg: PopulationEvolutionConfig,
        game_cfg: GameConfig,
        ) -> Dict[str, Any]:
    """
    Build population with dormant higher m agents, fun full game,
    returns results plus activation schedule for reporting
    """
    activation_list = build_activation_schedule(config=cfg)
    activation_list_dict={1: activation_list}
    
    total_agents = cfg.base_n + len(activation_list)
    cohorts = []
    
    cohorts.append({
        "count": cfg.base_n,
        "memory": cfg.base_m,
        "payoff": cfg.payoff,
        "strategies": cfg.base_s,
        "position_limit": 5,
        })
    
    cohorts.append({
        "count": len(activation_list),
        "memory": cfg.base_m + 1,
        "payoff": cfg.payoff,
        "strategies": cfg.base_s,
        "position_limit": 5,
        })
    
    pop_spec = {"total": total_agents, "cohorts": cohorts}
    
    game = Game(
        population_spec=pop_spec,
        cfg=game_cfg,
        activation_schedule=activation_list_dict,
        )
    results = game.run()
    
    labels = cohort_labels_from_meta(game.meta)
    
    return {
        "game_results": results,
        "activation_list": activation_list,
        "labels": labels,
        }
    
def build_evolution_report(pop_cfg: PopulationEvolutionConfig,
                           game_cfg: GameConfig,
                           ) -> str:
    """
    Experiment description
    Rolling MI at k=m_base and k=m_base + 1
    Price graph, attendance graph, wealth and success by cohort
    """
    all_results = run_evolution_experiment(cfg=pop_cfg, game_cfg=game_cfg)
    results = all_results["game_results"]
    activation_list = all_results["activation_list"]
    new_agents = len(activation_list)
    
    run_id = f"{pop_cfg.mode}_p{pop_cfg.proportion_new}_s{pop_cfg.seed}"
    logger = RunLogger(
        base_save_dir="simulation_runs",
        module="population_evolution",
        payoff=pop_cfg.payoff,
        run_id=run_id,
        seed=pop_cfg.seed,
        )
    logger.log_config(pop_cfg)
    
    attendance = results['Attendance']
    prices = results['Prices']
    cohort_ids = results['cohort_ids']
    price = results['Prices']
    final_wealth = results['final_wealth']
    final_wins = results['final_wins']
    
    # calculate success rate for agents accounting for late starters
    total_rounds =  game_cfg.rounds
    active_rounds = np.where(cohort_ids == 0, total_rounds, 0).astype(float)
    cohort1_mask = cohort_ids == 1
    cohort1_positions = np.where(cohort1_mask)[0]
    for pos, act_from in zip(cohort1_positions, activation_list):
        active_rounds[pos] = total_rounds - act_from
        
    success_rate = final_wins / np.maximum(active_rounds, 1)
    
    labels = all_results['labels']
    
    intervention_round = pop_cfg.burn_in
    
    success_by_cohort = group_vector_by_cohort(success_rate, cohort_ids)
    wealth_by_cohort = group_vector_by_cohort(final_wealth, cohort_ids)
    
    report = ReportBuilder(logger.get_dir())
    
    # Experiment summary
    summary_text = _build_summary_text(pop_cfg, new_agents, activation_list)
    report.add_text_page(summary_text, title="Experiment Configuration")
    
    fig_k3 = create_rolling_mi_figure(
        attendance,
        k=pop_cfg.base_m,
        roll_window=pop_cfg.roll_window,
        step=pop_cfg.step,
        intervention_round=intervention_round,
        title=f'Rolling MI (k={pop_cfg.base_m}) - base agent memory',
        )
    
    report.add_figure(fig_k3, "MI_base_m", close_after=True)
    
    fig_k4 = create_rolling_mi_figure(
        attendance,
        k=pop_cfg.base_m + 1,
        roll_window=pop_cfg.roll_window,
        step=pop_cfg.step,
        intervention_round=intervention_round,
        title=f'Rolling MI (k={pop_cfg.base_m + 1}) - introduced agent memory',
        )

    report.add_figure(fig_k4, "MI_base_m+1", close_after=True)
    
    # Price graph
    fig_price = create_price_figure(prices, title="Prices")
    report.add_figure(fig_price, "price_series", close_after=True)
    
    # Success boxplot by cohort
    fig_success_box = create_boxplot_figure(
        success_by_cohort,
        labels,
        title="Success Rate by Cohort",
        ylabel="Success Rate")
    report.add_figure(fig_success_box, "success_boxplot", close_after=True)
    
    # Wealth distribution boxplot by cohort
    fig_wealth_box = create_boxplot_figure(
        wealth_by_cohort,
        labels,
        title="Wealth by Cohort",
        ylabel="Wealth")
    report.add_figure(fig_wealth_box, "wealth_boxplot", close_after=True)
    
    report_path = report.build()
    
    logger.log_metrics({
        "mode": pop_cfg.mode,
        "proportion_new": pop_cfg.proportion_new,
        "new_agents": new_agents,
        "total_agents": pop_cfg.base_n + new_agents,
        "burn_in": pop_cfg.burn_in,
        "total_rounds": pop_cfg.total_rounds,
        })
    logger.close()

    
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--config",
                        type=str,
                        required=True,
                        help="Path to JSON config file for a PopulationEvolutionConfig description",
                        )
    args = parser.parse_args()
    
    pop_cfg, game_cfg = load_config(args.config)
    
    path = build_evolution_report(pop_cfg, game_cfg)
    print(path)
    
if __name__ == "__main__":
    main()