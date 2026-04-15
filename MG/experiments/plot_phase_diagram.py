#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:41:18 2025

@author: petermillington
"""

import os
from datetime import datetime
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as sp_kurtosis

from core.game import Game
from core.game_config import GameConfig
from utils.logger import log_simulation
from analysis.information_metrics import compute_history_statistics
from analysis.report_builder import ReportBuilder

@dataclass
class PhaseDiagramConfig:
    payoff_key: str
    m_values: Iterable[int] = range(3,11)
    num_players: int = 301
    num_strategies: int = 2
    position_limit: int = None        
    noise_players: int = 0
    noise_allow_no_action: bool = False
    noise_position_limit: int = None
    rounds: int = 10_000
    num_games: int = 20
    market_maker: bool | None = None
    output_dir: str = "plots/phase"
    run_name: str = "phase_diagram"
    log_path: str = "logs/simulation_log.txt"
    compute_information_metrics: bool = False   #opt in for information metrics calculation
    # Info lag relative to m applied to the window for which information metrics are applied
    info_lag: int = 0

def load_config(path: str) -> PhaseDiagramConfig:
    with open(path, "r") as f:
        data =json.load(f)
        config_data = {k: v for k, v in data.items() if not k.startswith('_')}
    return PhaseDiagramConfig(**config_data)


def simulate_single_game(args):
    """
    Run one game with memory m and return market statistics
    
    Returns a dict with keys:
        'sigma2' = variance of attendance
        'var prices' = variance of prices
        'mean returns' = mean of returns
        'var returns' = variance of returns
        
    Parameters
    ----------
    args : tuple
        memory m, cfg.
    see PhaseDiagramConfig for cfg
    
    Returns
    -------
    dict with keys:
        sigma2          variance of attendance
        mean_returns    mean log-return
        var_returns     variance of log-returns
        mi              mutual information (bits)         [info metrics only]
        nmi             normalised mutual information     [info metrics only]
        market_info     Brouty-Garcin market information [info metrics only]
        predictability  fraction of histories with P≠0.5 [info metrics only]
        reject_emh_95   bool: G-test rejects EMH at 95%  [info metrics only]
        coverage        fraction of 2^m histories with   [info metrics only]
                        >= 10 observations (sparse-sample
                        reliability indicator)
    """
    m, cfg, game_id = args

    population_spec = {
        "total": cfg.num_players + cfg.noise_players,
        "cohorts":[
            {
                "count": cfg.num_players,
                "agent_type": "strategic",
                "memory": m,
                "payoff": cfg.payoff_key,
                "strategies": cfg.num_strategies,
                "position_limit": cfg.position_limit,
                }
            ]
        }

    if cfg.noise_players> 0:
        population_spec["cohorts"].append(
            {
                "agent_type": "noise",
                "count": cfg.noise_players,
                "allow_no_action": cfg.noise_allow_no_action,
                "position_limit": cfg.noise_position_limit})

    cfg_game = GameConfig(
        rounds=cfg.rounds,
        lambda_=1/(cfg.num_players * 50),
        mm=None,
        price=100,
        record_agent_series=True,
        seed=hash((m, game_id)) & 0x7FFFFFFF
        )

    game = Game(
        population_spec=population_spec,
        cfg=cfg_game,
    )
    results = game.run()

    attendance = results["Attendance"]
    prices = results["Prices"]

    prices_safe = np.where(prices <= 0, np.nan, prices)
    log_prices = np.log(prices_safe)
    returns = np.diff(log_prices)           # length rounds

    output = {
        "sigma2": np.nanvar(attendance),
        "kurtosis": float(sp_kurtosis(attendance, fisher=True, nan_policy="omit")),
        "mean_returns": np.nanmean(returns),
        "var_returns": np.nanvar(returns),
        }

    if cfg.compute_information_metrics:
        k = m +cfg.info_lag
        if k <= 0:
            raise ValueError(
                f"info_lag={cfg.info_lag} gives k={k} <= 0 for m={m}. "
                "Increase info_lag or reduce the minimum m value."
                )
        info = compute_history_statistics(attendance, window=k)
        output['mi'] = info['mutual_information']
        output['nmi'] = info['nmi']
        output['market_info'] = info['market_information']
        output['predictability'] = info['predictability']
        output['reject_emh_95'] = info['reject_emh_95']
        
        # Coverage: fraction of all possible histories seen at least 10 times
        n_sufficient = sum(
            1 for c in info['transition_counts'].values()
            if c[-1] + c[1] >= 10
            )
        output['coverage'] = n_sufficient / (2 ** m)

    return output

# Figure builders

def _fig_phase_diagram(alphas, normalized_vols, label):
    """σ²/N vs α, log-log scale."""
    fig, ax = plt.subplots(figsize=(8,6))
    ax.loglog(alphas, normalized_vols, "o-", label=label)
    ax.set_xlabel(r"$\alpha = 2^m / N$")
    ax.set_ylabel(r"$\sigma^2 / N$")
    ax.set_title("Phase Diagram (Log-Log)")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    return fig

def _fig_kurtosis(alphas, mean_kurtosis, label):
    """ Excess kurtosis vs α, semilog x"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(alphas, mean_kurtosis, "o-", label=label)
    ax.axhline(0, color='gray', ls='--', lw=0.8, label='Gaussian baseline')
    ax.set_xlabel(r"$\alpha = 2^m / N$")
    ax.set_ylabel(r"Excess kurtosis of $A$")
    ax.set_title("Excess Kurtosis of Attendance vs α")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    return fig
    
def _fig_mean_return(alphas, return_mean, label):
    """Mean |log-return| vs α, log-log scale."""
    fig, ax = plt.subplots(figsize=(8,6))
    ax.loglog(alphas, np.abs(return_mean), "o-", label=label)
    ax.set_xlabel(r"$\alpha = 2^m / N$")
    ax.set_ylabel(r"$|\bar{r}|$")
    ax.set_title("Mean |Return| (Log-Log)")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    return fig

def _fig_mi_vs_alpha(alphas, mean_mi, mean_nmi, m_values, label, info_lag=0):
    """
    Mutual information and NMI vs α (log x, linear y).
 
    MI = I(H_m; A_{t+1}) measures how many bits of information the m-bit
    price history provides about the direction of next-period attendance.
    A peak on the coordinated (low-α) side of α_c would show that the
    phase structure has a direct information-theoretic signature.
    
    Comparability note: each point uses a different window k=m, so raw MI values 
    are not on an exactly comparable due to the fact that as m increases the strategy
    table increases in size so information exploitation is degraded.
    """
    lag_desc = (r"$k = m$" if info_lag == 0
                else rf"$k = m {'+' if info_lag > 0 else ''}{info_lag}$")

    fig,  (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(r"Mutual Information vs $\alpha$  [window " + lag_desc + "]")
    
    k_label = "m" if info_lag == 0 else (f"m+{info_lag}" if info_lag > 0 else f"m{info_lag}")
    ax1.semilogx(alphas, mean_mi, "o-", label=label)
    ax1.set_xlabel(r"$\alpha = 2^m / N$")
    ax1.set_ylabel(rf"$\overline{{I}}(H_{{{k_label}}};\,A_{{t+1}})$ (bits)")
    ax1.set_title("Mean Mutual Information")
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    
    ax2.semilogx(alphas, mean_nmi, "s-", color="tab:orange", label=label)
    ax2.set_xlabel(r"$\alpha = 2^m / N$")
    ax2.set_ylabel(r"$\overline{\mathrm{NMI}}$")
    ax2.set_title("Mean Normalised MI")
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    
    fig.tight_layout()
    return fig

def _fig_info_panel(alphas, mean_market_info, mean_predictability,
                    emh_rejection_rate, mean_coverage, m_values, label):
    """
    Three-panel figure (log x, linear y):
      Left   — Brouty-Garcin market information vs α
      Centre — mean predictable-history fraction vs α
      Right  — EMH rejection rate (fraction of runs) vs α
 
    The EMH rejection rate is especially diagnostic: if the fraction of
    runs rejecting EMH exceeds the 5% null rate on the coordinated
    (low-α) side of α_c, the phase transition is also a transition in
    informational efficiency.
    """
    fig,  axes = plt.subplots(1, 4, figsize=(20,5))
    fig.suptitle(r"Information Efficiency vs $\alpha$")
    
    axes[0].semilogx(alphas, mean_market_info, "o-", color="tab:blue", label=label)
    axes[0].set_xlabel(r"$\alpha = 2^m / N$")
    axes[0].set_ylabel(r"$\overline{I_{\mathrm{mkt}}}$ (Brouty-Garcin)")
    axes[0].set_title("Mean Market Information")
    axes[0].grid(True, which="both", ls="--")
    axes[0].legend()
    
    axes[1].semilogx(alphas, mean_predictability, "s-", color="tab:green", label=label)
    axes[1].set_xlabel(r"$\alpha = 2^m / N$")
    axes[1].set_ylabel("Mean predictable fraction")
    axes[1].set_title("Mean Predictability")
    axes[1].grid(True, which="both", ls="--")
    axes[1].legend()
    
    axes[2].semilogx(alphas, emh_rejection_rate, "^-", color="tab:red", label=label)
    axes[2].set_xlabel(r"$\alpha = 2^m / N$")
    axes[2].set_ylabel("EMH rejection rate (95%)")
    axes[2].set_title("Fraction of runs rejecting EMH")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].axhline(0.05, color='gray', ls='--', lw=0.8, label="Null rate (5%)")
    axes[2].grid(True, which="both", ls="--")
    axes[2].legend()
    
    axes[3].semilogx(alphas, mean_coverage, "^-", color="tab:purple", label=label)
    axes[3].set_xlabel(r"$\alpha = 2^m / N$")
    axes[3].set_ylabel(r"Coverage (fraction of $2^m$ histories")
    axes[3].set_title(r"History coverage ($\geq 10$ histories)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].axhline(0.10, color='gray', ls='--', lw=0.8, label="10% threshold")
    axes[3].grid(True, which="both", ls="--")
    axes[3].legend()
    
    fig.tight_layout()
    return fig

def _build_info_text(cfg: PhaseDiagramConfig, timestamp: str,
                     alphas, normalized_vols, mean_kurtosis) -> str:
    """Text content for information summary page"""
    m_list = list(cfg.m_values)
    lines = [
        "PHASE DIAGRAM EXPERIMENT",
        "=" * 60,
        "",
        f"Timestamp          : {timestamp}",
        f"Payoff scheme      : {cfg.payoff_key}",
        "",
        "POPULATION",
        "-" * 40,
        f"  Strategic agents   : {cfg.num_players}",
        f"  Strategies each    : {cfg.num_strategies}",
        f"  Position limit     : {cfg.position_limit}",
        f"  Noise players      : {cfg.noise_players}",
    ]
    if cfg.noise_players > 0:
        lines += [
            f"    Allow no-action  : {cfg.noise_allow_no_action}",
            f"    Position limit   : {cfg.noise_position_limit}",
            ]
    lines += [
        "",
        "SIMULATION",
        "-" * 40,
        f"  Rounds per game    : {cfg.rounds:,}",
        f"  Repetitions        : {cfg.num_games}",
        f"  Market maker       : {cfg.market_maker}",
        f"  λ                  : 1 / (N×50) = {1/(cfg.num_players*50):.6f}",
        "",
        "PARAMETER SWEEP",
        "-" * 40,
        f"  m values           : {m_list}",
        f"  α range            : [{min(alphas):.5f}, {max(alphas):.5f}]",
        "",
        "RESULTS SUMMARY",
        "-" * 40,
        f"  σ²/N range         : [{min(normalized_vols):.5f}, {max(normalized_vols):.5f}]",
        f"  Excess kurtosis    : {min(mean_kurtosis):.3f},  {max(mean_kurtosis):.3f}]",
        "",
        "INFORMATION METRICS",
        "-" * 40,
        f"  Enabled            : {cfg.compute_information_metrics}",
        ]
    if cfg.compute_information_metrics:
        lag_str = ("0  (k = m, agents' own horizon)"
                   if cfg.info_lag == 0 else
                   f"{cfg.info_lag:+d}   (k = m{cfg.info_lag:+d})")
        lines += [
            f"  Window lag          : {lag_str}",
            "  MI is bounded by H(A) <= 1 bit regardless of k;",
            "  values are comparable in bits across all alpha points.",
            ]
    lines += [
        "",
        "APPROXIMATION NOTE",
        "-" * 40,
        "  Virtual strategy scoring uses the standard annealed",
        "  approximation: each virtual strategy is scored against",
        "  the actual flow A, neglecting the O(1/N) self-impact",
        "  correction that arises if the agent had played the",
        "  virtual action. The correction is largest near α_c",
        "  where |A| is small. Results are directly comparable",
        "  to the MG literature.",
        "",
        "OUTPUT",
        "-" * 40,
        f"  Directory          : {cfg.output_dir}/{cfg.run_name}_{timestamp}/",
        ]
    return "\n".join(lines)

def run_phase_diagram(cfg: PhaseDiagramConfig):
    """
    Run num_games repetitions for each m in m_values and produce:
      {output_dir}/{run_name}_{timestamp}/
      ├── experiment_report.pdf   — information page + all figures
      └── figures/
          ├── phase_diagram.pdf
          ├── mean_return.pdf
          ├── mi_vs_alpha.pdf          [if compute_information_metrics]
          └── info_efficiency_panel.pdf [if compute_information_metrics]
    """
    alphas = []
    normalized_vols = []
    mean_kurtosis = []
    return_mean = []
    metadata = []

    info_mean_mi = []
    info_mean_nmi = []
    info_mean_market_info = []
    info_mean_predictability = []
    info_emh_rejection_rate = []
    info_mean_coverage = []
    

    with ProcessPoolExecutor() as executor:
        for m in cfg.m_values:
            args_list = [(m, cfg, game_id) for game_id in range(cfg.num_games)]
            results = list(executor.map(simulate_single_game, args_list))

            sigmas = np.array([r["sigma2"] for r in results], dtype=float)
            kurt = np.array([r["kurtosis"] for r in results], dtype=float)
            r_mean = np.array([r["mean_returns"] for r in results], dtype=float)

            sigma2_mean = np.mean(sigmas)
            r_mean_ave = np.mean(r_mean)
            alpha = 2 ** m / cfg.num_players
            
            alphas.append(alpha)
            normalized_vols.append(sigma2_mean / cfg.num_players)
            mean_kurtosis.append(float(np.mean(kurt)))
            return_mean.append(r_mean_ave)
            
            if cfg.compute_information_metrics:
                info_mean_mi.append(float(np.mean([r["mi"] for r in results])))
                info_mean_nmi.append(float(np.mean([r['nmi'] for r in results])))
                info_mean_market_info.append(float(np.mean([r['market_info'] for r in results])))
                info_mean_predictability.append(float(np.mean([r['predictability'] for r in results])))
                info_emh_rejection_rate.append(float(np.mean([r['reject_emh_95'] for r in results])))
                info_mean_coverage.append(float(np.mean([r['coverage'] for r in results])))
            
            metadata.append(
                f"m={m}, alpha={alpha:.5f}, "
                f"sigma^2/N={sigma2_mean / cfg.num_players:.5f}, "
                f"m_maker={cfg.market_maker}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, f"{cfg.run_name}_{timestamp}")

    label = cfg.payoff_key

    with ReportBuilder(run_dir, report_name="phase_diagram_exp_report.pdf") as report:
        
        report.add_text_page(
            _build_info_text(cfg, timestamp, alphas, normalized_vols, mean_kurtosis),
            title="Experiment Configuration"
            )
        
        report.add_figure(_fig_phase_diagram(alphas, normalized_vols, label),
                          "phase_diagram"
                          )
        
        report.add_figure(_fig_kurtosis(alphas, mean_kurtosis, label),
                          "kurtosis"
                          )
        
        if cfg.compute_information_metrics:
            m_values = list(cfg.m_values)
            report.add_figure(
                _fig_mi_vs_alpha(alphas, info_mean_mi, info_mean_nmi,
                                 m_values, label,
                                 info_lag=cfg.info_lag),
                "mi_vs_alpha"
                              )
            report.add_figure(
                _fig_info_panel(
                    alphas,
                    info_mean_market_info,
                    info_mean_predictability,
                    info_emh_rejection_rate,
                    info_mean_coverage,
                    m_values,
                    label),
                "info_efficiency_panel"
                )

    # Logging
    metadata.insert(0, f"[{timestamp}] Phase Diagram with {label}")
    metadata.extend(
        [
            f"Players: {cfg.num_players}",
            f"Strategies: {cfg.num_strategies}",
            f"Rounds: {cfg.rounds}",
            f"Repetitions: {cfg.num_games}",
            f"Market Maker: {cfg.market_maker}",
            f"Position Limit: {cfg.position_limit}",
            f"Information metrics: {cfg.compute_information_metrics}",
            f"output directory: {run_dir}"
        ]
    )
    metadata.append("="*60)

    log_simulation(metadata, cfg.log_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file for the phase diagram experiment.",
        )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_phase_diagram(cfg)
    
if __name__ == "__main__":
    main()
