#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Type Characterization Experiment

Proves that:
1. Minority Game payoffs induce mean-reverting (contrarian) strategies
2. Dollar Game payoffs induce trend-following (momentum) strategies

Method:
- Run full games with your actual Game engine
- Extract learned strategy tables from results
- Measure correlation between strategy actions and recent history or actions (m bits)
- Aggregate statistics and create publication-quality plots

This integrates with your existing codebase.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

from core.game import Game
from core.game_config import GameConfig
from utils.logger import RunLogger


# ==============================================================================
# Strategy Analysis Functions
# ==============================================================================

def decode_history_sum(pattern_index: int, memory: int) -> Tuple[int, int]:
    """
    Convert pattern index back to history sequence.
    
    Example: pattern_index=5, memory=3
    Binary: 101 → [1, 0, 1] → [-1, +1, -1] (where 0→-1, 1→+1)
    """
    binary_str = format(pattern_index, f'0{memory}b')
    history = [1 if b == '1' else -1 for b in binary_str]
    minority_sum = np.sum(history)
    implied_flow = -np.sum(history)
    return minority_sum, implied_flow


def measure_flow_correlation(strategy_table: np.ndarray, memory: int) -> float:
    """
    Measure correlation between strategy action and recent history trend.
    
    For each possible history pattern:
    - What does the strategy recommend? (±1)
    - What was the recent trend? (sign of sum of history)
    - Correlation = strategy_action × recent_trend
    
    Positive correlation → trend-following (momentum)
    Negative correlation → mean-reverting (contrarian)
    
    Returns:
        correlation: float in [-1, 1]
            +1 = perfect trend-following
            -1 = perfect mean-reversion
             0 = no pattern
    """
    correlations = []
    
    for pattern_idx in range(len(strategy_table)):
        # What does strategy recommend?
        action = strategy_table[pattern_idx]  # +1 or -1
        
        # What was recent trend (last bit of history)?
        minority_sum, flow_sum = decode_history_sum(pattern_idx, memory)
        
        # Correlation: +1 if same sign, -1 if opposite
        if flow_sum != 0:
            flow_direction = np.sign(flow_sum)
            corr = action * flow_direction
            flow_direction = np.sign(flow_sum)
            corr = action * flow_direction
        else:
            corr = 0
        
        correlations.append(corr)
    
    
    # Average correlation across all patterns
    return {
        "correlation": float(np.mean(correlations)),
        "interpretation": "Correlation with aggregate flow direction",
        "positive_means": "Momentum / Trend-following",
        "negative_means": "Contrarian / Mean-reverting"
    }

def measure_minority_correlation(strategy_table: np.ndarray, memory: int) -> Dict:
    """
    Alternative: Correlate with MINORITY sum (for comparison).
    
    This is the opposite interpretation.
    
    Positive correlation = Contrarian (match minority = oppose majority)
    Negative correlation = Momentum (oppose minority = join majority)
    """
    correlations = []
    
    for pattern_idx in range(len(strategy_table)):
        action = strategy_table[pattern_idx]
        minority_sum, _ = decode_history_sum(pattern_idx, memory)
        
        if minority_sum != 0:
            minority_direction = np.sign(minority_sum)
            corr = action * minority_direction
        else:
            corr = 0
        
        correlations.append(corr)
    
    mean_corr = float(np.mean(correlations))
    
    return {
        "correlation": mean_corr,
        "interpretation": "Correlation with minority direction",
        "positive_means": "Contrarian (matches minority)",
        "negative_means": "Momentum (opposes minority)"
    }


def comprehensive_correlation_analysis(strategy_table: np.ndarray, memory: int) -> Dict:
    """
    Compute both correlations for complete picture.
    """
    flow_result = measure_flow_correlation(strategy_table, memory)
    minority_result = measure_minority_correlation(strategy_table, memory)
    
    return {
        "flow_correlation": flow_result["correlation"],
        "minority_correlation": minority_result["correlation"],
        "memory": memory,
        "num_patterns": len(strategy_table)
    }

def analyze_strategies_from_results(
    results: Dict,
    payoff_name: str
) -> Tuple[List[float], Dict]:
    """
    Extract and analyze strategies from game results.
    
    Args:
        results: Dictionary from game.run() with "strategies" key
        payoff_name: Name of payoff scheme
    
    Returns:
        correlations: List of correlation values (one per agent)
        metadata: Dict with summary statistics
    """
    strategies = results.get("strategies")
    
    if strategies is None:
        raise ValueError(
            "No strategies found in results. "
            "Make sure record_strategies=True in GameConfig"
        )
    
    memory = results.get("memory")
    if memory is None:
        # Infer from array size
        memory = int(np.log2(strategies.shape[-1]))
    
    # Analyze each agent's strategy
    n_agents = strategies.shape[0]
    flow_correlations = []
    minority_correlations = []
    
    for i in range(n_agents):
        strategy_table = strategies[i, :]
        analysis = comprehensive_correlation_analysis(strategy_table, memory)
        flow_correlations.append(analysis['flow_correlation'])
        minority_correlations.append(analysis['minority_correlation'])
    
    # Statistical test
    flow_arr = np.array(flow_correlations)
    t_stat, p_value = stats.ttest_1samp(flow_arr, 0)
    
    metadata = {
        "payoff": payoff_name,
        "n_agents": n_agents,
        "memory": memory,
        "mean_flow_corr": np.mean(flow_arr),
        "std_flow_corr": np.std(flow_arr),
        "median_flow_corr": np.median(flow_arr),
        "t_statistic": t_stat,
        "p_value": p_value
    }
    
    return flow_correlations, minority_correlations, metadata


# ==============================================================================
# Experiment Execution
# ==============================================================================

def run_strategy_characterization(
    payoff_names: List[str] = ["BinaryMG", "ScaledMG", "DollarGame"],
    n_agents: int = 301,
    memory: int = 5,
    strategies: int = 2,
    rounds: int = 20000,
    n_runs: int = 5,
    seed: int = 42,
    logger: RunLogger = None
) -> Dict[str, Dict]:
    """
    Run complete strategy characterization experiment.
    
    Returns:
        results_by_game: Dict mapping payoff name to analysis results
    """
    print("\n" + "="*80)
    print("STRATEGY CHARACTERIZATION EXPERIMENT")
    print("Using SUM of history bits (correct approach)")
    print("="*80)
    print()
    
    logger = RunLogger(
        module="StrategyCharacterization",
        run_id="sum_based",
        seed=seed
    )
    
    logger.log_params({
        "method": "sum_of_history_bits",
        "payoffs": payoff_names,
        "n_agents": n_agents,
        "memory": memory,
        "strategies_per_agent": strategies,
        "rounds": rounds,
        "n_runs": n_runs
    })
    
    all_results = {}
    
    for payoff_name in payoff_names:
        print(f"\n{'='*70}")
        print(f"Analyzing {payoff_name}")
        print(f"{'='*70}")
        
        all_flow_corrs = []
        all_minority_corrs = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)
            
            # Build population spec
            population_spec = {
                "total": n_agents,
                "cohorts": [{
                    "count": n_agents,
                    "memory": memory,
                    "payoff": payoff_name,
                    "strategies": strategies,
                    "position_limit": 1 if payoff_name == "DollarGame" else 0,
                    "agent_type": "strategic"
                }]
            }
            
            # Configure game with strategy recording
            cfg = GameConfig(
                rounds=rounds,
                lambda_=1/(n_agents * 50),
                mm=None,
                price=100,
                seed=seed + run,
                record_agent_series=False,  # Don't need time series
                record_strategies=True       # DO need strategies
            )
            
            # Run game
            game = Game(population_spec=population_spec, cfg=cfg)
            results = game.run()
            
            # Analyze strategies
            flow_corrs, minority_corrs, metadata = analyze_strategies_from_results(
                results, payoff_name
            )
            
            all_flow_corrs.extend(flow_corrs)
            all_minority_corrs.extend(minority_corrs)
            
            print(f"Mean flow correlation: {np.mean(flow_corrs):+.3f}")
        
        # Aggregate across runs
        flow_arr = np.array(all_flow_corrs)
        t_stat, p_value = stats.ttest_1samp(all_flow_corrs, 0)
        
        all_results[payoff_name] = {
            "flow_correlations": all_flow_corrs,
            "minority_correlations": all_minority_corrs,
            "mean_flow": np.mean(flow_arr),
            "std_flow": np.std(flow_arr),
            "median_flow": np.median(flow_arr),
            "t_statistic": t_stat,
            "p_value": p_value,
            "n_agents": n_agents * n_runs,
            "memory": memory
        }
        
        print(f"\n{payoff_name} Summary:")
        print(f"  Mean correlation: {all_results[payoff_name]['mean_flow']:+.4f} ± {all_results[payoff_name]['std_flow']:.4f}")
        print(f"  Median: {all_results[payoff_name]['median_flow']:+.4f}")
        print(f"  t-statistic: {t_stat:+.2f}, p-value: {p_value:.2e}")
        
        # Classify
        mean_flow = all_results[payoff_name]['mean_flow']
        if mean_flow < -0.1 and p_value < 0.01:
            classification = "MEAN-REVERTING (contrarian) ***"
        elif mean_flow > 0.1 and p_value < 0.01:
            classification = "TREND-FOLLOWING (momentum) ***"
        elif mean_flow > 0.05 and p_value < 0.05:
            classification = "Weak pattern *"
        else:
            classification = "NEUTRAL"
        
        print(f"  → Classification: {classification}")
        
        # Save to logger if provided
        if logger:
            logger.log_metrics({
                f"{payoff_name}_mean_flow_corr": mean_flow,
                f"{payoff_name}_p_value": p_value
            })
    
    return all_results


# ==============================================================================
# Visualization
# ==============================================================================

def plot_correlation_distributions(
    all_results: Dict[str, Dict],
    save_dir: str = "plots"
) -> str:
    """
    Create violin plots showing correlation distributions.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 6), sharey=True)
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, (game_name, ax) in enumerate(zip(all_results.keys(), axes)):
        data = all_results[game_name]
        flow_corrs = data["flow_correlations"]
        
        # Violin plot
        parts = ax.violinplot(
            [flow_corrs],
            positions=[0],
            widths=0.7,
            showmeans=True,
            showmedians=True
        )
        
        # Color based on mean
        if data['mean_flow'] < -0.1:
            color = '#2ca02c'  # Green for mean-reverting
        elif data['mean_flow'] > 0.05:
            color = '#ff7f0e'  # Orange for trend-following
        else:
            color = '#1f77b4'  # Blue for neutral
        
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axhline(y=-0.1, color='green', linestyle='--', linewidth=0.5, alpha=0.3, label='Mean-revert threshold')
        ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=0.5, alpha=0.3, label='Trend-follow threshold')
        
        # Labels
        ax.set_title(game_name, fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Strategy-flow Correlation", fontsize=12, fontweight='bold')
        ax.set_xlim([-0.5, 0.5])
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([-1.1, 1.1])
        
        # Annotate statistics
        stats_text = (
            f"μ = {data['mean_flow']:+.3f}\n"
            f"σ = {data['std_flow']:.3f}\n"
            f"t = {data['t_statistic']:.1f}\n"
            f"p < {data['p_value']:.0e}"
        )
        
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
        )
        
        # Classification
        if data['mean_flow'] < -0.1 and data['p_value'] < 0.01:
            classification = "Mean-Reverting"
            text_color = 'green'
        elif data['mean_flow'] > 0.1 and data['p_value'] < 0.01:
            classification = "Trend-Following"
            text_color = 'orange'
        else:
            classification = "Neutral"
            text_color = 'blue'
        
        ax.text(
            0.5, 0.95,
            classification,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='top',
            color=text_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=text_color, linewidth=2)
        )
    
    plt.suptitle(
        "Strategy Characteristics: Flow Correlation\n(Sum of history bits approach)",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"strategy_correlations_{timestamp}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved correlation plot: {save_path}")
    return save_path


def create_summary_table(
    all_results: Dict[str, Dict],
    save_dir: str = "plots"
) -> str:
    """
    Create summary table as PDF.
    """
    # Build DataFrame
    rows = []
    for game_name, data in all_results.items():
        rows.append({
            "Game": game_name,
            "Mean ρ": f"{data['mean_flow']:+.3f}",
            "Std ρ": f"{data['std_flow']:.3f}",
            "Median ρ": f"{data['median_flow']:+.3f}",
            "t-stat": f"{data['t_statistic']:+.1f}",
            "p-value": f"{data['p_value']:.2e}",
            "N": data['n_agents']
        })
    
    df = pd.DataFrame(rows)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.15, 0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on classification
    for i, (_, row_data) in enumerate(df.iterrows(), start=1):
        game_name = row_data['Game']
        data = all_results[game_name]
        
        if data['mean_flow'] < -0.1 and data['p_value'] < 0.01:
            color = '#d4edda'  # Light green
        elif data['mean_flow'] > 0.1 and data['p_value'] < 0.01:
            color = '#fff3cd'  # Light yellow
        else:
            color = 'white'
        
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title("Strategy-History Correlation Summary", fontsize=14, fontweight='bold', pad=20)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"strategy_summary_table_{timestamp}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary table: {save_path}")
    return save_path


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("STRATEGY TYPE CHARACTERIZATION EXPERIMENT")
    print("="*70)
    print("\nProving:")
    print("  1. Minority Game → Mean-Reversion")
    print("  2. Dollar Game → Trend-Following")
    print("\nUsing REAL learned strategies from actual game runs.")
    print()
    
    # Set up logger
    logger = RunLogger(
        module="StrategyCharacterization",
        run_id="learned_strategies",
        seed=42
    )
    
    logger.log_params({
        "experiment": "strategy_type_characterization",
        "payoffs": ["BinaryMG", "ScaledMG", "DollarGame"],
        "n_agents": 301,
        "memory": 5,
        "strategies_per_agent": 2,
        "rounds": 20000,
        "n_runs": 5
    })
    
    # Run experiment
    results = run_strategy_characterization(
        payoff_names=["BinaryMG", "ScaledMG", "DollarGame"],
        n_agents=501,
        memory=7,
        strategies=2,
        rounds=1000,
        n_runs=5,
        seed=42,
        logger=logger
    )
    
    # Create visualizations
    plots_dir = logger.subdir("plots")
    
    plot_path = plot_correlation_distributions(results, save_dir=plots_dir)
    table_path = create_summary_table(results, save_dir=plots_dir)
    
    # Save correlations as CSV
    for game_name, data in results.items():
        df = pd.DataFrame({
            "correlation": data["flow_correlations"]
        })
        csv_path = logger.log_table(df, name=f"correlations_{game_name}")
        print(f"Saved {game_name} correlations: {csv_path}")
    
    # Log artifacts
    logger.log_artifact(plot_path)
    logger.log_artifact(table_path)
    
    logger.close()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {logger.get_dir()}")
    print("\nKey findings:")
    
    for game_name, data in results.items():
        print(f"\n{game_name}:")
        print(f"  Mean correlation: {data['mean_flow']:+.4f} ± {data['std_flow']:.4f}")
        
        if data['mean_flow'] < -0.1 and data['p_value'] < 0.01:
            print(f"  ✓ CONFIRMED: Mean-reverting (contrarian) strategies")
        elif data['mean_flow'] > 0.1 and data['p_value'] < 0.01:
            print(f"  ✓ CONFIRMED: Trend-following (momentum) strategies")
        else:
            print(f"  ? Inconclusive or neutral pattern")
    
    print()


if __name__ == "__main__":
    main()
