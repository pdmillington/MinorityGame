#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 19:08:55 2026

Strategy Space Diagnostic

Analyzes whether the strategy space adequately covers realistic behavior:
1. How diverse are the strategies being used?
2. Do strategies converge or remain diverse?
3. What patterns emerge in successful strategies?
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from core.game import Game
from core.game_config import GameConfig

def analyze_strategy_coverage(m=10, N=100, s=10, rounds=10000):
    """
    Run a game and analyze which strategies are used and why.
    """
    population_spec = {
        "total": N,
        "cohorts": [{
            "count": N,
            "memory": m,
            "payoff": "ScaledMG",
            "strategies": s,
            "agent_type": "strategic"
        }]
    }
    
    cfg = GameConfig(
        rounds=rounds, 
        lambda_=0.001, 
        seed=42, 
        record_agent_series=True
    )
    
    game = Game(population_spec, cfg)
    results = game.run()
    
    # 1. Which strategies were chosen most often?
    final_strategies = [p.strategy for p in game.players]
    strategy_counts = Counter(final_strategies)
    
    print(f"\n=== Strategy Usage (m={m}, s={s}, N={N}) ===")
    print(f"Total unique strategies available per agent: {s}")
    print(f"Strategies actually used by agents: {len(strategy_counts)}")
    print(f"Most common strategy: {strategy_counts.most_common(1)[0]}")
    print(f"Strategy distribution: {dict(strategy_counts)}")
    
    # 2. Score distributions
    all_scores = np.vstack([p.scores for p in game.players])
    best_scores = all_scores.max(axis=1)
    worst_scores = all_scores.min(axis=1)
    score_spread = best_scores - worst_scores
    
    print(f"\n=== Score Analysis ===")
    print(f"Mean best score: {best_scores.mean():.2f}")
    print(f"Mean worst score: {worst_scores.mean():.2f}")
    print(f"Mean score spread: {score_spread.mean():.2f}")
    print(f"Agents with clear winner (spread > 100): {(score_spread > 100).sum()}")
    
    # 3. Wealth correlation with strategy switches
    switches = results["strategy_switches"]
    wealth = results["final_wealth"]
    correlation = np.corrcoef(switches, wealth)[0, 1]
    
    print(f"\n=== Adaptation Analysis ===")
    print(f"Mean strategy switches: {switches.mean():.1f}")
    print(f"Correlation (switches vs wealth): {correlation:.3f}")
    
    # 4. Check if strategies are truly different
    # Look at first few agents' strategies
    print(f"\n=== Strategy Patterns (first 3 agents) ===")
    for i in range(min(3, N)):
        p = game.players[i]
        print(f"\nAgent {i}:")
        print(f"  Chose strategy {p.strategy} (score: {p.scores[p.strategy]:.2f})")
        print(f"  Strategy {p.strategy} actions: {p.strategies[p.strategy][:16]}")  # First 16 bits
        print(f"  Final wealth: {p.wealth:.2f}")
    
    return {
        "game": game,
        "results": results,
        "strategy_counts": strategy_counts,
        "score_spread": score_spread,
        "correlation": correlation
    }


def compare_memory_sizes():
    """
    Compare strategy coverage across different memory sizes.
    """
    memory_sizes = [3, 5, 7, 9]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, m in enumerate(memory_sizes):
        print(f"\n{'='*60}")
        print(f"Testing m={m}")
        print('='*60)
        
        result = analyze_strategy_coverage(m=m, N=100, s=6, rounds=5000)
        
        # Plot score distributions
        ax = axes[idx]
        all_scores = np.vstack([p.scores for p in result["game"].players])
        
        ax.boxplot([all_scores[:, i] for i in range(all_scores.shape[1])])
        ax.set_title(f"m={m} (α={2**m/100:.3f})")
        ax.set_xlabel("Strategy Index")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("strategy_space_analysis.pdf")
    plt.close()
    
    print("\n=== Saved: strategy_space_analysis.pdf ===")


def test_strategy_uniqueness():
    """
    Test if agents with same memory have truly different strategies.
    """
    print("\n" + "="*60)
    print("TESTING STRATEGY UNIQUENESS")
    print("="*60)
    
    population_spec = {
        "total": 10,
        "cohorts": [{
            "count": 10,
            "memory": 4,
            "payoff": "ScaledMG",
            "strategies": 3,
            "agent_type": "strategic"
        }]
    }
    
    cfg = GameConfig(rounds=100, lambda_=0.001, seed=42)
    game = Game(population_spec, cfg)
    
    # Check if agents have different strategies
    print("\nStrategy banks (first 3 agents, first strategy each):")
    for i in range(min(3, len(game.players))):
        print(f"Agent {i}, Strategy 0: {game.players[i].strategies[0]}")
    
    # Are they unique?
    strategy_sigs = [tuple(p.strategies[0]) for p in game.players]
    unique_sigs = len(set(strategy_sigs))
    
    print(f"\nUnique strategy signatures: {unique_sigs}/{len(game.players)}")
    if unique_sigs == len(game.players):
        print("✓ All agents have different strategies (GOOD)")
    else:
        print("✗ Some agents have identical strategies (PROBLEM)")


if __name__ == "__main__":
    print("STRATEGY SPACE DIAGNOSTIC")
    print("="*60)
    
    # Test 1: Basic coverage
    analyze_strategy_coverage(m=5, N=100, s=10, rounds=5000)
    
    # Test 2: Uniqueness
    test_strategy_uniqueness()
    
    # Test 3: Compare memory sizes
    compare_memory_sizes()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
