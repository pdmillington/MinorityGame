#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:18:24 2025

@author: petermillington
"""

from scipy import stats
import numpy as np

def compute_volatility(actions):
    """Compute variance of total action A(t)."""
    return np.var(actions)

def compute_scaled_volatility(actions, N):
    """Compute σ² / N."""
    return np.var(actions) / N

def compute_alpha(memory, N):
    return 2 ** memory / N

def compute_win_rate(player):
    """Compute percentage of rounds player gained points."""
    total = len(player.actions)
    wins = sum(1 for i in range(1, total) if player.points > 0)
    return wins / total if total > 0 else 0

def compute_variance_statistics(attendance: np.ndarray, N: int) -> dict:
    """
    Compute variance ratio and statistical significance.
    
    Parameters
    ----------
    attendance : array
        Attendance time series
    N : int
        Population size
        
    Returns
    -------
    dict with:
        - variance: empirical variance
        - variance_ratio: σ²/N
        - z_score: standardized deviation from null
        - p_value: two-tailed test
        - classification: 'coordinated', 'random', 'anti-coordinated'
        - confidence_interval: 95% CI for variance ratio
    """
    
    variance = np.var(attendance)
    var_ratio = variance / N
    
    # Under null hypothesis (independent ±1 choices):
    # Var(A_t) = N, so σ²/N = 1
    # 
    # For the sample variance of T observations:
    # E[s²] ≈ σ² = N
    # Var[s²] ≈ 2σ⁴/(T-1) = 2N²/(T-1)
    # 
    # So for σ²/N:
    # E[σ²/N] ≈ 1
    # Var[σ²/N] ≈ 2N²/(T-1) / N² = 2/(T-1)
    # StdDev[σ²/N] ≈ sqrt(2/(T-1))
    
    T = len(attendance)
    expected_var_ratio = 1.0
    std_var_ratio = np.sqrt(2 / (T - 1))
    
    # Z-score and p-value
    z_score = (var_ratio - expected_var_ratio) / std_var_ratio
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
    
    # 95% confidence interval
    ci_lower = var_ratio - 1.96 * std_var_ratio
    ci_upper = var_ratio + 1.96 * std_var_ratio
    
    # Classification with statistical thresholds
    # Using 2-sigma (95% CI) as threshold for significance
    threshold = 1.96 * std_var_ratio
    
    if var_ratio < 1 - threshold:
        classification = 'coordinated'
        description = 'Agents coordinate (σ²/N < 1)'
    elif var_ratio > 1 + threshold:
        classification = 'anti-coordinated'
        description = 'Crowding/herding (σ²/N > 1)'
    else:
        classification = 'random'
        description = 'Consistent with random (σ²/N ≈ 1)'
    
    return {
        'variance': variance,
        'variance_ratio': var_ratio,
        'expected_ratio': expected_var_ratio,
        'std_dev': std_var_ratio,
        'z_score': z_score,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'classification': classification,
        'description': description,
        'is_significant': p_value < 0.05,
    }
