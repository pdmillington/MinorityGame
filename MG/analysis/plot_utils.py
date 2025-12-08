#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:56:40 2025

@author: petermillington
"""
from __future__ import annotations
import os
import math
from typing import Dict, List, Iterable, Optional, Sequence, Mapping, Any

import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter, AutoMinorLocator

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def plot_file_path(filename_stub,
              filename_prefix,
              save_dir
              ):
    bits = [filename_stub]
    if filename_prefix:
        bits.insert(0, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    path = os.path.join(save_dir, filename)
    return path

def best_grid(n: int) -> tuple[int, int]:
    """
    Choose the near square grid for n panels, eg 4 -> (2,2).
    """
    if n<=0:
        raise ValueError("n must be positive.")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def stat_summary(prices, periods_per_year=252):
    """
    Calculate statistical measures from a price series.
    
    Parameters:
    -----------
    prices : array-like
        Time series of prices
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252 for daily data)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - avg_ret: Mean log return (annualized)
        - vol: Volatility/standard deviation (annualized)
        - skew_ret: Skewness of log returns
        - kurt_ret: Excess kurtosis of log returns
        - n_obs: Number of return observations
    """
    # Input validation
    prices = np.asarray(prices)

    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations")

    if np.any(prices <= 0):
        raise ValueError("Prices must be positive for log returns")

    # Calculate log returns
    returns = np.diff(np.log(prices))

    # Handle NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("No valid returns after removing NaN values")

    # Calculate statistics (annualized)
    stats = {
        'avg_ret': np.mean(returns) * periods_per_year,  # Annualized mean return
        'vol': np.std(returns, ddof=1) * np.sqrt(periods_per_year),  # Annualized volatility
        'skew_ret': skew(returns),  # Skewness (no annualization)
        'kurt_ret': kurtosis(returns, fisher=True),  # Excess kurtosis (no annualization)
        'n_obs': len(returns)
    }

    return stats

def plot_population_grid(
        items: List[Sequence[float]],
        plot_fn: str,
        suptitle: str,
        save_dir: str,
        x_label: str,
        y_label: str,
        filename_stub: str,
        stat_sum: Optional[bool] = False,
        filename_prefix: Optional[str] = None,
        items2: Optional[List[Sequence[float]]] = None,
        titles: Optional[List[str]] = None,
        ) -> str:
    """
    Plot a list of 1D series in near grid of subplots.
    Each panel shows one series and its title
    """
    ensure_dir(save_dir)
    
    n = len(items)

    rows, cols = best_grid(n)
    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(4 * cols, 3 * rows),
                             sharex=False,
                             sharey=False,
                             squeeze=False)
    
    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        ax=axes[r,c]

        if items2 is None:
            plot_fn(ax=ax, title=titles[idx], y_data=item, stat_sum=stat_sum)
        else:
            x_item = items2[idx]
            plot_fn(ax=ax, title=titles[idx], x_data=x_item, y_data=item, stat_sum=stat_sum)
            
        if titles is not None:
            ax.set_title(titles[idx])
            
        if r == rows - 1:
            ax.set_xlabel(x_label)
        if c == 0:
            ax.set_ylabel(y_label)
            
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r, c])
            
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = plot_file_path(filename_stub=filename_stub,
                   filename_prefix=filename_prefix,
                   save_dir=save_dir)
    fig.savefig(path, format="pdf", dpi=300)
    plt.close(fig)
    return path
    
def plot_ms_series_grid(
        series_list: dict[tuple[int, int], Sequence[float]],
        m_values: Sequence[int],
        s_values: Sequence[int],
        suptitle: str,
        save_dir: str,
        filename_stub: str,
        filename_prefix: Optional[str] = None,
        lw: float = 0.5,
        ) -> str:
    """
    Specialist helper to plot series over grid for m x s for non-varying population
    """
    pass

def plot_series(ax, title, y_data, stat_sum, x_data=None):
    """
    Helper function for series to be placed on grid.
    """
    s = np.asarray(y_data, float)
    if x_data:
        x = x_data
    else:
        x = np.arange(len(s))
    ax.plot(x, s, lw=0.5)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    if stat_sum:
        try:
            stats = stat_summary(y_data)
            if all(k in stats for k in ("avg_ret", "vol", "skew_ret", "kurt_ret")):
                stats_text = (
                    f"μ: {stats['avg_ret']*100:.2f}%\n"
                    f"σ: {stats['vol']*100:.1f}%\n"
                    f"skew: {stats['skew_ret']:.2f}\n"
                    f"kurt: {stats['kurt_ret']:.2f}"
                )
                ax.text(
                    0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    fontsize=7,
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             edgecolor='gray',
                             alpha=0.8,
                             linewidth=0.5)
                )
        except Exception:
            pass
    
def plot_hist(ax, title, y_data, stat_sum, x_data=None):
    """
    Helper function for histogram to be placed on grid.
    """
    h = np.asarray(y_data, float)
    ax.hist(h, bins=30, alpha=0.75, edgecolor='black')
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    
def plot_scatter(ax, title, y_data, stat_sum, x_data=None):
    """
    Helper function for scatter to be placed on grid.
    """
    x = np.asarray(x_data, float)
    y = np.asarray(y_data, float)
    ax.scatter(x, y, s=18, marker='o', alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha= 0.6)
    
def plot_price_graph(prices: Sequence[float],
                     title: str,
                     ylabel: str = "Price",
                     save_dir: str = "plots",
                     filename_stub: str = "price_series",
                     filename_prefix: Optional[str] = None,
                     annotate_stats: bool = True,
                     ) -> str:
    """
    Build a grid of price time series plots and return the Matplotlib Figure.
    This function performs no filesystem I/O (no save).
    """
    ensure_dir(save_dir)

    p = np.array(prices, dtype=float)
    x = np.arange(len(p))

    # Always create a 2D array
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, p, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    if annotate_stats:
        try:
            stats = stat_summary(p)
            if all(k in stats for k in ("avg_ret", "vol", "skew_ret", "kurt_ret")):
                stats_text = (
                    f"μ: {stats['avg_ret']*100:.2f}%\n"
                    f"σ: {stats['vol']*100:.1f}%\n"
                    f"skew: {stats['skew_ret']:.2f}\n"
                    f"kurt: {stats['kurt_ret']:.2f}"
                )
                ax.text(
                    0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    fontsize=7,
                    va='top',
                    bbox={"boxstyle": 'round, pad=0.3',
                          "facecolor": 'white',
                          "edgecolor": 'gray',
                          "alpha": 0.8,
                          "linewidth": 0.5
                          },
                    )
        except Exception as e:
            print("stat annotation error", e)

    path = plot_file_path(filename_stub,
                          filename_prefix,
                          save_dir)
    fig.savefig(path, format="pdf", dpi=300)
    plt.close(fig)
    return path

def plot_metric_boxplot_by_cohort(
        metric_by_cohort: Mapping[int, Sequence[float]],
        labels: Mapping[int, str],
        title: str,
        ylabel: str,
        save_dir: str,
        filename_stub: str,
        filename_prefix: Optional[str] = None,
        rotation: int = 0,
        ) -> str:
    """
    Executes a boxplot of per_agent metric grouped by cohort.
    metric_by_cohort: cid -> sequence of values
    labels: cid -> multiline label string
    Returns a path to the saved file.
    """
    ensure_dir(save_dir)

    cids = sorted(metric_by_cohort.keys())
    data = [metric_by_cohort[cid] for cid in cids]
    tick_labels = [labels.get(cid, f"cohort {cid}") for cid in cids]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=tick_labels)
    ax.set_title(title)
    ax.set_xlabel("Cohort")
    ax.set_ylabel(ylabel)
    if rotation:
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation)
            tick.set_ha("right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    path = plot_file_path(filename_stub,
                          filename_prefix,
                          save_dir)
    fig.savefig(path, format="pdf", dpi=300)
    plt.close(fig)
    return path

def plot_metric_mean_line_by_cohort(
    metric_by_cohort: Mapping[int, Sequence[float]],
    labels: Mapping[int, str],
    title: str,
    ylabel: str,
    save_dir: str,
    filename_stub: str,
    filename_prefix: Optional[str] = None,
    show_std_band: bool = True,
) -> str:
    """
    Mean + optional std band of a metric by cohort.
    x-axis is cohorts (categorical), displayed as 0..K-1 with text labels.
    """
    ensure_dir(save_dir)
    
    cids = sorted(metric_by_cohort.keys())
    means = np.array([np.mean(metric_by_cohort[cid]) for cid in cids], float)
    stds  = np.array([np.std(metric_by_cohort[cid])  for cid in cids], float)
    
    x = np.arange(len(cids))
    xticks = [labels.get(cid, f"cohort {cid}") for cid in cids]
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x, means, marker="o", lw=1.0)
    if show_std_band:
        ax.fill_between(x, means - stds, means + stds, alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Cohort")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    
    path = plot_file_path(filename_stub,
                          filename_prefix,
                          save_dir)
    fig.savefig(path, format="pdf", dpi=300)
    plt.close(fig)
    return path
