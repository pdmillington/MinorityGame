#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:56:40 2025

Plot Utilities Module

Provides reusable figure creation functions and optional save wrappers.

Pattern:
    - create_*_figure() -> returns plt.Figure (for use with ReportBuilder)
    - plot_*() -> creates, saves, closes (backward compatible convenience)

@author: petermillington
"""
from __future__ import annotations
import os
import math
from typing import Optional, Sequence, Mapping

import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
import matplotlib.pyplot as plt

# Utility Functions

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def _build_filepath(save_dir: str,
                     filename_stub: str,
                     filename_prefix: Optional[str] = None,
                     ) -> str:
    bits = [filename_stub]
    if filename_prefix:
        bits.insert(0, filename_prefix)
    filename = "_".join(bits) + ".pdf"
    return os.path.join(save_dir, filename)

def _save_figure(fig: plt.Figure,
                 save_dir: str,
                 filename_stub: str,
                 filename_prefix: Optional[str] = None
                 ) -> str:
    """Save figure and return path."""
    ensure_dir(save_dir)
    path = _build_filepath(save_dir, filename_stub, filename_prefix)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
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

def _annotate_stats(ax: plt.Axes, prices: np.ndarray) -> None:
    """Add statistics annotation box to axis."""
    try:
        stats = stat_summary(prices)
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
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='gray',
                alpha=0.8,
                linewidth=0.5
            )
        )
    except Exception:
        pass  # Skip annotation if stats fail


def create_price_figure(
    prices: Sequence[float],
    title: str,
    ylabel: str = "Price",
    annotate_stats: bool = True,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create price time series figure.
    
    Parameters
    ----------
    prices : sequence of float
        Price series
    title : str
        Figure title
    ylabel : str
        Y-axis label
    annotate_stats : bool
        Whether to add statistics annotation box
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    p = np.array(prices, dtype=float)
    x = np.arange(len(p))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, p, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)

    if annotate_stats:
        _annotate_stats(ax, p)

    fig.tight_layout()
    return fig


def create_series_figure(
    series: Sequence[float],
    title: str,
    xlabel: str = "Round",
    ylabel: str = "Value",
    annotate_stats: bool = False,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create generic time series figure.
    """
    s = np.array(series, dtype=float)
    x = np.arange(len(s))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, s, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)

    if annotate_stats and np.all(s > 0):
        _annotate_stats(ax, s)
    
    fig.tight_layout()
    return fig


def create_boxplot_figure(
    metric_by_cohort: Mapping[int, Sequence[float]],
    labels: Mapping[int, str],
    title: str,
    ylabel: str,
    xlabel: str = "Cohort",
    rotation: int = 30,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create boxplot of per-agent metric grouped by cohort.
    
    Parameters
    ----------
    metric_by_cohort : mapping
        cid -> sequence of values
    labels : mapping
        cid -> display label string
    title : str
        Figure title
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    rotation : int
        X-tick label rotation
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    cids = sorted(metric_by_cohort.keys())
    data = [metric_by_cohort[cid] for cid in cids]
    tick_labels = [labels.get(cid, f"cohort {cid}") for cid in cids]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, labels=tick_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if rotation:
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation)
            tick.set_ha("right")
    
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def create_mean_line_figure(
    metric_by_cohort: Mapping[int, Sequence[float]],
    labels: Mapping[int, str],
    title: str,
    ylabel: str,
    xlabel: str = "Cohort",
    show_std_band: bool = True,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create mean + optional std band line plot by cohort.
    
    Returns
    -------
    plt.Figure
    """
    cids = sorted(metric_by_cohort.keys())
    means = np.array([np.mean(metric_by_cohort[cid]) for cid in cids], float)
    stds = np.array([np.std(metric_by_cohort[cid]) for cid in cids], float)

    x = np.arange(len(cids))
    xticks = [labels.get(cid, f"cohort {cid}") for cid in cids]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, means, marker="o", lw=1.0)
    
    if show_std_band:
        ax.fill_between(x, means - stds, means + stds, alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def create_histogram_figure(
    data: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    bins: int = 30,
    annotate_stats: bool = False,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create histogram figure.
    """
    h = np.asarray(data, float)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(h, bins=bins, alpha=0.75, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    if annotate_stats:
        stats_text = f"μ: {np.mean(h):.2f}\nσ: {np.std(h):.2f}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=7,
            va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8, linewidth=0.5)
        )
    
    fig.tight_layout()
    return fig


def create_scatter_figure(
    x_data: Sequence[float],
    y_data: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    annotate_corr: bool = True,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create scatter plot figure.
    """
    x = np.asarray(x_data, float)
    y = np.asarray(y_data, float)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=18, marker='o', alpha=0.6, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    if annotate_corr and len(x) > 2:
        try:
            corr, _ = pearsonr(x, y)
            ax.text(
                0.02, 0.98, f"ρ: {corr:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8, linewidth=0.5)
            )
        except Exception:
            pass
    
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Grid Plot Helpers (for multi-panel layouts)
# -----------------------------------------------------------------------------

def create_grid_figure(
    items: list,
    plot_fn: callable,
    titles: list[str],
    suptitle: str,
    xlabel: str,
    ylabel: str,
    items2: Optional[list] = None,
    stat_sum: bool = False,
    figsize_per_panel: tuple = (4, 3),
) -> plt.Figure:
    """
    Create a grid of subplots.
    
    Parameters
    ----------
    items : list
        Data for each panel (passed to plot_fn as y_data)
    plot_fn : callable
        Function(ax, title, y_data, stat_sum, x_data=None) to draw each panel
    titles : list of str
        Title for each panel
    suptitle : str
        Overall figure title
    xlabel, ylabel : str
        Axis labels (applied to bottom row / left column)
    items2 : list, optional
        Secondary data for each panel (passed as x_data)
    stat_sum : bool
        Whether to annotate statistics
    figsize_per_panel : tuple
        Size per panel for computing overall figure size
        
    Returns
    -------
    plt.Figure
    """
    n = len(items)
    rows, cols = best_grid(n)
    
    figsize = (figsize_per_panel[0] * cols, figsize_per_panel[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    if items2 is not None and len(items2) != n:
        raise ValueError("items and items2 must have same length")
    if titles is not None and len(titles) != n:
        raise ValueError("titles and items must have same length")

    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        x_item = items2[idx] if items2 else None
        plot_fn(ax=ax, title=titles[idx], y_data=item, stat_sum=stat_sum, x_data=x_item)

        if r == rows - 1:
            ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel(ylabel)

    # Hide unused panels
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r, c])

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# Panel plot functions for use with create_grid_figure
def panel_series(ax, title, y_data, stat_sum, x_data=None):
    """Draw a series panel."""
    s = np.asarray(y_data, float)
    x = x_data if x_data is not None else np.arange(len(s))
    ax.plot(x, s, lw=0.5)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    if stat_sum and np.all(s > 0):
        _annotate_stats(ax, s)


def panel_histogram(ax, title, y_data, stat_sum, x_data=None):
    """Draw a histogram panel."""
    h = np.asarray(y_data, float)
    ax.hist(h, bins=30, alpha=0.75, edgecolor='black')
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    if stat_sum:
        stats_text = f"μ: {np.mean(h):.2f}\nσ: {np.std(h):.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8, linewidth=0.5))


def panel_scatter(ax, title, y_data, stat_sum, x_data=None):
    """Draw a scatter panel."""
    x = np.asarray(x_data, float)
    y = np.asarray(y_data, float)
    ax.scatter(x, y, s=18, marker='o', alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    if stat_sum and len(x) > 2:
        try:
            corr, _ = pearsonr(x, y)
            ax.text(0.02, 0.98, f"ρ: {corr:.3f}", transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.8, linewidth=0.5))
        except Exception:
            pass


def panel_boxplot(ax, title, y_data, stat_sum, x_data=None):
    """Draw a boxplot panel (y_data should be dict: cid -> values)."""
    cids = sorted(y_data.keys())
    data = [y_data[cid] for cid in cids]
    x_labels = [f"{cid}" for cid in cids]
    ax.boxplot(data, labels=x_labels)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)


# -----------------------------------------------------------------------------
# Backward-Compatible Save Wrappers
# -----------------------------------------------------------------------------

def plot_price_graph(
    prices: Sequence[float],
    title: str,
    ylabel: str = "Price",
    save_dir: str = "plots",
    filename_stub: str = "price_series",
    filename_prefix: Optional[str] = None,
    annotate_stats: bool = True,
) -> str:
    """Create, save, and close price figure. Returns path."""
    fig = create_price_figure(prices, title, ylabel, annotate_stats)
    path = _save_figure(fig, save_dir, filename_stub, filename_prefix)
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
    rotation: int = 30,
) -> str:
    """Create, save, and close boxplot figure. Returns path."""
    fig = create_boxplot_figure(metric_by_cohort, labels, title, ylabel, rotation=rotation)
    path = _save_figure(fig, save_dir, filename_stub, filename_prefix)
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
    """Create, save, and close mean line figure. Returns path."""
    fig = create_mean_line_figure(metric_by_cohort, labels, title, ylabel, show_std_band=show_std_band)
    path = _save_figure(fig, save_dir, filename_stub, filename_prefix)
    plt.close(fig)
    return path
