#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 16:50:29 2026

@author: petermillington
"""

import os
import numpy as np
from scipy.stats import chisquare, entropy
from scipy.stats import gamma as gamma_dist
from scipy.stats import chi2 as chi2_dist
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Patch
from analysis.cohort_utils import summarize_population, format_population_summary
from analysis.stats import compute_variance_statistics

def binary_history_to_index(history: np.ndarray) -> int:
    """Convert binary history [-1,1] to integer index [0, 2^m-1]"""
    binary = (history == 1).astype(int)
    return int(''.join(map(str, binary)), 2)

# Visualisation helpers
# Colour thresholds for mutual information interpretation
_MI_STRONG  = 0.05  # bits: clear predictive structure
_MI_WEAK    = 0.01  # bits for marginal predictive structure
_PRED_HIGH  = 0.30  # fraction of histories that are predictable
_PRED_MED   = 0.10  # medium predictability


def _mi_colour(mi: float) -> str:
    """Traffic light colour scheme for MI value"""
    if mi >= _MI_STRONG:
        return "#90EE90"  # light green
    if mi >= _MI_WEAK:
        return "#FFD580"  # amber
    return "#FFB3B3"  # light red


def _pred_colour(pred:float) -> str:
    """Traffic light colour for a predictability fraction"""
    if pred > _PRED_HIGH:
        return "#90EE90"
    if pred >= _PRED_MED:
        return "#FFD580"  # amber
    return "#FFB3B3"  # light red

def _emh_colour(info: dict) -> str:
    """Traffic light colour for EMH rejection level"""
    if info.get('reject_emh_99'):
        return "#90EE90"   #Green strong rejection
    if info.get('reject_emh_95'):
        return "#FFD580"  # amber, moderate rejection
    if info.get('reject_emh_90'):
        return "#FFE5B4"  #pale amber, weak rejection
    return "#FFB3B3"  # light red


def _delta_colour(delta: float) -> str:
    if delta > _MI_WEAK:
        return "#FFD580"  #inaccessible info exists, amber
    return "#90EE90"  #green, agents capturing available info


def _mi_verdict(mi: float) -> str:
    if mi >= _MI_STRONG:
        return "Strong predictive structure\n - history contains exploitable information"
    if mi >= _MI_WEAK:
        return "Weak predictive structure\n - marginal signal, perhaps noise driven"
    return "Market unpredictable at this window"
    
def _pred_verdict(pred: float) -> str:
    if pred >= _PRED_HIGH:
        return "Many history patterns\n deviate significantly from p = 0.5"
    if pred >= _PRED_MED:
        return "Some history patterns show\n statistically significant deviations"
    return "Very few, or no, history patterns\n contain predictions"

    
def _emh_verdict(info: dict,) -> str:
    """Plain-English EMH test verdict."""
    if info.get('reject_emh_99'):
        return f"EMH rejected at 99%\nG={info['G_statistic']:.3f}, p={info['G_p_value']:.4f}"
    if info.get('reject_emh_95'):
        return f"EMH rejected at 95%\nG={info['G_statistic']:.3f}, p={info['G_p_value']:.4f}"
    if info.get('reject_emh_90'):
        return f"EMH rejected at 90% only\nG={info['G_statistic']:.3f}, p={info['G_p_value']:.4f}"
    return f"Cannot reject EMH\nG={info['G_statistic']:.3f}, p={info['G_p_value']:.4f}"



def _delta_verdict(delta: float) -> str:
    if delta > _MI_WEAK:
        return (
            "Exploitable structure exists\n beyond agent memory.\n"
            "  Longer-memory strategies may\n outperform current agents."
        )
    return (
        "Agents appear to capture most\n available information.\n"
        "  Extending memory is unlikely\n to improve performance significantly."
    )


def _draw_coloured_box(
        ax: plt.Axes,
        x: float, y:float,
        width: float, height: float,
        label: str, value_str: str,
        verdict: str, colour: str,
        fontsize: int = 9,
        ) -> None:
    """Draw a labelled interpretation box on given coordinates (ax)"""
    patch = FancyBboxPatch(
        (x,y), width, height,
        boxstyle="round, pad=0.01",
        linewidth=1.2,
        edgecolor="#555555",
        facecolor=colour,
        transform=ax.transAxes,
        clip_on=False,
        zorder=2,
        )
    ax.add_patch(patch)
    cx = x + width / 2
    # Label at top of box
    ax.text(cx, y + height - 0.018, label,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize, fontweight="bold", zorder=3)
    # Value in centre
    ax.text(cx, y + height/2 + 0.005, value_str,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize + 1, fontweight="bold", zorder=3)
    # Verdict at bottom
    ax.text(cx, y + 0.012, verdict,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize - 1, style="italic",
            wrap=True, zorder=3)


def compute_emh_critical_values(k: int, n: int, 
                                 confidence_levels: list = [0.90, 0.95, 0.99, 0.999]) -> dict:
    """
    Compute critical values for the Brouty-Garcin EMH test.
    
    Under EMH, the market information estimator follows asymptotically:
    Gamma(shape=2^(L-1), scale=1/(ln(2)*n))
    
    Parameters
    ----------
    k : int
        Memory window length (L in Brouty-Garcin notation)
    n : int
        Number of observations in the window
    confidence_levels : list
        Confidence levels for critical values
        
    Returns
    -------
    dict with critical values and whether EMH is rejected
    """
    shape = 2 ** (k - 1)
    scale = 1.0 / (np.log(2) * n)
    
    critical_values = {}
    for conf in confidence_levels:
        critical_values[conf] = gamma_dist.ppf(conf, a=shape, scale=scale)
    
    return {
        'shape': shape,
        'scale': scale,
        'critical_values': critical_values,
        'mean': shape * scale,        # Expected value under EMH
        'std': np.sqrt(shape) * scale # Std dev under EMH
    }


def test_emh(market_information: float, k: int, n: int) -> dict:
    """
    Test whether EMH can be rejected given observed market information.
    
    Parameters
    ----------
    market_information : float
        Observed market information (Brouty-Garcin measure)
    k : int
        Memory window length
    n : int
        Number of observations
        
    Returns
    -------
    dict with p-value and rejection decisions
    """
    shape = 2 ** (k - 1)
    scale = 1.0 / (np.log(2) * n)
    
    # p-value: probability of observing this MI or higher under EMH
    p_value = 1 - gamma_dist.cdf(market_information, a=shape, scale=scale)
    
    cv = compute_emh_critical_values(k, n)
    
    return {
        'market_information': market_information,
        'p_value': p_value,
        'reject_90': market_information > cv['critical_values'][0.90],
        'reject_95': market_information > cv['critical_values'][0.95],
        'reject_99': market_information > cv['critical_values'][0.99],
        'reject_999': market_information > cv['critical_values'][0.999],
        'critical_values': cv['critical_values'],
        'emh_mean': cv['mean'],
        'emh_std': cv['std']
    }



def compute_history_statistics(attendance: np.ndarray, 
                                window: int) -> Dict:
    """
    Analyze predictive power of binary history.
    
    Parameters
    ----------
    attendance : array of shape (rounds,)
        Attendance time series
    memory : int
        History window size
        
    Returns
    -------
    dict containing:
        - transition_counts: dict mapping history_index -> {-1: count, 1: count}
        - chi_squared: test statistic for deviation from 0.5
        - p_value: significance
        - mutual_information: bits of information history provides about next outcome
        - predictability: fraction of histories with P(outcome) significantly != 0.5
    """
    # Convert attendance to binary history
    k = window
    history = np.sign(attendance)
    
    # Build transition table
    transition_counts = {}
    for t in range(k, len(history)):
        h = history[t-k:t]
        idx = binary_history_to_index(h)
        outcome = history[t]
        
        if idx not in transition_counts:
            transition_counts[idx] = {-1: 0, 1: 0}
        transition_counts[idx][outcome] += 1
    
    # Compute statistics
    results = {
        'transition_counts': transition_counts,
        'window': k,
        'num_histories': len(transition_counts),
    }
    
    # Chi-squared test: do probabilities deviate from 0.5?
    chi2_stats = []
    p_values = []
    predictable_count = 0
    
    for idx, counts in transition_counts.items():
        total = counts[-1] + counts[1]
        if total < 10:  # Skip rare histories
            continue
            
        expected = total / 2
        observed = [counts[-1], counts[1]]
        chi2, p = chisquare(observed, [expected, expected])
        chi2_stats.append(chi2)
        p_values.append(p)
        
        if p < 0.05:  # Significantly different from 0.5
            predictable_count += 1
    
    results['predictability'] = predictable_count / len(transition_counts) if transition_counts else 0
    
    # Mutual information: I(history; next_outcome)
    # This measures how many bits of information the history provides
    mi, nmi, market_information = compute_mutual_information(transition_counts, window)
    
    # Global MI chi-squared test
    total = sum(c[-1] + c[1] for c in transition_counts.values())
    G = 2 * total * np.log(2) * mi  # G-test statistic
    df = 2**k - 1                    # degrees of freedom
    p_global = 1 - chi2_dist.cdf(G, df=df)
    
    results['mutual_information'] = mi
    results['nmi'] = nmi
    results['market_information'] = market_information
    results['G_statistic'] = G
    results['G_critical_90'] = chi2_dist.ppf(0.90, df=df) / (2 * total * np.log(2))
    results['G_critical_95'] = chi2_dist.ppf(0.95, df=df) / (2 * total * np.log(2))
    results['G_critical_99'] = chi2_dist.ppf(0.99, df=df) / (2 * total * np.log(2))
    results['G_df'] = df
    results['G_p_value'] = p_global
    results['reject_emh_90'] = p_global < 0.10
    results['reject_emh_95'] = p_global < 0.05
    results['reject_emh_99'] = p_global < 0.01

    return results


def compute_mutual_information(transition_counts: Dict, k: int) -> float:
    """
    Compute I(H; A_{t+1}) and NMI where H is history, A is next attendance sign.
    
    MI = sum_h sum_a P(h,a) log(P(h,a) / (P(h)P(a)))
    """
    # Get marginals
    total_count = sum(c[-1] + c[1] for c in transition_counts.values())
    
    # P(a)
    p_minus = sum(c[-1] for c in transition_counts.values()) / total_count
    p_plus = 1 - p_minus
    
    # Marginal entropy of outcome
    h_a = 0.0
    for p in [p_minus, p_plus]:
        if p > 0:
            h_a -= p *np.log2(p)
    
    mi = 0.0
    h_h = 0.0    # Marginal entropy of history
    h_joint = 0.0
    
    for idx, counts in transition_counts.items():
        p_h = (counts[-1] + counts[1]) / total_count
        if p_h > 0:
            h_h -= p_h * np.log2(p_h)
        for outcome in [-1, 1]:
            if counts[outcome] == 0:
                continue
                
            p_ha = counts[outcome] / total_count
            p_a = p_minus if outcome == -1 else p_plus
            h_joint -= p_ha * np.log2(p_ha)
            
            mi += p_ha * np.log2(p_ha / (p_h * p_a))
    
    joint_entropy = np.sqrt (h_h * h_a)

    nmi = mi / joint_entropy if joint_entropy > 0 else 0.0
    
    # Brouty-Garcin market information measure
    h_max = k + 1  # maximum possible joint entropy
    market_information = h_max - h_joint
    
    return mi, nmi,market_information


def plot_transition_probabilities(transition_counts: Dict,
                                   memory: int,
                                   save_path: str = None):
    """
    Visualize P(A_{t+1}=1 | history) for each history.
    """
    indices = sorted(transition_counts.keys())
    probs = []
    counts_total = []
    
    for idx in indices:
        c = transition_counts[idx]
        total = c[-1] + c[1]
        prob_plus = c[1] / total if total > 0 else 0.5
        probs.append(prob_plus)
        counts_total.append(total)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: probability of +1 given history
    ax1.bar(indices, probs, alpha=0.7)
    ax1.axhline(0.5, color='red', linestyle='--', label='Random (p=0.5)')
    ax1.set_xlabel('History index')
    ax1.set_ylabel('P(A_{t+1}=1 | history)')
    ax1.set_title(f'Predictive power of m={memory} history')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Bottom: sample size for each history
    ax2.bar(indices, counts_total, alpha=0.7, color='gray')
    ax2.set_xlabel('History index')
    ax2.set_ylabel('Observation count')
    ax2.set_title('Sample size per history')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
    return fig

def create_summary_page(stats: Dict, save_path: str):
    """Create a standalone PDF page with statistical summary."""
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Letter size
    ax.axis('off')
    
    window = stats['window']
    
    # Title
    title_text = f"Information Analysis Summary\nMemory Window: k = {window}"
    ax.text(0.5, 0.95, title_text,
            ha='center', va='top',
            fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    y_pos = 0.88
    
    #Population parameters if available
    if 'population_summary' in stats:
        pop_summary = stats['population_summary']
        n_cohorts = len(pop_summary['cohorts'])
        
        #Adjust font size depending on the number of cohorts
        if n_cohorts <= 3:
            pop_fontsize = 9
            spacing_needed = 0.2
        if n_cohorts <= 6:
            pop_fontsize = 8
            spacing_needed = 0.26
        else:
            pop_fontsize = 7
            spacing_needed = 0.32
        
        pop_text = "POPULATION STRUCTURE\n" + "-" * 60 + "\n"
        pop_text += format_population_summary(pop_summary)
        
        ax.text(0.1, y_pos, pop_text,
                ha='left', va='top',
                fontsize=pop_fontsize,
                family='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue',
                          alpha=0.3, pad=0.8))
        y_pos -= spacing_needed
        
    variance_section = ""
    if 'variance_statistics' in stats:
        vs = stats['variance_statistics']
        
        # Determine color based on classification
        if vs['classification'] == 'coordinated':
            var_color = 'lightgreen'
        elif vs['classification'] == 'anti-coordinated':
            var_color = 'lightcoral'
        else:
            var_color = 'lightyellow'
            
        variance_section = f"""
        VARIANCE ANALYSIS
        {'─' * 60}
        
        Variance Ratio (σ²/N):         {vs['variance_ratio']:.4f}
        95% Confidence Interval:       [{vs['ci_lower']:.4f}, {vs['ci_upper']:.4f}]
        Z-score:                       {vs['z_score']:.2f} (p={vs['p_value']:.4f})

        {vs['description']}
        {'✓ Statistically significant' if vs['is_significant'] else '✗ Not significant (p > 0.05)'}
        
        """
        ax.text(0.1, y_pos, variance_section,
                ha='left', va='top',
                fontsize=9,
                family='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor=var_color,
                          alpha=0.3, pad=0.6))
        y_pos -= 0.18
    
    # Main statistics
    main_stats = f"""
    KEY METRICS(Analysis window: k = {window})
    {'─' * 60}
    Unique Histories Observed:     {stats['num_histories']} / {2**window} ({100*stats['num_histories']/2**window:.1f}%)
    Mutual Information:            {stats['mutual_information']:.4f} bits
    Normalised Mutual Information: {stats['nmi']:.4f} bits
    Market Information (B-G):      {stats['market_information']:.4f} bits
    G-statistic:                   {stats['G_statistic']:.4f}
    EMH test p-value:              {stats['G_p_value']:.4f}
    Reject EMH at 90%:             {'✓' if stats['reject_emh_90'] else '✗'}
    Reject EMH at 95%:             {'✓' if stats['reject_emh_95'] else '✗'}  
    Reject EMH at 99%:             {'✓' if stats['reject_emh_99'] else '✗'}
    Predictable Histories:         {stats['predictability']:.1%}
    """
    
    ax.text(0.1, y_pos, main_stats,
            ha='left', va='top',
            fontsize=10,
            family='monospace',
            transform=ax.transAxes)
    y_pos -=0.2
    
    # Interpretation
    if stats.get('reject_emh_95'):
        interp = "✓ EMH rejected at 95% confidence\n"
        interp += f"  G = {stats['G_statistic']:.3f}, p = {stats['G_p_value']:.4f}"
        color = 'darkgreen'
        box_color = 'lightgreen'
    elif stats.get('reject_emh_90'):
        interp = "~ EMH rejected at 90% confidence only\n"
        interp += f"  G = {stats['G_statistic']:.3f}, p = {stats['G_p_value']:.4f}"
        color = 'darkorange'
        box_color = 'lightyellow'
    else:
        interp = "✗ Cannot reject EMH\n"
        interp += f"  G = {stats['G_statistic']:.3f}, p = {stats['G_p_value']:.4f}"
        color = 'darkred'
        box_color = 'lightcoral'

    ax.text(0.1, y_pos, interp,
            ha='left', va='top',
            fontsize=10,
            color=color,
            weight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.6, pad=0.8))
    
    y_pos -= 0.15

    # Footnotes
    footnote = """
    Notes:
    • Mutual Information quantifies bits of information history provides about next outcome
    • Predictable fraction: histories where P(outcome=+1) significantly ≠ 0.5 (p < 0.05)
    • MI measures predictability not coordination
    • Normalized variance σ²/N measures coordination: <1 coordinated, ≈1 uncorrelated, >1 crowding
    • Statistical significance uses 2-tailed test with analytical standard error
    • α_eff = 2^(Σ w_i m_i) / N,  α_weighted = (Σ n_i 2^m_i) / N²
    """
    
    ax.text(0.1, 0.08, footnote,
            ha='left', va='top',
            fontsize=7.5,
            style='italic',
            color='gray',
            transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Information summary page
def create_information_summary_figure(
        information_results: List[dict],
        base_m: int,
        mi_at_m: float,
        mi_at_m1: float,
        ) -> plt.Figure:
    """
    Produce a rich A4 summary page for the information-metrics section.
    
    Contains:
    - Per-window statistics table (MI, G-statistic, EMH rejection)
    - Colour-coded interpretation boxes (MI, predictability, inaccessible info)
    - Plain-English narrative paragraph
    
    Parameters
    ----------
    information_results : list of dicts from compute_history_statistics
    base_m : int — base agent memory window
    mi_at_m : float — MI at window = base_m
    mi_at_m1 : float — MI at window = base_m + 1
    """
    fig = plt.figure(figsize=(8.27,11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Title
    ax.text(0.5, 0.97, "Information Metrics - Summary",
            ha="center", va="top", fontsize=14, fontweight="bold",
            transform=ax.transAxes)
    ax.axhline(0.954, xmin=0.05, xmax=0.95, color="#333333", linewidth=0.8)
    
    # Section 1: per-window statistics table
    ax.text(0.06, 0.93, "Per-window Statistics",
            ha='left', va='top', fontsize=10, fontweight='bold',
            transform=ax.transAxes)
    
    headers = ["Window k", "MI (bits)", "Predictability", "G-statistic", "p-value", "Reject EMH?"]
    col_x   = [0.06, 0.21, 0.37, 0.54, 0.68, 0.75]
    row_y   = 0.90
    
    for hdr, cx in zip(headers, col_x):
        ax.text(cx, row_y, hdr,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontweight="bold", color="#111111",
                fontfamily="monospace")
        
    ax.axhline(row_y - 0.015, xmin=0.05, xmax=0.95,
                   color="#888888", linewidth=0.6)
        
    for i, info in enumerate(information_results):
        k    = info["window"]
        mi   = info["mutual_information"]
        pred = info["predictability"]
        G    = info["G_statistic"]
        pval = info["G_p_value"]
        reject = ('✓ 99%' if info.get('reject_emh_99') else
                  '✓ 95%' if info.get('reject_emh_95') else
                  '✓ 90%' if info.get('reject_emh_90') else '✗')
        ry   = row_y -0.03 - i * 0.028
        
        row_vals = [
            f"k = {k}",
            f"{mi:.4f}",
            f"{pred:.1%}",
            f"{G:.2f}",
            f"{pval:.4f}",
            reject
            ]
        row_colours = [
            None, _mi_colour(mi), _pred_colour(pred), None, None,
            _emh_colour(info),
            ]
        
        for val, cx, bg in zip(row_vals, col_x, row_colours):
            if bg is not None:
                patch = FancyBboxPatch(
                    (cx-0.005, ry - 0.015), 0.17, 0.024,
                    boxstyle="round, pad=0.005",
                    linewidth=0.5, edgecolor="#aaaaaa",
                    facecolor=bg, transform=ax.transAxes,
                    clip_on=False, zorder=1,
                    )
                ax.add_patch(patch)
            ax.text(cx, ry, val,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontfamily="monospace", zorder=2)

    # Section 2: interpretation boxes with colour coding
    section_y = row_y - 0.03 - len(information_results) * 0.028 - 0.04
    ax.text(0.06, section_y, f"Interpretation (base window k = {base_m})",
            ha="left", va="top", fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    
    box_top =section_y - 0.03
    box_h = 0.14
    box_w = 0.20
    box_gap = 0.02
    starts = [0.06 + i * (box_w + box_gap) for i in range(4)]
    
    base_info = information_results[0]
    mi_base = base_info["mutual_information"]
    pred_base = base_info["predictability"]
    mi_delta = mi_at_m1 - mi_at_m
    
    boxes = [
        (f"Mutual Information\n(k = {base_m})",
         f"{mi_base:.4f} bits",
         _mi_verdict(mi_base),
         _mi_colour(mi_base)),
        (f"Predictability\n(k = {base_m})",
         f"{pred_base:.1%} of histories",
         _pred_verdict(pred_base),
         _pred_colour(pred_base)),
        ("Inaccessible Info\n(Δ MI, k→k+1)",
         f"{mi_delta:+.4f} bits",
         _delta_verdict(mi_delta),
         _delta_colour(mi_delta)),
        ("EMH Test\n(G-statistic)",
        f"p = {base_info['G_p_value']:.4f}",
        _emh_verdict(base_info),
        _emh_colour(base_info)),
    ]
    
    for (lbl, val, verdict, col), sx in zip(boxes, starts):
        _draw_coloured_box(ax, sx, box_top - box_h,
                           box_w, box_h, lbl,
                           val, verdict, col,
                           fontsize=8,
                           )

    # Section 3: narrative paragraph
    nar_y = box_top - box_h - 0.06
    ax.text(0.06, nar_y, "Narrative",
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    
    if mi_base >= _MI_STRONG:
        mi_sentence = (
            f"The history of length k={base_m} carries {mi_base:.4f} bits of mutual information "
            "with the next outcome, indicating a clear departure from randomness."
        )
    elif mi_base >= _MI_WEAK:
        mi_sentence = (
            f"The history of length k={base_m} carries {mi_base:.4f} bits of mutual information, "
            "a marginal signal that may reflect transient correlations rather than stable structure."
        )
    else:
        mi_sentence = (
            f"The history of length k={base_m} carries only {mi_base:.4f} bits of mutual information, "
            "consistent with near-random market dynamics at this memory scale."
        )
    
    #EMH
    if base_info.get('reject_emh_99'):
        emh_sentence = (
            f"The global G-test strongly rejects the EMH (G={base_info['G_statistic']:.2f}, "
            f"p={base_info['G_p_value']:.4f}), confirming statistically significant "
            "predictive structure at this window."
        )
    elif base_info.get('reject_emh_95'):
        emh_sentence = (
            f"The global G-test rejects the EMH at 95% confidence "
            f"(G={base_info['G_statistic']:.2f}, p={base_info['G_p_value']:.4f})."
        )
    elif base_info.get('reject_emh_90'):
        emh_sentence = (
            f"The global G-test rejects the EMH at 90% confidence only "
            f"(G={base_info['G_statistic']:.2f}, p={base_info['G_p_value']:.4f}), "
            "suggesting a marginal departure from efficiency."
        )
    else:
        emh_sentence = (
            f"The global G-test cannot reject the EMH "
            f"(G={base_info['G_statistic']:.2f}, p={base_info['G_p_value']:.4f}), "
            "consistent with an informationally efficient series at this window."
        )
        
    #Predictability, but no longer used.
    if pred_base >= _PRED_HIGH:
        pred_sentence = (
            f"{pred_base:.0%} of observed history patterns produce next-period probabilities "
            "that are statistically distinguishable from 0.5 (p < 0.05), "
            "suggesting systematic directional biases."
        )
    elif pred_base >= _PRED_MED:
        pred_sentence = (
            f"{pred_base:.0%} of observed history patterns show statistically significant "
            "deviations from p = 0.5, pointing to isolated but not pervasive predictability."
        )
    else:
        pred_sentence = (
            f"Only {pred_base:.0%} of observed history patterns show statistically significant "
            "deviations from p = 0.5, consistent with an effectively unpredictable series."
        )

    #Delta
    if mi_delta > _MI_WEAK:
        delta_sentence = (
            f"Moving to window k={base_m+1} adds {mi_delta:.4f} bits, revealing structure "
            f"that agents with memory k={base_m} cannot access. "
            "This represents a potential information advantage for longer-memory strategies."
        )
    else:
        delta_sentence = (
            f"Extending the window to k={base_m+1} adds only {mi_delta:.4f} bits, "
            f"suggesting that agents with memory k={base_m} are already exploiting "
            "most of the predictable structure present in the series."
        )

    narrative = f"{mi_sentence}\n\n{emh_sentence}\n\n{delta_sentence}"
    
    ax.text(0.06, nar_y - 0.03, narrative,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, wrap=True,
            bbox=dict(boxstyle="round,pad=0.04",
                      facecolor="#F4F4F8", edgecolor="#CCCCCC",
                      linewidth=0.8),
            multialignment="left")

    # Explanatory footer
    ax.text(0.5, 0.02,
            "MI = mutual information| G = global G-test statistic (chi-squared)| "
            "p-value vs. H₀: P(outcome|history)=0.5 for all histories",
            ha="center", va="bottom", fontsize=7, color="#666666",
            )

    fig.subplots_adjust()
    return fig


# Information figure - detailed bar chart
def create_information_figure(
        info_results: Dict[str, Any],
        window: int,
        ) -> plt.Figure:
    """
    Create information analysis figure for a specified window size
    Top panel contains conditional probabilities per history,
    bottom panel history occurences
    Stats box: MI, G-statistic, global p-value, EMH verdict.
    """
    transition_counts = info_results["transition_counts"]
    mi      = info_results["mutual_information"]
    pred    = info_results["predictability"]
    G  = info_results["G_statistic"]
    pval   = info_results["G_p_value"]
    
    indices = sorted(transition_counts.keys())
    probs = []
    counts_total = []
    bar_colours = []
    
    for idx in indices:
        c = transition_counts[idx]
        total = c[-1] + c[1]
        prob_plus = c[1] / total if total > 0 else 0.5
        probs.append(prob_plus)
        counts_total.append(total)
        if total > 10:
            expected = total / 2
            _, p = chisquare([c[-1], c[1]], [expected, expected])
            bar_colours.append("#4CAF50" if p < 0.05 else "#5B8DB8")
        else:
            bar_colours.append("#AAAAAA")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
    fig.suptitle(f"Information Analysis: window size k = {window}",
                  fontsize=13, fontweight="bold",y=1.01)
    
    #Top graph gives probability of +1 given history
    ax1.bar(indices, probs, color=bar_colours, alpha=0.85, edgecolor="#333333", linewidth=0.4)
    ax1.axhline(0.5, color='red', linestyle='--')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("History Index", fontsize=10)
    ax1.set_ylabel('P(A_{t+1}=1 | history)', fontsize=10)
    ax1.set_title(f'Predictive power of k={window} history', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=9)
    
    # Colour legend for bars
    bar_legend = [
        plt.Line2D([0], [0], color='red', linestyle='--', label='Random (p = 0.5)'),
        Patch(facecolor="#4CAF50", label="Predictable (p < 0.05)"),
        Patch(facecolor="#5B8DB8", label="Not predictable"),
        Patch(facecolor="#AAAAAA", label="Sparse (n < 10)"),
    ]
    ax1.legend(handles=bar_legend, fontsize=8, loc="upper left")
    
    # Stats box with global test
    stats_text = (
        f"MI = {mi:.4f} bits\n"
        f"Predictability = {pred:.1%}\n"
        f"G-statistic = {G:.2f}\n"
        f"Global p-value = {pval:.4f}\n"
        f"{'─'*24}\n"
        f"{_emh_verdict(info_results)}"
    )
    ax1.text(0.98, 0.97, stats_text,
             transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor=_emh_colour(info_results),
                       edgecolor="#555555",
                       linewidth=0.8,
                       alpha=0.9))
    
    #Bottom graph shows sample counts per history
    ax2.bar(indices, counts_total, alpha=0.7, color="#5B8DB8",
            edgecolor="#333333", linewidth=0.4)
    ax2.set_xlabel('History Index', fontsize=10)
    ax2.set_ylabel('Sample Count', fontsize=10)
    ax2.set_title('Observations per history pattern', fontsize=11)
    ax2.axhline(10, color='orange', linestyle=':', linewidth=1.2,
                label='Min threshold (n=10)')
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.35)
    
    fig.tight_layout()
    return fig


def create_rolling_mi_figure(
        attendance: np.ndarray,
        k: int,
        roll_window: int = 2000,
        step: int = 100,
        intervention_round: Optional[int] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
    """
    Rolling mutual inforamtion with chi-squared EMH threshold lines.
    
    Parameters
    ----------
    Attendance: np.ndarray - full attendance time series
    k: int - history window for MI computation
    roll_window: int - number of observations per rolling window
    step: int - step between windows
    intervention_round: int, optional - marks a vertical line when, for instance, an agent is introduced
    title: str, optional - override default title
    """
    mi_rolling = []
    round_indices = []
    
    for start in range(0, len(attendance) - roll_window, step):
        chunk = attendance[start:start + roll_window]
        stats = compute_history_statistics(chunk, k)
        mi_rolling.append(stats['mutual_information'])
        round_indices.append(start + roll_window)
    
    # Chi-squatred critical values
    df = 2**k - 1
    crit_90 = chi2_dist.ppf(0.90, df=df) / (2 * roll_window * np.log(2))
    crit_95 = chi2_dist.ppf(0.95, df=df) / (2 * roll_window * np.log(2))
    crit_99 = chi2_dist.ppf(0.99, df=df) / (2 * roll_window * np.log(2))
    
    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(round_indices, mi_rolling, linewidth=1.5, color='steelblue',
            label=f"MI (k={k})")
    ax.hlines(crit_90, round_indices[0], round_indices[-1],
              colors='orange', linestyles='--', label='90% threshold', alpha=0.8)
    ax.hlines(crit_95, round_indices[0], round_indices[-1],
              colors='red', linestyles='--', label='95% threshold', alpha=0.8)
    ax.hlines(crit_99, round_indices[0], round_indices[-1],
              colors='darkred', linestyles='--', label='99% threshold', alpha=0.7)
    
    if intervention_round is not None:
        ax.axvline(intervention_round, color='black', linestyle=':',
                   linewidth=1.5, label=f'Intervention (round{intervention_round})')
    
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Mutual Information (bits)', fontsize=11)
    ax.set_title(
        title or f'Rolling MI (k={k}, window={roll_window} rounds)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def create_combined_summary(information_results: list, 
                           base_m: int,
                           attendance: np.ndarray,
                           save_path: str,
                           success_data: dict = None,
                           price_history: np.ndarray = None):
    """
    Create a comprehensive multi-page PDF with all analysis.
    
    Parameters
    ----------
    information_results : list
        List of stats dicts from information_report
    base_m : int
        Agent memory size
    attendance : np.ndarray
        Full attendance time series
    save_path : str
        Output PDF path
    success_data : dict, optional
        Success rate data from boxplot experiment
        Should have keys: 'data', 'labels', 'cohort_data' (optional)
    price_history : np.ndarray, optional
        Price/wealth time series
    """
    with PdfPages(save_path) as pdf:

        # PAGE 1: Overview with population summary

        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        fig.suptitle(f'Information Analysis Overview (Agent memory m={base_m})', 
                    fontsize=14, fontweight='bold')
        
        windows = [info.get('window', info.get('memory')) for info in information_results]
        mi_values = [info['mutual_information'] for info in information_results]
        predict_values = [info['predictability'] for info in information_results]
        
        # Top left: MI vs window
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(windows, mi_values, 'o-', linewidth=2, markersize=8)
        ax1.axvline(base_m, color='red', linestyle='--', label=f'Agent memory (m={base_m})', linewidth=2)
        ax1.set_xlabel('Analysis window k', fontsize=10)
        ax1.set_ylabel('Mutual Information (bits)', fontsize=10)
        ax1.set_title('Information vs Window Size', fontweight='bold', fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Top middle: Predictability vs window
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(windows, predict_values, 's-', linewidth=2, markersize=8, color='green')
        ax2.axvline(base_m, color='red', linestyle='--', label=f'Agent memory (m={base_m})', linewidth=2)
        ax2.set_xlabel('Analysis window k', fontsize=10)
        ax2.set_ylabel('Predictable fraction', fontsize=10)
        ax2.set_title('Predictability vs Window Size', fontweight='bold', fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Top right: Summary table
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        table_data = []
        table_data.append(['k', 'MI', 'NMI', 'Predict', 'G-stat', 'p-val', 'Reject?', 'Histories'])
        for info in information_results:
            k = info.get('window', info.get('memory'))
            reject =    '✓ 99%' if info.get('reject_emh_99') else \
                        '✓ 95%' if info.get('reject_emh_95') else \
                        '✓ 90%' if info.get('reject_emh_90') else '✗'
            table_data.append([
                str(k),
                f"{info['mutual_information']:.4f}",
                f"{info['nmi']:.4f}",
                f"{info['predictability']:.1%}",
                f"{info['G_statistic']:.2f}",
                f"{info['G_p_value']:4f}",
                reject,
                f"{info['num_histories']}/{2**k}"
            ])
        
        table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        ax3.set_title('Summary Statistics', fontweight='bold', fontsize=11)
        
        # Middle left: Variance analysis
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        if 'variance_statistics' in information_results[0]:
            vs = information_results[0]['variance_statistics']
            var_text = f"""VARIANCE ANALYSIS
{'─' * 30}
σ²/N = {vs['variance_ratio']:.4f}
95% CI: [{vs['ci_lower']:.3f}, {vs['ci_upper']:.3f}]

{vs['classification']}
{vs['description']}

{'✓ Significant' if vs['is_significant'] else '✗ Not significant'}
p = {vs['p_value']:.4f}"""
            
            ax4.text(0.05, 0.95, var_text,
                    transform=ax4.transAxes,
                    fontsize=9, family='monospace',
                    verticalalignment='top')
            ax4.set_title('Variance Analysis', fontweight='bold', fontsize=11)
        
        # Middle center & right: Population structure (spans 2 columns)
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        if 'population_summary' in information_results[0]:
            pop_text = format_population_summary(information_results[0]['population_summary'])
            ax5.text(0.05, 0.95, pop_text,
                    transform=ax5.transAxes,
                    fontsize=9, family='monospace',
                    verticalalignment='top')
            ax5.set_title('Population Structure', fontweight='bold', fontsize=11)
        
        # Bottom row: Attendance history (spans all 3 columns)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.plot(attendance[-2000:], linewidth=0.5, alpha=0.7, color='blue')
        ax6.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax6.set_title('Recent Attendance History (last 2000 rounds)', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Round', fontsize=10)
        ax6.set_ylabel('Attendance', fontsize=10)
        ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 2: Price history (if provided)
        if price_history is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
            fig.suptitle('Price/Wealth Evolution', fontsize=14, fontweight='bold')
            
            # Full history
            ax1.plot(price_history, linewidth=0.5, alpha=0.7)
            ax1.set_title('Full Price History', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Round', fontsize=10)
            ax1.set_ylabel('Price', fontsize=10)
            ax1.grid(alpha=0.3)
            
            # Recent history (last 2000)
            ax2.plot(price_history[-2000:], linewidth=0.5, alpha=0.7, color='orange')
            ax2.set_title('Recent Price History (last 2000 rounds)', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Round', fontsize=10)
            ax2.set_ylabel('Price', fontsize=10)
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # PAGE 3: Success boxplot by cohort (if provided)
        if success_data is not None:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            
            if 'cohort_data' in success_data:
                # Multiple cohorts - create grouped boxplot
                cohort_names = list(success_data['cohort_data'].keys())
                n_cohorts = len(cohort_names)
                n_configs = len(success_data['labels'])
                
                # Prepare data for grouped boxplot
                all_data = []
                positions = []
                colors = plt.cm.Set3(range(n_cohorts))
                
                for config_idx in range(n_configs):
                    for cohort_idx, cohort_name in enumerate(cohort_names):
                        data = success_data['cohort_data'][cohort_name][config_idx]
                        all_data.append(data)
                        pos = config_idx * (n_cohorts + 1) + cohort_idx
                        positions.append(pos)
                
                bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True)
                
                # Color by cohort
                for i, patch in enumerate(bp['boxes']):
                    cohort_idx = i % n_cohorts
                    patch.set_facecolor(colors[cohort_idx])
                
                # Set x-ticks at center of each group
                tick_positions = [i * (n_cohorts + 1) + (n_cohorts - 1) / 2 for i in range(n_configs)]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(success_data['labels'], rotation=45, ha='right')
                
                # Legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=colors[i], label=name) 
                                 for i, name in enumerate(cohort_names)]
                ax.legend(handles=legend_elements, loc='best', title='Cohorts')
                
            else:
                # Simple boxplot
                bp = ax.boxplot(success_data['data'], labels=success_data['labels'], patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                ax.set_xticklabels(success_data['labels'], rotation=45, ha='right')
            
            ax.set_title('Success Rate Distribution by Configuration', fontsize=14, fontweight='bold')
            ax.set_ylabel('Success Rate', fontsize=12)
            ax.set_xlabel('Configuration', fontsize=12)
            ax.grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # SUBSEQUENT PAGES: Detailed analysis for each window
        for info in information_results:
            k = info.get('window', info.get('memory'))
            
            # Create the transition probability plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
            fig.suptitle(f'Detailed Analysis: Window k={k}', fontsize=14, fontweight='bold')
            
            indices = sorted(info['transition_counts'].keys())
            probs = []
            counts_total = []
            
            for idx in indices:
                c = info['transition_counts'][idx]
                total = c[-1] + c[1]
                prob_plus = c[1] / total if total > 0 else 0.5
                probs.append(prob_plus)
                counts_total.append(total)
            
            # Top: probability of +1 given history
            ax1.bar(indices, probs, alpha=0.7)
            ax1.axhline(0.5, color='red', linestyle='--', label='Random (p=0.5)', linewidth=2)
            ax1.set_xlabel('History index', fontsize=11)
            ax1.set_ylabel('P(A_{t+1}=1 | history)', fontsize=11)
            ax1.set_title(f'Predictive power of k={k} history', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)
            
            # Add statistics box
            stats_text = (
                f"Statistics (k={k}):\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Unique histories: {info['num_histories']} / {2**k}\n"
                f"Mutual Information: {info['mutual_information']:.4f} bits\n"
                f"G-statistic: {info['G_statistic']:.4f}\n"
                f"p-value: {info['G_p_value']:.4f}\n"
                f"Predictable fraction: {info['predictability']:.1%}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
            )
            
            if info.get('reject_emh_95'):
                stats_text += "✓ EMH rejected at 95%"
                box_color = 'lightgreen'
            elif info.get('reject_emh_90'):
                stats_text +="~ EMH rejected at 90% only"
                box_color = 'lightyellow'
            else:
                stats_text += "✗ Cannot reject EMH"
                box_color = 'lightcoral'
            
            ax1.text(0.98, 0.97, stats_text,
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
                    family='monospace')
            
            # Bottom: sample size for each history
            ax2.bar(indices, counts_total, alpha=0.7, color='gray')
            ax2.set_xlabel('History index', fontsize=11)
            ax2.set_ylabel('Observation count', fontsize=11)
            ax2.set_title('Sample size per history', fontsize=13, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"\nCombined analysis saved to: {save_path}")
    
def format_variance_report(var_stats: dict) -> str:
    """Format variance statistics for display."""
    lines = []
    lines.append(f"Variance Ratio: σ²/N = {var_stats['variance_ratio']:.4f}")
    lines.append(f"95% CI: [{var_stats['ci_lower']:.4f}, {var_stats['ci_upper']:.4f}]")
    lines.append(f"Z-score: {var_stats['z_score']:.2f}, p = {var_stats['p_value']:.4f}")
    
    if var_stats['is_significant']:
        lines.append(f"✓ Significant deviation: {var_stats['classification']}")
    else:
        lines.append(f"✗ Not significant: {var_stats['classification']}")
    
    return "\n".join(lines)

def information_report(attendance: np.ndarray,
                       window: int,
                       population_spec: dict = None,
                       save_dir: str = None) -> Dict:
    """
    Full information analysis report.
    
    Use after running a game:
        game = Game(...)
        results = game.run()
        info = information_report(results['Attendance'], memory=game.players[0].memory)
    """
    k = window
    stats = compute_history_statistics(attendance, k)
    
    if population_spec:
        stats['population_summary'] = summarize_population(population_spec)
        
        #Compute normalised variance for context
        N = population_spec['total']
        var_stats = compute_variance_statistics(attendance, N)
        stats['variance_statistics'] = var_stats
        stats["N"] = N
    
    print(f"\n{'='*60}")
    print(f"Information Analysis (window={k})")
    print(f"{'='*60}")
    print(f"Number of unique histories observed: {stats['num_histories']} / {2**k}")
    print(f"Mutual Information I(H; A_{{t+1}}): {stats['mutual_information']:.4f} bits")
    print(f"NMI:                               {stats['nmi']:.4f}")
    print(f"G-statistic:                       {stats['G_statistic']:.4f} (df={stats['G_df']})")
    print(f"EMH test p-value:                  {stats['G_p_value']:.4f}")
    print(f"Reject EMH at 90%: {'✓' if stats['reject_emh_90'] else '✗'}  "
          f"95%: {'✓' if stats['reject_emh_95'] else '✗'}  "
          f"99%: {'✓' if stats['reject_emh_99'] else '✗'}")
    print(f"Predictable histories (descriptive): {stats['predictability']:.2%}")
    
    if 'variance_stats' in stats:
        print(f"\n{format_variance_report(stats['variance_statistics'])}")

    if stats.get('reject_emh_99'):
        print("\n✓ EMH rejected at 99% confidence")
    elif stats.get('reject_emh_95'):
        print("\n✓ EMH rejected at 95% confidence")
    elif stats.get('reject_emh_90'):
        print("\n~ EMH rejected at 90% confidence only")
    else:
        print("\n✗ Cannot reject EMH at 90% confidence")
    
    if save_dir:

        os.makedirs(save_dir, exist_ok=True)
        
        # Create summary page
        summary_path = os.path.join(save_dir, f"summary_window_{k}.pdf")
        create_summary_page(stats, summary_path)
        print(f"\nSummary page saved to: {summary_path}")
        
        # Create detailed plot
        plot_path = os.path.join(save_dir, f"history_predictability_k{k}.pdf")
        plot_transition_probabilities(stats['transition_counts'], k, 
                                     save_path=plot_path)
        print(f"Detail plot saved to: {plot_path}")
        
    return stats
