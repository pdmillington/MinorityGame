#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 20:38:22 2025

@author: petermillington
"""

import numpy as np

def cohort_labels_from_meta(meta):

    labels = {}
    for cid, c in enumerate(meta["cohorts"]):
        if c.agent_type == "noise":
            labels[cid] = "Noise Player"
        else:
            payoff_name = c.payoff.replace("Payoff", "")
            labels[cid] = f"m={c.memory}\nS={c.strategies}\n{payoff_name}"
    return labels

def cohort_map_from_spec(population_spec: dict) -> tuple:
    """
    Returns
    -------
    cohort_ids  : ndarray (N,) of int   — matches group_vector_by_cohort
    cohort_labels: dict {int: str}      — matches cohort_labels_from_meta pattern
    """
    ids    = []
    labels = {}
    for cid, c in enumerate(population_spec["cohorts"]):
        label = f"{c['payoff']}_m{c['memory']}_s{c['strategies']}"
        labels[cid] = label
        ids.extend([cid] * c["count"])
    return np.array(ids), labels

def group_vector_by_cohort(x, cohort_ids):
    x = np.asarray(x)
    cohort_ids = np.asarray(cohort_ids)
    return {
        int(cid): x[cohort_ids == cid]
        for cid in np.unique(cohort_ids)
    }

def group_timeseries_mean_by_cohort(X, cohort_ids):
    X = np.asarray(X)
    cohort_ids = np.asarray(cohort_ids)
    out = {}
    for cid in np.unique(cohort_ids):
        mask = cohort_ids == cid
        out[int(cid)] = X[:, mask].mean(axis=1)
    return out

def summarize_population(population_spec: dict) -> dict:
    """
    Extract key population metrics for reporting.
    
    Returns summary dict with:
        - N_total: total population
        - cohorts: list of cohort summaries
        - alpha_effective: weighted average alpha
        - memory_range: (min_m, max_m)
        - is_homogeneous: whether all agents have same m
    """
    N_total = population_spec['total']
    cohorts = population_spec['cohorts']

    cohort_summaries = []
    m_values = []
    m_weights = []

    for cohort in cohorts:
        count = cohort['count']
        m = cohort.get('memory',1)
        payoff = cohort.get('payoff', 'Noise')
        strategies = cohort.get('strategies', 2)

        cohort_summaries.append({
            'count': count,
            'memory': m,
            'payoff': payoff,
            'strategies': strategies,
            'fraction': count / N_total
        })

        m_values.append(m)
        m_weights.append(count)

    # Weighted average memory for alpha calculation
    m_eff = sum(m * w for m, w in zip(m_values, m_weights)) / sum(m_weights)
    alpha_eff = (2 ** m_eff) / N_total

    return {
        'N_total': N_total,
        'cohorts': cohort_summaries,
        'alpha_effective': alpha_eff,
        'alpha_weighted_geometric': (sum(w * 2**m for m, w in zip(m_values, m_weights))
                                     / sum(m_weights)) / N_total,
        'memory_range': (min(m_values), max(m_values)),
        'is_homogeneous': len(set(m_values)) == 1,
        'm_effective': m_eff,
    }

def format_population_summary(pop_summary: dict, compact: bool = None) -> str:
    """
    Create human-readable summary of population structure.
    
    Parameters
    ----------
    pop_summary : dict
        Output from summarize_population()
    compact : bool, optional
        If True, use compact format. Auto-detected if None.
    """
    n_cohorts = len(pop_summary['cohorts'])

    # Auto-detect compact mode for many cohorts
    if compact is None:
        compact = n_cohorts > 6

    lines = []
    lines.append(f"Total Population: N = {pop_summary['N_total']}")
    if not pop_summary['is_homogeneous']:
        lines.append(f"Memory Range: m ∈ [{pop_summary['memory_range'][0]}, {pop_summary['memory_range'][1]}]")

    if pop_summary['is_homogeneous']:
        m = pop_summary['memory_range'][0]
        lines.append(f"Memory homogeneous: all agents have m = {m}")
        lines.append(f"α = 2^m/N = {2**m}/{pop_summary['N_total']} = {pop_summary['alpha_effective']:.4f}")
    else:
        lines.append(f"Effective Memory: m_eff = {pop_summary['m_effective']:.2f}")
        lines.append(f"α_eff = 2^m_eff/N = {pop_summary['alpha_effective']:.4f}")
        lines.append(f"α_weighted = {pop_summary['alpha_weighted_geometric']:.4f}")
        lines.append("")
    
    if compact:
        # Compact format: one line per cohort
        lines.append("Cohorts: [n (%), m, s, payoff]")
        for i, c in enumerate(pop_summary['cohorts'], 1):
            lines.append(f"  {i}. {c['count']} ({c['fraction']:.0%}), "
                        f"m={c['memory']}, s={c['strategies']}, {c['payoff']}")
    else:
        # Verbose format: easier to read for few cohorts
        lines.append("Cohorts:")
        for i, cohort in enumerate(pop_summary['cohorts'], 1):
            lines.append(f"  {i}. n={cohort['count']} ({cohort['fraction']:.1%}): "
                        f"m={cohort['memory']}, {cohort['strategies']} strategies, "
                        f"{cohort['payoff']}")

    return "\n".join(lines)
