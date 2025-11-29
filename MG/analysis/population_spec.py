#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 17:24:39 2025

@author: petermillington
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import List, Optional

def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    elif x is None:
        return []
    else:
        return [x]

def make_hetero_population_spec(
    m_values,
    num_players_per_cohort: int,
    payoffs,
    s_values,
    position_limit: int = 0,
):
    """
    Cartesian builder:
    - m_values, s_values, payoffs can be scalar or list → all combinations become cohorts
    - num_players is total agents, divided across cohorts (remainder goes into first).
    """
    m_values = ensure_list(m_values)
    s_values = ensure_list(s_values)
    payoffs  = ensure_list(payoffs)

    combos = list(itertools.product(m_values, s_values, payoffs))
    num_cohorts = len(combos)
    if num_cohorts == 0:
        raise ValueError("No cohorts: m_values, s_values or payoffs is empty.")

    base_count = num_players_per_cohort // num_cohorts
    remainder  = num_players_per_cohort - base_count * num_cohorts

    cohorts = []
    for idx, (m, s, p) in enumerate(combos):
        count = base_count + (remainder if idx == 0 else 0)
        cohorts.append({
            "count": count,
            "memory": m,
            "payoff": p,
            "strategies": s,
            "position_limit": position_limit,
        })

    total = sum(c["count"] for c in cohorts)
    return {"total": total, "cohorts": cohorts}

# Top level builder of population
def build_population_spec(pop_cfg: SuccessBoxplotCohortConfig) -> dict:
    if pop_cfg.cohorts:
        cohorts = []
        for c in pop_cfg.cohorts:
            cohorts.append({
                "count":c.count,
                "memory": c.memory,
                "payoff": c.payoff,
                "strategies": c.strategies,
                "position_limit": c.position_limit,
                })
        total = sum(c["count"] for c in cohorts)
        return {"total": total, "cohorts": cohorts}
    if pop_cfg.mode == "cartesian":
        return make_hetero_population_spec(
            m_values=pop_cfg.m_values,
            num_players_per_cohort=pop_cfg.num_players_per_cohort,
            payoffs=pop_cfg.payoffs,
            s_values=pop_cfg.s_values,
            position_limit=pop_cfg.position_limit)