#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 17:24:39 2025

@author: petermillington
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Literal

def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None:
        return []
    return [x]

@dataclass
class CohortConfig:
    count: int
    memory: int
    strategies: int
    payoff: str
    position_limit: int = 0

@dataclass
class PopulationConfig:
    """
    Dataclass for plotting success by cohort for heterogeneous memory or strategies
    allowing JSON file input of experiment configuration.
    """
    payoffs: Iterable[str]|str
    num_players_per_cohort: int
    m_values: Iterable[int]|int
    s_values: Iterable[int]|int
    position_limit: int = 0
    rounds: int = 1_000
    lambda_: float = None
    market_maker: bool | None = None
    price: float = 100
    record_agent_series: bool = True
    save_dir: str = "plots/success"
    mode: str = "cartesian"   #or "zip"
    cohorts: List[CohortConfig] = field(default_factory=list)
    seed: str = 1234

@dataclass
class PopulationFamilyConfig:
    base_cohorts: List[dict]
    #A single parameter or parameter group is varied across different Populations
    vary: Literal[
        "memory_shift",
        "strategies_shift",
        "total_size",
        "payoff_weights",
        "scale_cohort",
        ]
    #Values of the parameter to vary
    values: List[float]

    #Extra information to know which cohorts are affected
    target_payoff: str | None = None
    target_tag: str | None = None
    rounds: int = 1_000
    lambda_: float = None
    market_maker: bool | None = None
    price: float = 100
    record_agent_series: bool = True
    save_dir: str = "plots/success"
    mode: str = "cartesian"   #or "zip"
    seed: str = 1234

def vary_memory(base_cohorts, delta_m):
    new_cohorts = []
    for c in base_cohorts:
        new_cohorts.append({
            **c,
            "memory": + delta_m
            })
    total = sum(c["count"] for c in new_cohorts)
    return {"total": total, "cohorts": new_cohorts}

def vary_strategies(base_cohorts, delta_s):
    new_cohorts = []
    for c in base_cohorts:
        new_cohorts.append({
            **c,
            "strategies": + delta_s
            })
    total = sum(c["count"] for c in new_cohorts)
    return {"total": total, "cohorts": new_cohorts}

def vary_total_size(base_cohorts, N_target):
    base_total = sum(c["count"] for c in base_cohorts)
    factor = N_target / base_total

    new_cohorts = []
    for c in base_cohorts:
        new_cohorts.append({
            **c,
            "count": int(round(c["count"] * factor))
            })
    total = sum(c["count"] for c in new_cohorts)
    return {"total": total, "cohorts": new_cohorts}

# rescale within each group proportionally
def rescale_group(group, target_total):
    base = sum(c["count"] for c in group)
    if base == 0:
        return []
    factor = target_total / base
    out = []
    for c in group:
        out.append({**c, "count": int(round(c["count"] * factor))})
    return out

def vary_payoff_weights(base_cohorts, target_payoff, target_share):
    A = [c for c in base_cohorts if c["payoff"] == target_payoff]
    B = [c for c in base_cohorts if c["payoff"] != target_payoff]

    base_A = sum(c["count"] for c in A)
    base_B = sum(c["count"] for c in B)
    base_total = base_A + base_B

    A_target = int(round(base_total * target_share))
    B_target = base_total - A_target

    A_new = rescale_group(A, A_target)
    B_new = rescale_group(B, B_target)

    cohorts = A_new + B_new
    total = sum([c["count"] for c in cohorts])

    return {"total": total, "cohorts": cohorts}

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
def build_population_spec(pop_cfg: PopulationConfig) -> dict:
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

    return print("Not possible to construct a population")

def build_population_variant(family_cfg: PopulationFamilyConfig, value)-> dict:
    base =family_cfg.base_cohorts

    if family_cfg.vary == "memory_shift":
        return vary_memory(base_cohorts=base, delta_m=int(value))

    if family_cfg.vary == "strategies_shift":
        return vary_strategies(base_cohorts=base, delta_s=int(value))

    if family_cfg.vary == "total_size":
        return vary_total_size(base_cohorts=base, N_target=int(value))

    if family_cfg.vary == "payoff_share":
        if family_cfg.target_payoff is None:
            raise ValueError("Target payoff needs to be set.")
        target_payoff = family_cfg.target_payoff
        return vary_payoff_weights(base_cohorts=base,
                                 target_payoff=target_payoff,
                                 target_share =float(value))

    return print("Not possible to construct a Family of Populations")
    