#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/single_strat_worker.py

Worker module for the single-strategy proportion experiment.

Supports sweeping across both proportions of single-strategy agents
and memory lengths (m values), running all games in parallel via
ProcessPoolExecutor.

All worker arguments are plain Python primitives so they pickle cleanly
across process boundaries.

Public API
----------
run_proportion_sweep(...)
    Sweep proportions for a single m value.

run_full_sweep(...)
    Sweep both proportions and m values. Returns nested dict:
    results[m][proportion][label] -> list of mean success rates per run.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np


# ── Population spec builder ───────────────────────────────────────────────────

def _make_population_spec(
    n_total:        int,
    proportion:     float,
    m:              int,
    n_strats:       int,
    payoff:         str,
    position_limit: int = 0,
) -> dict:
    """
    Build a plain population_spec dict for a given proportion and memory.

    Cohort 0 = adaptive agents  (strategies = n_strats)
    Cohort 1 = single-strategy  (strategies = 1)

    Both cohorts use agent_type='strategic'. Single-strategy agents cannot
    switch because they only have one strategy to choose from.
    """
    n_single   = max(1, round(proportion * n_total)) if proportion > 0.0 else 0
    n_adaptive = n_total

    cohorts = []
    if n_adaptive > 0:
        cohorts.append({
            "count":          n_adaptive,
            "memory":         m,
            "strategies":     n_strats,
            "payoff":         payoff,
            "position_limit": position_limit,
            "agent_type":     "strategic",
            "score_lambda":   0.0,
        })
    if n_single > 0:
        cohorts.append({
            "count":          n_single,
            "memory":         m,
            "strategies":     1,
            "payoff":         payoff,
            "position_limit": position_limit,
            "agent_type":     "strategic",
            "score_lambda":   0.0,
        })

    return {"total": n_total + n_single, "cohorts": cohorts}


def _cohort_labels(population_spec: dict) -> Dict[int, str]:
    """cohort_id -> 'adaptive' or 'single_strat'."""
    return {
        idx: ("single_strat" if c["strategies"] == 1 else "adaptive")
        for idx, c in enumerate(population_spec["cohorts"])
    }


# ── Module-level worker ───────────────────────────────────────────────────────

def _run_single_strat_game(args: tuple) -> dict:
    """
    Run one game using the main Game engine.

    Must be module-level for pickling across ProcessPoolExecutor workers.

    Args tuple
    ----------
    m               int     memory length
    proportion      float   fraction of single-strategy agents
    run_idx         int     run index (offsets seed)
    n_total         int
    n_strats        int     strategies for adaptive cohort
    payoff          str
    rounds          int
    seed_base       int
    position_limit  int
    """
    (m, proportion, run_idx, n_total, n_strats,
     payoff, rounds, seed_base, position_limit) = args

    # Import here so the worker process gets its own fresh import
    from core.game import Game
    from core.game_config import GameConfig

    seed            = seed_base + run_idx
    population_spec = _make_population_spec(
        n_total, proportion, m, n_strats, payoff, position_limit
    )
    label_map = _cohort_labels(population_spec)

    cfg_game = GameConfig(
        rounds=rounds,
        lambda_=1.0 / (n_total * 50),
        mm=None,
        price=100,
        seed=seed,
        record_agent_series=True,   # needed for correct final_wins
    )

    game    = Game(population_spec=population_spec, cfg=cfg_game)
    results = game.run()

    cohort_ids = results["cohort_ids"]
    final_wins = results["final_wins"]
    success    = final_wins / max(1, rounds)

    success_by_cohort: Dict[int, float] = {
        int(cid): float(np.mean(success[cohort_ids == cid]))
        for cid in np.unique(cohort_ids)
    }

    final_points = results["final_points"]
    points_per_round_by_cohort: Dict[int, float] = {
        int(cid): float(np.mean(final_points[cohort_ids == cid])) / max(1, rounds)
        for cid in np.unique(cohort_ids)
    }
    return {
        "m":           m,
        "proportion":  proportion,
        "run_idx":     run_idx,
        "label_map":   label_map,
        "success":     success_by_cohort,
        "points":      points_per_round_by_cohort, 
        "attendance":  results["Attendance"],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_full_sweep(
    m_values:       List[int],
    proportions:    List[float],
    n_total:        int           = 301,
    n_strats:       int           = 2,
    payoff:         str           = "ScaledMGPayoff",
    rounds:         int           = 10_000,
    n_runs:         int           = 20,
    seed_base:      int           = 42,
    max_workers:    Optional[int] = None,
    position_limit: int           = 0,
) -> dict:
    """
    Sweep both m values and single-strategy proportions in parallel.

    Parameters
    ----------
    m_values        : list of memory lengths to sweep
    proportions     : list of single-strategy proportions in [0, 1]
    n_total         : total agents per game
    n_strats        : strategies for the adaptive cohort
    payoff          : payoff key string
    rounds          : rounds per game
    n_runs          : independent games per (m, proportion) combination
    seed_base       : base random seed; actual seed = seed_base + run_idx
    max_workers     : parallel workers (None = all available cores)
    position_limit  : position limit (0 = unconstrained)

    Returns
    -------
    dict with keys:

    "success"
        dict: m -> proportion -> label -> list[float]
        Mean success rate per game, one entry per run.

    "attendance"
        dict: m -> proportion -> np.ndarray
        Attendance series from run 0 of each (m, proportion), for MI analysis.

    "label_map"
        dict: m -> proportion -> dict[int, str]
    """
    tasks = [
        (m, proportion, run_idx, n_total, n_strats,
         payoff, rounds, seed_base, position_limit)
        for m in m_values
        for proportion in proportions
        for run_idx in range(n_runs)
    ]

    # Nested storage: m -> proportion -> ...
    success_store: Dict = {
        m: {p: {} for p in proportions} for m in m_values
    }
    points_store: Dict = {
        m: {p: {} for p in proportions} for m in m_values
    }
    attendance_store: Dict = {m: {} for m in m_values}
    label_map_store:  Dict = {m: {} for m in m_values}

    total = len(tasks)
    done  = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_strat_game, t): t for t in tasks}
        for future in as_completed(futures):
            res  = future.result()
            m    = res["m"]
            prop = res["proportion"]
            ridx = res["run_idx"]

            if prop not in label_map_store[m]:
                label_map_store[m][prop] = res["label_map"]

            if ridx == 0:
                attendance_store[m][prop] = res["attendance"]

            for cid, mean_sr in res["success"].items():
                label = res["label_map"].get(cid, f"cohort_{cid}")
                success_store[m][prop].setdefault(label, [])
                success_store[m][prop][label].append(mean_sr)
            
            for cid, mean_pts in res["points"].items():
                label = res["label_map"].get(cid, f"cohort_{cid}")
                points_store[m][prop].setdefault(label, [])
                points_store[m][prop][label].append(mean_pts)

            done += 1
            print(
                f"  [{done}/{total}]  m={m}  proportion={prop:.0%}"
                f"  run={ridx}",
                end="\r",
            )

    print(f"\nSweep complete. {total} games run.")

    return {
        "success":    success_store,
        "points":     points_store,
        "attendance": attendance_store,
        "label_map":  label_map_store,
    }


def run_proportion_sweep(
    proportions:    List[float],
    m:              int           = 5,
    n_total:        int           = 301,
    n_strats:       int           = 2,
    payoff:         str           = "ScaledMGPayoff",
    rounds:         int           = 10_000,
    n_runs:         int           = 20,
    seed_base:      int           = 42,
    max_workers:    Optional[int] = None,
    position_limit: int           = 0,
) -> dict:
    """
    Convenience wrapper: sweep proportions for a single m value.
    Returns the same structure as run_full_sweep but for one m.
    """
    full = run_full_sweep(
        m_values=[m],
        proportions=proportions,
        n_total=n_total,
        n_strats=n_strats,
        payoff=payoff,
        rounds=rounds,
        n_runs=n_runs,
        seed_base=seed_base,
        max_workers=max_workers,
        position_limit=position_limit,
    )
    return {
        "success":    full["success"][m],
        "attendance": full["attendance"][m],
        "label_map":  full["label_map"][m],
    }
