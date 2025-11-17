#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:07:44 2025

@author: petermillington
"""
import numpy as np
from numpy.random import SeedSequence
from payoffs.mg import PAYOFF_REGISTRY
from core.PopulationFactory import PopulationFactory

def build_players(population_spec,
                  PlayerClass,
                  rng=None,
                  shuffle=True,
                  per_player_seeds=False,
                  master_seed=None):
    """
    Returns: list[PlayerClass], cohort_id_per_player (np.array)
    """
    meta = PopulationFactory(population_spec, rng=rng).build()
    cohorts = meta["cohorts"]

    players = []
    cohort_id_vec = []

    # Reproducible child seeds
    seed_iter = None
    if per_player_seeds:
        ss = SeedSequence(master_seed if master_seed is not None else 0)
        child_seeds = iter(ss.spawn(meta["N"]))

    for c_id, c in enumerate(cohorts):
        payoff_obj = PAYOFF_REGISTRY[c.payoff]
        
        for _ in range(c.count):
            kwargs = dict(memory=c.memory, num_strategies=c.strategies,
                          payoff=payoff_obj, position_limit=c.position_limit)
            if per_player_seeds:
                kwargs["rng"] = np.random.default_rng(child_seeds.__next__())
            p = PlayerClass(**kwargs)
            # Optional, but handy for stats later:
            p.cohort_id = c_id
            players.append(p)
            cohort_id_vec.append(c_id)

    if shuffle:
        # avoid cohort blocks in order (optional)

        idx = np.arange(len(players))
        np.random.default_rng(rng).shuffle(idx)
        players = [players[i] for i in idx]
        cohort_id_vec = np.array(cohort_id_vec, dtype=int)[idx]
    else:
        cohort_id_vec = np.array(cohort_id_vec, dtype=int)

    return players, meta, cohort_id_vec
