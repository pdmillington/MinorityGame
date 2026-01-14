#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:07:44 2025

@author: petermillington
"""
import numpy as np
from numpy.random import SeedSequence
from payoffs.mg import PAYOFF_REGISTRY
from core.population_factory import PopulationFactory

def build_players(population_spec,
                  agent_class_map,
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
    child_seeds = None
    ss = SeedSequence(master_seed if master_seed is not None else 0)
    child_seeds = iter(ss.spawn(meta["N"]))
    
    if agent_class_map is None:
        raise ValueError("no agent class map defined")

    for c_id, c in enumerate(cohorts):
        agent_type = getattr(c, "agent_type", "strategic")
        Cls = agent_class_map.get(agent_type)
        if Cls is None: 
            raise ValueError(f"Agent class not defined for {agent_type}")
        
        plim = None if (c.position_limit is None or c.position_limit == 0) else int(c.position_limit)
        
        if agent_type == "strategic":
            payoff_obj = PAYOFF_REGISTRY[c.payoff]
            base_kwargs = dict(
                memory = c.memory,
                num_strategies = c.strategies,
                payoff = payoff_obj,
                position_limit = plim,
            )
        elif agent_type == "noise":
            base_kwargs = dict(
                position_limit = plim,
                allow_no_action = getattr(c, "allow_no_action", False)
            )
        else:
            raise ValueError(f"Unknown agent_type={agent_type}")
        
        for _ in range(c.count):
            kwargs = dict(base_kwargs)
            if per_player_seeds:
                kwargs["rng"] = np.random.default_rng(next(child_seeds))
            else:
                kwargs["rng"] = rng

            p = Cls(**kwargs)
            # Optional, but handy for stats later:
            p.cohort_id = c_id
            players.append(p)
            cohort_id_vec.append(c_id)

    cohort_id_vec = np.array(cohort_id_vec, dtype=int)
    if shuffle:
        # avoid cohort blocks in order (optional)
        idx = np.arange(len(players))
        rng.shuffle(idx)
        players = [players[i] for i in idx]
        cohort_id_vec = cohort_id_vec[idx]
   
    return players, meta, cohort_id_vec
