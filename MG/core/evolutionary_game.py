#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:05:50 2026

@author: petermillington

Extends game with the posibility of strategy evolution.

Each generation runs a game of rounds_per_generation.  The agents are ranked
by wealth across the whole population.  Note that the wealth ranking is 'monetary'
wealth from buying and selling shares and not success rates as measured by the 
strategy objectives.  A fraction of candidates ranked in the bottom poverty 
percentile are given renewed strategies, although their payoff type is preserved.
Strategy scores restart from 0.  Wealth is maintained or not by reset_wealth to 
model bankruptcy or simple strategy change.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
 
from core.game import Game
from core.base_agent import StrategicAgent
from data.stats_recorder import StatsRecorder

@dataclass
class GenerationRecord:
    """
    Stores replacement diagnostics for a single generation.
    
    Attributes
    ----------
    generation: int
        Generation index
    wealth_threshold: float
        Actual wealth value at the poverty_percentile cutoff
    eligible_indices: list
        Indices of all agents at or below the wealth threshold
    replaced_indices: list
        Indices of agents that have actually been replaced
    cohort_replacement_counts: dict
        cohort_id -> number of agents replaced in that cohort
    cohort_replacement_rates: dict
        cohort_id -> fraction of cohort replaced
    cohort_mean_wealth_replaced: dict
        cohort_id -> mean wealth of replaced agents (nan if none replaced)
    cohort_mean_wealth_surviving: dict
        cohort_id -> mean wealth of surviving agents
    """
    generation: int
    wealth_threshold: float
    eligible_indices: list
    replaced_indices: list
    cohort_replacement_counts: dict
    cohort_replacement_rates: dict
    cohort_mean_wealth_replaced: dict
    cohort_mean_wealth_surviving: dict
    
class EvolutionaryGame(Game):
    """
    Extends Game with inter-generational strategy evolution
    
    Parameters
    -------
    population_spec : dict
        Population specification — supports mixed payoff cohorts
    cfg : GameConfig
        Game configuration. cfg.rounds should equal rounds_per_generation.
    rounds_per_generation : int
        Number of rounds in each generation
    num_generations : int
        Total number of generations to run
    poverty_percentile : float
        Fraction of population eligible for replacement, e.g. 0.20 = bottom 20%.
        Ranking is population-wide across all cohorts.
    replacement_rate : float
        Fraction of eligible agents that are actually replaced, e.g. 0.50.
        Selection within eligible is random.
    reset_wealth : bool
        If True, replaced agents have wealth, cash and position reset to zero
        (bankruptcy). If False, only strategies and scores are replaced.
    carry_price : bool
        If True, price and history persist across generations.
        If False, price resets to cfg.price and history is redrawn each generation.
    """
    def __init__(self,
                 population_spec: dict,
                 cfg,
                 rounds_per_generation: int,
                 num_generations: int,
                 poverty_percentile: float,
                 replacement_rate: float,
                 reset_wealth: bool,
                 ):
        
        super().__init__(population_spec, cfg)
        
        self.rounds_per_generation = rounds_per_generation
        self.num_generations = num_generations
        self.poverty_percentile = poverty_percentile
        self.replacement_rate = replacement_rate
        self.reset_wealth = reset_wealth
        
        # lists to store generation records across generations
        self.all_generation_results: list = []
        self.generation_records: list = []
        
    def _reset_stats(self) -> None:
        """
        Reinitialise StatsRecorder for a fresh generation
        Must be called before each generation is run
        """
        k_max = max(p.num_strategies for p in self.players
                    if hasattr(p, 'num_strategies'))
        self.stats = StatsRecorder(
            N=self.n,
            rounds= self.rounds_per_generation,
            k_max=k_max,
            record_agent_series=self.cfg.record_agent_series,
            )

    def evolve_population(self, generation: int) -> GenerationRecord:
        """
        Evolve the population after a generation completes.
        1. Rank agents by wealth across whole population
        2. Identify bottom percentile
        3. Randomly select a fraction for strategy replacement
        4. Redraw strategies for selected agents, reset scores
        5. Reset wealth if bankruptcy mode chosen
        6. Build and return GenerationRecord

        Parameters
        ----------
        generation : int
            Current generation index.

        Returns
        -------
        GenerationRecord
            Full diagnostics for this generation's evolution step.
        """
        wealths = np.array([p.wealth for p in self.players])
        n_eligible = max(1, int(self.n * self.poverty_percentile))
        sorted_indices = np.argsort(wealths)
        eligible = sorted_indices[:n_eligible].tolist()
        threshold = float(wealths[sorted_indices[n_eligible - 1]])
        
        eligible = [i for i, w in enumerate(wealths) if w <= threshold]
        
        n_replace = max(1, int(len(eligible) * self.replacement_rate))
        replaced = list(
            self.rng.choice(eligible, size=n_replace, replace=False))
        replaced_set = set(replaced)
        
        # Cohort level diagnostics computed before replacement
        cohort_ids = sorted(set(p.cohort_id for p in self.players
                                if p.cohort_id is not None))
        cohort_replacement_counts = {}
        cohort_replacement_rates = {}
        cohort_mean_wealth_replaced = {}
        cohort_mean_wealth_surviving = {}
        
        for c_id in cohort_ids:
            cohort_indices = [i for i,p in enumerate(self.players)
                              if p.cohort_id == c_id]
            replaced_in_cohort = [i for i in cohort_indices
                                  if i in replaced_set]
            surviving_in_cohort = [i for i in cohort_indices
                                   if i not in replaced_set]
            
            cohort_replacement_counts[c_id] = len(replaced_in_cohort)
            cohort_replacement_rates[c_id] = (
                len(replaced_in_cohort) / len(cohort_indices)
                if cohort_indices else 0.0)
            cohort_mean_wealth_replaced[c_id] = (
                float(np.mean([wealths[i] for i in replaced_in_cohort]))
                if replaced_in_cohort else float('nan'))
            cohort_mean_wealth_surviving[c_id] = (
                float(np.mean([wealths[i] for i in surviving_in_cohort]))
                if surviving_in_cohort else float('nan'))

        # apply replacements, reselect strategies, reset scores
        for i in replaced:
            p = self.players[i]
            if isinstance(p, StrategicAgent):
                p.strategies = self.rng.choice(
                    [-1, +1],
                    size=(p.num_strategies, 2 ** p.memory))
                p.scores = np.zeros(p.num_strategies)
                p.strategy = None
                p.strategy_switches = 0
                p._pending = None
                p._prev = None
                
            if self.reset_wealth:
                p.position = 0
                p.cash = 0.0
                p.wealth = 0.0
                p.points = 0.0
                p.wins = 0
        
        return GenerationRecord(
            generation=generation,
            wealth_threshold=float(threshold),
            eligible_indices=eligible,
            replaced_indices=replaced,
            cohort_replacement_counts=cohort_replacement_counts,
            cohort_replacement_rates=cohort_replacement_rates,
            cohort_mean_wealth_replaced=cohort_mean_wealth_replaced,
            cohort_mean_wealth_surviving=cohort_mean_wealth_surviving,
            )
    
    
    def run_evolutionary(self) -> dict:
        """
        Run num_generations generations with evolutionary changes between them.
        Price and history are continuous, self.stats is reinitialized between gens.

        Returns
        -------
        dict with keys:
            'generation_results'  : list of per-generation results dicts
            'generation_records'  : list of GenerationRecord objects
        """
        for gen in range(self.num_generations):
            
            self._reset_stats()
            
            # Run one generation
            results = self.run()
            results["generation"] = gen
            self.all_generation_results.append(results)
            
            # Evolve
            if gen < self.num_generations - 1:
                record = self.evolve_population(gen)
                self.generation_records.append(record)
            
        return {
            "generation_results": self.all_generation_results,
            "generation_records": self.generation_records,
            }

    
    
    
    