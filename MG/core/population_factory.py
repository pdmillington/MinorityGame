#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:22:58 2025

@author: petermillington
"""

from dataclasses import dataclass
import itertools as it
import numpy as np

@dataclass(frozen=True)
class Cohort:
    count: int
    memory: int
    payoff: str
    strategies: int
    position_limit: int = None

class PopulationFactory:
    def __init__(self, spec, rng=None):
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng

    def build(self):
        cohorts = self._cohorts_from_spec(self.spec)
        total = sum(c.count for c in cohorts)

        payoff_map = {"BinaryMG":0,
                      "ScaledMG":1,
                      "SmallMinority":2,
                      "DollarGame": 3}
        memory      = np.empty(total, dtype=np.int32)
        strategies  = np.empty(total, dtype=np.int16)
        payoff_code = np.empty(total, dtype=np.int8)
        cohort_id   = np.empty(total, dtype=np.int32)
        position_limit = np.empty(total, dtype=np.int64)

        i = 0
        for c_id, c in enumerate(cohorts):
            j = i + c.count
            memory[i:j]      = c.memory
            strategies[i:j]  = c.strategies
            payoff_code[i:j] = payoff_map[c.payoff]
            cohort_id[i:j]   = c_id
            position_limit[i:j] = c.position_limit
            i = j

        return {
            "N": total,
            "memory": memory,
            "strategies": strategies,
            "payoff_code": payoff_code,
            "cohort_id": cohort_id,
            "position_limit": position_limit,
            "cohorts": cohorts,
            "payoff_map": payoff_map
        }

    # --- spec normalization (implement 3 paths) ---
    def _cohorts_from_spec(self, spec):
        if "cohorts" in spec: return self._normalize_cohorts(spec)
        if "grid" in spec:    return self._cohorts_from_grid(spec)
        if "sample" in spec:  return self._cohorts_from_sample(spec)
        raise ValueError("Provide one of: 'cohorts', 'grid', or 'sample'.")

    #TODO: logic added around position limit to ensure that None is returned if empty etc.
    def _normalize_cohorts(self, spec):
        total = spec.get("total")
        parts = spec["cohorts"]
        absol = sum(p["count"] for p in parts if isinstance(p["count"], int))
        props = [p for p in parts if isinstance(p["count"], float)]
        if props and total is None:
            raise ValueError("Proportional counts require 'total'.")
        remaining = 0 if total is None else total - absol
        if total is not None and remaining < 0:
            raise ValueError("Absolute cohort counts exceed total.")
        out = [Cohort(p["count"], p["memory"], p["payoff"], p["strategies"], p["position_limit"])
               for p in parts if isinstance(p["count"], int) and p["count"]>0]
        if props:
            w = np.array([p["count"] for p in props], float); w /= w.sum()
            raw = w * remaining
            base = np.floor(raw).astype(int)
            leftover = remaining - base.sum()
            order = np.argsort(-(raw - base))
            base[order[:leftover]] += 1
            out += [Cohort(int(c), p["memory"], p["payoff"], p["strategies"], p["position_limit"])
                    for p, c in zip(props, base) if c>0]
        return out

    def _cohorts_from_grid(self, spec):

        total = spec["total"]
        axes = spec["grid"]
        names = list(axes.keys())
        values = [axes[n]["values"] for n in names]
        weights = [np.array(axes[n].get("weights", [1]*len(axes[n]["values"])),
                            float) for n in names]
        weights = [w/w.sum() for w in weights]
        combos, fracs = [], []
        for idxs in it.product(*[range(len(v)) for v in values]):
            params = {names[i]: values[i][j] for i, j in enumerate(idxs)}
            w = float(np.prod([weights[i][j] for i, j in enumerate(idxs)]))
            combos.append(params); fracs.append(w)
        fracs = np.array(fracs); fracs /= fracs.sum()
        counts_raw = fracs * total
        base = np.floor(counts_raw).astype(int)
        leftover = total - base.sum()
        order = np.argsort(-(counts_raw - base))
        base[order[:leftover]] += 1
        out = [Cohort(int(c), p["memory"], p["payoff"], p["strategies"], p["position_limit"])
               for p, c in zip(combos, base) if c>0]
        return out


    #TODO:review code to make sure that position limit functions and check what this does 
    def _cohorts_from_sample(self, spec):
        total = spec["total"]; S = spec["sample"]
        mem = self.rng.choice(S["memory"]["choice"], p=S["memory"]["p"], size=total)
        pay = self.rng.choice(S["payoff"]["choice"], p=S["payoff"]["p"], size=total)
        k   = self.rng.choice(S["strategies"]["choice"], p=S["strategies"]["p"], size=total)
        uniq, counts = np.unique(list(zip(mem, pay, k)), axis=0, return_counts=True)
        return [Cohort(int(c), int(m), str(p), int(s)) for (m,p,s), c in zip(uniq, counts)]
