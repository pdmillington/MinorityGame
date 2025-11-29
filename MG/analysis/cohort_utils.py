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
        payoff_name = c.payoff.replace("Payoff", "")
        labels[cid] = f"m={c.memory}\nS={c.strategies}\n{payoff_name}"
    return labels

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
