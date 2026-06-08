#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gc_challet_replication.py

Replicates Figures 7 and 8 from:
    Challet, Chessa, Marsili, Zhang (2000)
    "From Minority Games to Real Markets"
    Quantitative Finance 1(1): 168-176

Figure 7: Average gain per agent vs number of producers
          for speculators and producers separately.

Figure 8: Average number of active speculators vs number of producers.

Setup (matching paper as closely as possible)
---------------------------------------------
    N_speculators = 107   (fixed)
    m = 5                 (alpha = 2^5/107 ≈ 0.30, near critical point)
    s = 2                 (strategies per speculator)
    payoff = ScaledMG     (g = A_t, matching paper's gain definition)
    grand_canonical = True
    gc_threshold = 0.0    (trade if best score >= 0)
    Producers: always_trade = True, s = 1
    Producer count swept from 0 to N_speculators

Usage
-----
    python experiments/gc_challet_replication.py
    python experiments/gc_challet_replication.py --config config/gc_challet.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.game_config import GameConfig
from utils.logger import RunLogger


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GCChallletConfig:
    # Population
    n_speculators:    int         = 107
    m:                int         = 5
    s:                int         = 2
    payoff_key:       str         = "ScaledMG"

    # Producer sweep
    producer_counts:  List[int]   = field(
        default_factory=lambda: list(range(0, 108, 11))
    )                                        # 0, 11, 22, ... 107

    # Game
    rounds:           int         = 10_000
    num_runs:         int         = 100      # 500 for publication quality
    gc_threshold:     float       = 0.0

    # Output
    output_dir:       str         = "results/gc_challet"
    run_name:         str         = "gc_challet"


def load_config(path: str) -> GCChallletConfig:
    with open(path) as f:
        data = {k: v for k, v in json.load(f).items()
                if not k.startswith("_")}
    return GCChallletConfig(**data)


# ── Population spec builder ───────────────────────────────────────────────────

def _make_spec(n_speculators: int, n_producers: int, m: int,
               s: int, payoff_key: str) -> dict:
    """
    Two-cohort spec:
    Cohort 0: speculators  (s strategies, grand canonical)
    Cohort 1: producers    (1 strategy,   always_trade=True)
    """
    cohorts = [{
        "count":          n_speculators,
        "memory":         m,
        "strategies":     s,
        "payoff":         payoff_key,
        "position_limit": None,
        "agent_type":     "strategic",
        "score_lambda":   0.0,
        "always_trade":   False,
    }]
    if n_producers > 0:
        cohorts.append({
            "count":          n_producers,
            "memory":         m,
            "strategies":     1,
            "payoff":         payoff_key,
            "position_limit": None,
            "agent_type":     "strategic",
            "score_lambda":   0.0,
            "always_trade":   True,
        })
    total = n_speculators + n_producers
    return {"total": total, "cohorts": cohorts}


# ── Module-level worker ───────────────────────────────────────────────────────

def _run_one(args: tuple) -> dict:
    """
    Run one game and return per-cohort gain and mean active speculator count.

    Returns
    -------
    dict with keys:
        n_producers         int
        run_idx             int
        gain_speculator     float   mean points per round per speculator
        gain_producer       float   mean points per round per producer
                                    (nan if n_producers == 0)
        active_speculators  float   mean number of active speculators per round
    """
    (n_producers, run_idx, n_speculators, m, s,
     payoff_key, rounds, gc_threshold) = args

    from core.game import Game
    from core.game_config import GameConfig

    spec = _make_spec(n_speculators, n_producers, m, s, payoff_key)

    cfg_game = GameConfig(
        rounds=rounds,
        lambda_=1.0 / (n_speculators * 50),   # lambda based on N_speculators
        mm=None,
        price=100,
        seed=hash((n_producers, run_idx)) & 0x7FFFFFFF,
        record_agent_series=True,
        grand_canonical=True,
        gc_threshold=gc_threshold,
    )

    game    = Game(population_spec=spec, cfg=cfg_game)
    results = game.run()

    cohort_ids   = results["cohort_ids"]       # (N,)
    final_points = results["final_points"]     # (N,) cumulative

    # Mean gain per round per agent, by cohort
    spec_mask = cohort_ids == 0
    prod_mask = cohort_ids == 1

    gain_spec = float(np.mean(final_points[spec_mask])) / rounds
    gain_prod = float(np.mean(final_points[prod_mask])) / rounds \
                if prod_mask.any() else np.nan

    # Active speculators per round
    # position series shape (rounds+1, N) — active if position changed
    # More precisely: active if |chosen| > 0, i.e. position[t] != position[t-1]
    position = results["position"]              # (rounds+1, N)
    if position is not None:
        delta      = np.abs(np.diff(position[:, spec_mask], axis=0))  # (rounds, N_spec)
        active_per_round = (delta > 0).sum(axis=1)                    # (rounds,)
        mean_active = float(np.mean(active_per_round))
    else:
        mean_active = np.nan

    return {
        "n_producers":        n_producers,
        "run_idx":            run_idx,
        "gain_speculator":    gain_spec,
        "gain_producer":      gain_prod,
        "active_speculators": mean_active,
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run_gc_challet(cfg: GCChallletConfig) -> None:

    logger = RunLogger(
        base_save_dir=cfg.output_dir,
        module=cfg.run_name,
        payoff=cfg.payoff_key,
    )
    logger.log_params(asdict(cfg))

    total = len(cfg.producer_counts) * cfg.num_runs
    print(f"Output: {logger.get_dir()}")
    print(f"Total games: {total}  "
          f"({len(cfg.producer_counts)} producer counts × {cfg.num_runs} runs)")

    tasks = [
        (n_prod, run_idx,
         cfg.n_speculators, cfg.m, cfg.s,
         cfg.payoff_key, cfg.rounds, cfg.gc_threshold)
        for n_prod in cfg.producer_counts
        for run_idx in range(cfg.num_runs)
    ]

    rows = []
    done = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            r = rows[-1]
            print(
                f"  [{done}/{total}]  "
                f"n_producers={r['n_producers']}  run={r['run_idx']}",
                end="\r",
            )

    print(f"\nAll games complete.")

    df = pd.DataFrame(rows)
    logger.log_table(df, "results")

    # Summary
    summary = (
        df.groupby("n_producers")
        .agg(
            mean_gain_spec    =("gain_speculator",    "mean"),
            std_gain_spec     =("gain_speculator",    "std"),
            mean_gain_prod    =("gain_producer",      "mean"),
            std_gain_prod     =("gain_producer",      "std"),
            mean_active_spec  =("active_speculators", "mean"),
            std_active_spec   =("active_speculators", "std"),
            n_runs            =("run_idx",            "count"),
        )
        .reset_index()
    )
    # Producer proportion P (in units of N_speculators, as Challet uses)
    summary["P"] = summary["n_producers"] / cfg.n_speculators
    logger.log_table(summary, "summary")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig7 = _plot_gain(summary, cfg)
    fig8 = _plot_active(summary, cfg)

    for fig, name in [(fig7, "fig7_gain"), (fig8, "fig8_active")]:
        path = logger.get_dir() + f"/{name}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved: {path}")

    logger.close()
    print(f"Done. Run folder: {logger.get_dir()}")


def _plot_gain(summary: pd.DataFrame, cfg: GCChallletConfig) -> plt.Figure:
    """
    Figure 7: average gain per agent vs producer count P.
    Separate lines for speculators and producers.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    P = summary["P"]

    # Speculators
    ax.errorbar(
        P, summary["mean_gain_spec"],
        yerr=summary["std_gain_spec"] / np.sqrt(cfg.num_runs),
        marker="o", color="steelblue", linewidth=1.5,
        capsize=3, label="Speculators",
    )

    # Producers (NaN at P=0 since no producers exist)
    prod = summary.dropna(subset=["mean_gain_prod"])
    ax.errorbar(
        prod["P"], prod["mean_gain_prod"],
        yerr=prod["std_gain_prod"] / np.sqrt(cfg.num_runs),
        marker="s", color="tomato", linewidth=1.5,
        capsize=3, label="Producers",
    )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Number of producers $P$ (in units of $N_s$)")
    ax.set_ylabel("Mean gain per agent per round")
    ax.set_title(
        f"Grand Canonical MG — Gain vs Producer Count\n"
        f"(cf. Challet et al. 2000, Fig. 7)\n"
        f"$N_s$={cfg.n_speculators}, m={cfg.m}, s={cfg.s}, "
        f"rounds={cfg.rounds:,}, runs={cfg.num_runs}",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_active(summary: pd.DataFrame, cfg: GCChallletConfig) -> plt.Figure:
    """
    Figure 8: mean number of active speculators vs producer count P.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(
        summary["P"],
        summary["mean_active_spec"],
        yerr=summary["std_active_spec"] / np.sqrt(cfg.num_runs),
        marker="o", color="steelblue", linewidth=1.5, capsize=3,
    )

    ax.axhline(cfg.n_speculators, color="grey", linestyle="--",
               linewidth=0.8, label=f"Max ($N_s$={cfg.n_speculators})")
    ax.set_xlabel(r"Number of producers $P$ (in units of $N_s$)")
    ax.set_ylabel("Mean active speculators per round")
    ax.set_title(
        f"Grand Canonical MG — Active Speculators vs Producer Count\n"
        f"(cf. Challet et al. 2000, Fig. 8)\n"
        f"$N_s$={cfg.n_speculators}, m={cfg.m}, s={cfg.s}, "
        f"rounds={cfg.rounds:,}, runs={cfg.num_runs}",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replicate Challet et al. 2000 Figs 7 & 8 "
                    "using grand canonical MG."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config. If omitted, defaults are used.",
    )
    args   = parser.parse_args()
    cfg    = load_config(args.config) if args.config else GCChallletConfig()
    run_gc_challet(cfg)


if __name__ == "__main__":
    main()
