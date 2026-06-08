#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
challet_fig2_replication.py

Replicates Figure 2 from:
    Challet, Chessa, Marsili, Zhang (2000)
    "Modeling Market Mechanism with Minority Game"

Figure 2: Gain of producers and speculators vs number of producers
          (in P units, where P = 2^m).
          N_speculators = 641, m = 8, s = 2, alpha = 0.40
          Producer count swept from 0 to 10P = 2560
          Average over 200 realisations.

Note: This is the canonical MG (all agents trade every round),
      not the grand canonical version. c=0 in the paper refers
      to the capital parameter (no bankruptcy), not grand canonical.

Usage
-----
    python experiments/challet_fig2_replication.py
    python experiments/challet_fig2_replication.py --config config/challet_fig2.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.logger import RunLogger


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ChallletFig2Config:
    # Population — matching paper exactly
    n_speculators:   int       = 641
    m:               int       = 8        # P = 2^m = 256
    s:               int       = 2
    payoff_key:      str       = "ScaledMG"

    # Producer sweep: 0 to 10P in steps of P
    # P = 2^8 = 256, so counts are 0, 256, 512, ..., 2560
    # Finer grid near zero for better curve shape
    producer_counts: List[int] = field(default_factory=lambda: (
        [0] +
        list(range(64, 257, 64)) +      # 0.25P, 0.5P, 0.75P, 1P
        list(range(512, 2561, 256))     # 2P, 3P, ... 10P
    ))

    # Game
    rounds:          int       = 10_000
    num_runs:        int       = 200
    grand_canonical: bool      = False    # canonical version for this figure
    gc_threshold:    float     = 0.0

    # Output
    output_dir:      str       = "results/challet_fig2"
    run_name:        str       = "challet_fig2"


def load_config(path: str) -> ChallletFig2Config:
    with open(path) as f:
        data = {k: v for k, v in json.load(f).items()
                if not k.startswith("_")}
    return ChallletFig2Config(**data)


# ── Population spec builder ───────────────────────────────────────────────────

def _make_spec(n_speculators: int, n_producers: int,
               m: int, s: int, payoff_key: str) -> dict:
    """
    Two-cohort spec:
    Cohort 0: speculators  (s strategies)
    Cohort 1: producers    (1 strategy, always_trade=True)
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
    return {"total": n_speculators + n_producers, "cohorts": cohorts}


# ── Module-level worker ───────────────────────────────────────────────────────

def _run_one(args: tuple) -> dict:
    """
    Run one game and return per-cohort mean gain per round.
    """
    (n_producers, run_idx, n_speculators, m, s,
     payoff_key, rounds, grand_canonical, gc_threshold) = args

    from core.game import Game
    from core.game_config import GameConfig

    spec = _make_spec(n_speculators, n_producers, m, s, payoff_key)

    cfg_game = GameConfig(
        rounds=rounds,
        lambda_=1.0 / (n_speculators * 50),
        mm=None,
        price=100,
        seed=hash((n_producers, run_idx)) & 0x7FFFFFFF,
        record_agent_series=True,
        grand_canonical=grand_canonical,
        gc_threshold=gc_threshold,
    )

    game    = Game(population_spec=spec, cfg=cfg_game)
    results = game.run()

    cohort_ids   = results["cohort_ids"]
    final_points = results["final_points"]

    spec_mask = cohort_ids == 0
    prod_mask = cohort_ids == 1

    gain_spec = float(np.mean(final_points[spec_mask])) / rounds
    gain_prod = (float(np.mean(final_points[prod_mask])) / rounds
                 if prod_mask.any() else np.nan)

    # Active speculators per round from position series
    position = results.get("position")
    if position is not None:
        delta       = np.abs(np.diff(position[:, spec_mask], axis=0))
        mean_active = float(np.mean((delta > 0).sum(axis=1)))
    else:
        mean_active = np.nan

    return {
        "n_producers":        n_producers,
        "run_idx":            run_idx,
        "gain_speculator":    gain_spec,
        "gain_producer":      gain_prod,
        "active_speculators": mean_active,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_challet_fig2(cfg: ChallletFig2Config) -> None:

    P     = 2 ** cfg.m     # = 256 for m=8
    alpha = P / cfg.n_speculators

    logger = RunLogger(
        base_save_dir=cfg.output_dir,
        module=cfg.run_name,
        payoff=cfg.payoff_key,
    )
    logger.log_params({**asdict(cfg), "P": P, "alpha": alpha})

    total = len(cfg.producer_counts) * cfg.num_runs
    print(f"P = 2^{cfg.m} = {P},  alpha = {alpha:.3f}")
    print(f"Output: {logger.get_dir()}")
    print(f"Total games: {total}  "
          f"({len(cfg.producer_counts)} producer counts "
          f"× {cfg.num_runs} runs)")

    tasks = [
        (n_prod, run_idx,
         cfg.n_speculators, cfg.m, cfg.s,
         cfg.payoff_key, cfg.rounds,
         cfg.grand_canonical, cfg.gc_threshold)
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
                f"n_producers={r['n_producers']:4d}  "
                f"run={r['run_idx']}",
                end="\r",
            )

    print(f"\nAll games complete.")

    df = pd.DataFrame(rows)
    logger.log_table(df, "results")

    summary = (
        df.groupby("n_producers")
        .agg(
            mean_gain_spec   =("gain_speculator",    "mean"),
            std_gain_spec    =("gain_speculator",    "std"),
            mean_gain_prod   =("gain_producer",      "mean"),
            std_gain_prod    =("gain_producer",      "std"),
            mean_active_spec =("active_speculators", "mean"),
            std_active_spec  =("active_speculators", "std"),
            n_runs           =("run_idx",            "count"),
        )
        .reset_index()
    )
    summary["P_units"] = summary["n_producers"] / P   # x-axis in P units
    logger.log_table(summary, "summary")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig2  = _plot_gain(summary, cfg, P, alpha)
    figAC = _plot_active(summary, cfg, P, alpha)

    for fig, name in [(fig2, "fig2_gain"), (figAC, "active_speculators")]:
        path = logger.get_dir() + f"/{name}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved: {path}")

    logger.close()
    print(f"Done. Run folder: {logger.get_dir()}")


def _plot_gain(summary: pd.DataFrame,
               cfg: ChallletFig2Config,
               P: int, alpha: float) -> plt.Figure:
    """
    Gain of producers and speculators vs number of producers in P units.
    Matches Challet et al. 2000 Fig 2 layout.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    se_spec = summary["std_gain_spec"] / np.sqrt(cfg.num_runs)
    se_prod = summary["std_gain_prod"] / np.sqrt(cfg.num_runs)

    ax.errorbar(
        summary["P_units"], summary["mean_gain_spec"],
        yerr=se_spec,
        marker="o", color="steelblue", linewidth=1.5,
        capsize=3, label="Speculators",
    )

    prod = summary.dropna(subset=["mean_gain_prod"])
    ax.errorbar(
        prod["P_units"], prod["mean_gain_prod"],
        yerr=se_prod.loc[prod.index],
        marker="s", color="tomato", linewidth=1.5,
        capsize=3, label="Producers",
    )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Number of producers (in $P = 2^m$ units)")
    ax.set_ylabel("Mean gain per agent per round")
    ax.set_title(
        f"Gain vs Producer Count\n"
        f"(cf. Challet et al. 2000, Fig. 2)\n"
        f"$N_s$={cfg.n_speculators}, m={cfg.m}, s={cfg.s}, "
        f"$P$={P}, $\\alpha$={alpha:.2f}, "
        f"rounds={cfg.rounds:,}, runs={cfg.num_runs}",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_active(summary: pd.DataFrame,
                 cfg: ChallletFig2Config,
                 P: int, alpha: float) -> plt.Figure:
    """Mean active speculators per round vs producer count."""
    fig, ax = plt.subplots(figsize=(6, 4))
    se = summary["std_active_spec"] / np.sqrt(cfg.num_runs)

    ax.errorbar(
        summary["P_units"], summary["mean_active_spec"],
        yerr=se, marker="o", color="steelblue",
        linewidth=1.5, capsize=3,
    )
    ax.axhline(cfg.n_speculators, color="grey", linestyle="--",
               linewidth=0.8, label=f"Max ($N_s$={cfg.n_speculators})")
    ax.set_xlabel(r"Number of producers (in $P = 2^m$ units)")
    ax.set_ylabel("Mean active speculators per round")
    ax.set_title(
        f"Active Speculators vs Producer Count\n"
        f"$N_s$={cfg.n_speculators}, m={cfg.m}, s={cfg.s}, "
        f"$P$={P}, $\\alpha$={alpha:.2f}, "
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
        description="Replicate Challet et al. 2000 Fig 2."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config. If omitted, defaults are used.",
    )
    args = parser.parse_args()
    cfg  = load_config(args.config) if args.config else ChallletFig2Config()
    run_challet_fig2(cfg)


if __name__ == "__main__":
    main()
