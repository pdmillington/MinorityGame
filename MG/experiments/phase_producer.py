#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_producer.py

Phase diagram experiment varying the proportion of single-strategy
(producer) agents alongside adaptive multi-strategy (speculator) agents.

Produces a family of normalised-variance vs alpha curves, one per
producer proportion, on a single log-log plot — directly comparable
to the standard MG phase diagram.

Usage
-----
    python experiments/phase_producer.py --config config/phase_producer.json

Config fields (all optional, defaults shown)
--------------------------------------------
    payoff_key       : "ScaledMG"
    m_values         : [3, 4, 5, 6, 7, 8, 9, 10, 11]
    num_players      : 301          # total N (adaptive + producers)
    num_strategies   : 2            # strategies for adaptive cohort
    proportions      : [0.0, 0.1, 0.25, 0.5]
    position_limit   : null
    rounds           : 10000
    num_games        : 20
    output_dir       : "results/phase_producer"
    run_name         : "phase_producer"
    log_path         : "logs/simulation_log.txt"

Output (under results/phase_producer/<timestamp>_phase_producer/)
-----------------------------------------------------------------
    phase_diagram_producer.pdf   — main figure
    results.csv                  — raw sigma2/N per (proportion, m, game)
    summary.csv                  — mean sigma2/N per (proportion, m)
    params.json
    run_info.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from core.game import Game
from core.game_config import GameConfig
from utils.logger import RunLogger


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ProducerPhaseDiagramConfig:
    payoff_key:     str         = "ScaledMG"
    m_values:       List[int]   = field(default_factory=lambda: list(range(3, 12)))
    num_players:    int         = 301
    num_strategies: int         = 2
    proportions:    List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5])
    position_limit: int         = None
    rounds:         int         = 10_000
    num_games:      int         = 20
    output_dir:     str         = "results/phase_producer"
    run_name:       str         = "phase_producer"
    log_path:       str         = "logs/simulation_log.txt"


def load_config(path: str) -> ProducerPhaseDiagramConfig:
    with open(path) as f:
        data = {k: v for k, v in json.load(f).items()
                if not k.startswith("_")}
    return ProducerPhaseDiagramConfig(**data)


# ── Population spec builder ───────────────────────────────────────────────────

def _make_spec(m: int, proportion: float, num_players: int,
               num_strategies: int, payoff_key: str,
               position_limit) -> dict:
    """
    Build a two-cohort population spec for a given m and producer proportion.
    Cohort 0: adaptive agents  (strategies = num_strategies)
    Cohort 1: producers        (strategies = 1)
    """
    n_prod     = max(1, round(proportion * num_players)) if proportion > 0 else 0
    n_adaptive = num_players

    cohorts = []
    if n_adaptive > 0:
        cohorts.append({
            "count":          n_adaptive,
            "memory":         m,
            "strategies":     num_strategies,
            "payoff":         payoff_key,
            "position_limit": position_limit,
            "agent_type":     "strategic",
            "score_lambda":   0.0,
        })
    if n_prod > 0:
        cohorts.append({
            "count":          n_prod,
            "memory":         m,
            "strategies":     1,
            "payoff":         payoff_key,
            "position_limit": position_limit,
            "agent_type":     "strategic",
            "score_lambda":   0.0,
        })

    return {"total": n_adaptive + n_prod, "cohorts": cohorts}


# ── Module-level worker (must be top-level for pickling) ─────────────────────

def _simulate_one(args: tuple) -> dict:
    """
    Run one game for a given (m, proportion, game_id) and return sigma2.

    Returns
    -------
    dict with keys: m, proportion, game_id, sigma2, sigma2_N
    """
    (m, proportion, game_id,
     num_players, num_strategies, payoff_key,
     position_limit, rounds) = args

    # Import here so each worker process gets its own fresh import
    from core.game import Game
    from core.game_config import GameConfig

    spec     = _make_spec(m, proportion, num_players, num_strategies,
                          payoff_key, position_limit)
    cfg_game = GameConfig(
        rounds=rounds,
        lambda_=1.0 / (num_players * 50),
        mm=None,
        price=100,
        seed=hash((m, proportion, game_id)) & 0x7FFFFFFF,
        record_agent_series=True,
    )

    game    = Game(population_spec=spec, cfg=cfg_game)
    results = game.run()

    sigma2 = float(np.var(results["Attendance"]))

    return {
        "m":          m,
        "proportion": proportion,
        "game_id":    game_id,
        "sigma2":     sigma2,
        "sigma2_N":   sigma2 / num_players,
        "alpha":      (2 ** m) / num_players,
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run_phase_producer(cfg: ProducerPhaseDiagramConfig) -> None:

    logger = RunLogger(
        base_save_dir=cfg.output_dir,
        module=cfg.run_name,
        payoff=cfg.payoff_key,
    )
    logger.log_params(asdict(cfg))

    print(f"Output: {logger.get_dir()}")
    print(f"Total games: "
          f"{len(cfg.m_values) * len(cfg.proportions) * cfg.num_games}")

    # Build full task list
    tasks = [
        (m, proportion, game_id,
         cfg.num_players, cfg.num_strategies, cfg.payoff_key,
         cfg.position_limit, cfg.rounds)
        for proportion in cfg.proportions
        for m in cfg.m_values
        for game_id in range(cfg.num_games)
    ]

    rows = []
    total = len(tasks)
    done  = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_simulate_one, t): t for t in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            r = rows[-1]
            print(
                f"  [{done}/{total}]  proportion={r['proportion']:.0%}"
                f"  m={r['m']}  game={r['game_id']}",
                end="\r",
            )

    print(f"\nAll games complete.")

    # ── Results ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    logger.log_table(df, "results")

    summary = (
        df.groupby(["proportion", "m", "alpha"])["sigma2_N"]
        .agg(mean_sigma2_N="mean", std_sigma2_N="std", n="count")
        .reset_index()
    )
    logger.log_table(summary, "summary")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = _plot_phase_diagram(summary, cfg)
    pdf_path = logger.subdir("figures") + "/phase_diagram_producer.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    logger.log_artifact(pdf_path)
    print(f"Figure: {pdf_path}")

    logger.close()
    print(f"Done. Run folder: {logger.get_dir()}")


def _plot_phase_diagram(
    summary: pd.DataFrame,
    cfg: ProducerPhaseDiagramConfig,
) -> plt.Figure:
    """
    Log-log phase diagram: sigma2/N vs alpha, one line per proportion.
    Error bars show ±1 std across games.
    """
    proportions = sorted(summary["proportion"].unique())
    colours     = cm.viridis(np.linspace(0.1, 0.9, len(proportions)))

    fig, ax = plt.subplots(figsize=(7, 5))

    for proportion, colour in zip(proportions, colours):
        sub = summary[summary["proportion"] == proportion].sort_values("alpha")

        ax.errorbar(
            sub["alpha"],
            sub["mean_sigma2_N"],
            yerr=sub["std_sigma2_N"],
            marker="o",
            color=colour,
            label=f"{proportion:.0%} producers",
            linewidth=1.5,
            markersize=5,
            capsize=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha_{\mathrm{eff}} = 2^m / N_{\mathrm{adaptive}}$", fontsize=11)
    ax.set_ylabel(r"$\sigma^2 / N$", fontsize=11)
    ax.set_title(
        f"Phase Diagram — Producer Proportion Sweep\n"
        f"N={cfg.num_players}, s={cfg.num_strategies}, "
        f"payoff={cfg.payoff_key}, "
        f"rounds={cfg.rounds:,}, games={cfg.num_games}",
        fontsize=9,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase diagram sweep over producer proportions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. If omitted, defaults are used.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else ProducerPhaseDiagramConfig()
    run_phase_producer(cfg)


if __name__ == "__main__":
    main()
