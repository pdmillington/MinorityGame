#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_scaled_binary.py
========================

Compares ScaledMG and BinaryMG payoff schemes across a range of
attendance, strategy-switching and per-agent success diagnostics.

Usage
-----
    python experiments/compare_scaled_binary.py --config config/compare_scaled_binary.json

Config fields (all optional, defaults shown)
--------------------------------------------
    m          : 8
    s          : 5
    N          : 501
    rounds     : 10000
    intervals  : [500, 1000, 2000, 5000]
    output_dir : "results/compare_scaled_binary"
    run_name   : "compare_scaled_binary"

Output (under output_dir/run_name_<timestamp>/)
-----------------------------------------------
    attendance_series.pdf
    attendance_histogram.pdf
    strategy_switches.pdf
    success_evolution.pdf
    switch_vs_success.pdf
    <label>_individual_heatmap.pdf   (one per payoff)
    <label>_top_bottom_success.pdf   (one per payoff)
"""

import os
import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, pearsonr

from core.game import Game
from core.game_config import GameConfig
from utils.logger import RunLogger


PAYOFFS = ["ScaledMG", "BinaryMG"]


@dataclass
class CompareScaledBinaryConfig:
    m: int = 8
    s: int = 5
    N: int = 501
    rounds: int = 10_000
    intervals: List[int] = field(default_factory=lambda: [500, 1000, 2000, 5000])
    output_dir: str = "results/compare_scaled_binary"
    run_name: str = "compare_scaled_binary"


def load_config(path: str) -> CompareScaledBinaryConfig:
    with open(path) as f:
        data = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return CompareScaledBinaryConfig(**data)


# ── Game runner ───────────────────────────────────────────────────────────────

def run_games(cfg: CompareScaledBinaryConfig) -> Dict[str, dict]:
    """Run one game per payoff scheme; return {label: results_dict}."""
    games = {}
    for label in PAYOFFS:
        population_spec = {"cohorts": [{
            "count": cfg.N,
            "agent_type": "strategic",
            "memory": cfg.m,
            "strategies": cfg.s,
            "payoff": label,
        }]}
        game_cfg = GameConfig(
            rounds=cfg.rounds,
            record_agent_series=True,
            record_strategies=False,
        )
        game = Game(population_spec=population_spec, cfg=game_cfg)
        games[label] = game.run()
    return games


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_attendance_series(games: Dict[str, dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    for label, r in games.items():
        plt.plot(r["Attendance"], label=label, alpha=0.8)
    plt.title("Attendance Time Series Comparison")
    plt.xlabel("Round")
    plt.ylabel("A(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_attendance_histogram(games: Dict[str, dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, r in games.items():
        plt.hist(r["Attendance"], bins=80, alpha=0.6, label=label, edgecolor="black")
    plt.title("Attendance Distribution Comparison")
    plt.xlabel("A(t)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_strategy_switching(games: Dict[str, dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, r in games.items():
        switches = r.get("strategy_switches")
        if switches is not None:
            plt.hist(switches, bins=30, alpha=0.6, edgecolor="black", label=label)
    plt.title("Strategy Switching Frequency")
    plt.xlabel("Switches per agent")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_success_evolution(games: Dict[str, dict],
                           intervals: List[int],
                           save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, r in games.items():
        wins = r.get("wins")          # (rounds+1, N)
        if wins is None:
            continue
        # Convert cumulative wins to per-round indicator (diff, then average)
        wins_per_round = np.diff(wins, axis=0).astype(float)  # (rounds, N)
        rounds, N = wins_per_round.shape
        for interval in intervals:
            full = rounds // interval
            seg = wins_per_round[:full * interval].reshape(full, interval, N)
            avg = seg.mean(axis=(1, 2))   # mean over interval and players
            ax.plot(np.arange(full) * interval, avg,
                    label=f"{label} Δt={interval}")
    ax.set_title("Evolution of Average Success Rate")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean win rate")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_switch_vs_success(games: Dict[str, dict], save_path: str) -> List[str]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    log_lines = []
    for label, r in games.items():
        switches = r.get("strategy_switches")
        final_wins = r.get("final_wins")
        if switches is None or final_wins is None:
            continue
        success = final_wins / max(r.get("rounds", 1), 1)
        plt.scatter(switches, success, alpha=0.6, label=label)
        if len(switches) > 1:
            corr, _ = pearsonr(switches, success)
            log_lines.append(f"{label}: switches–success correlation = {corr:.3f}")
    plt.title("Strategy Switching vs Success Rate")
    plt.xlabel("Strategy switches")
    plt.ylabel("Win rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return log_lines


def plot_individual_heatmap(r: dict, label: str, interval: int,
                            save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    wins = r.get("wins")
    if wins is None:
        return
    wins_per_round = np.diff(wins, axis=0).astype(float)  # (rounds, N)
    rounds, N = wins_per_round.shape
    full = rounds // interval
    seg = wins_per_round[:full * interval].reshape(full, interval, N)
    matrix = seg.mean(axis=1).T           # (N, full_intervals)
    avg_success = matrix.mean(axis=1)
    sorted_matrix = matrix[np.argsort(avg_success)[::-1]]

    plt.figure(figsize=(10, 6))
    plt.imshow(sorted_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Win rate")
    plt.xlabel("Interval")
    plt.ylabel("Agent (sorted by mean win rate)")
    plt.title(f"{label}: Individual win-rate evolution (interval={interval})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_top_bottom(r: dict, label: str, top_n: int, interval: int,
                    save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    wins = r.get("wins")
    if wins is None:
        return
    wins_per_round = np.diff(wins, axis=0).astype(float)  # (rounds, N)
    rounds, N = wins_per_round.shape
    full = rounds // interval
    seg = wins_per_round[:full * interval].reshape(full, interval, N)
    matrix = seg.mean(axis=1).T           # (N, full)
    avg = matrix.mean(axis=1)
    top = np.argsort(avg)[-top_n:]
    bottom = np.argsort(avg)[:top_n]
    x = np.arange(full) * interval

    plt.figure(figsize=(10, 6))
    for idx in top:
        plt.plot(x, matrix[idx], linestyle="-", lw=2, label=f"Top agent {idx}")
    for idx in bottom:
        plt.plot(x, matrix[idx], linestyle="--", label=f"Bottom agent {idx}")
    plt.title(f"{label}: Best and Worst Agent Win-Rate Evolution")
    plt.xlabel("Round")
    plt.ylabel("Win rate")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_attendance_stats(games: Dict[str, dict]) -> List[str]:
    lines = ["\nAttendance Summary"]
    for label, r in games.items():
        A = np.asarray(r["Attendance"], dtype=float)
        lines += [
            f"\n{label}:",
            f"  Mean:     {A.mean():.3f}",
            f"  Std:      {A.std():.3f}",
            f"  Skewness: {skew(A):.3f}",
            f"  Kurtosis: {kurtosis(A):.3f}",
        ]
    print("\n".join(lines))
    return lines


# ── Main entry point ──────────────────────────────────────────────────────────

def run_compare(cfg: CompareScaledBinaryConfig) -> None:
    logger = RunLogger(
        base_save_dir=cfg.output_dir,
        module=cfg.run_name,
        payoff="ScaledMG_vs_BinaryMG",
    )
    base = logger.get_dir()
    print(f"Output: {base}")

    games = run_games(cfg)

    plot_attendance_series(games,    f"{base}/attendance_series.pdf")
    plot_attendance_histogram(games, f"{base}/attendance_histogram.pdf")
    plot_strategy_switching(games,   f"{base}/strategy_switches.pdf")
    plot_success_evolution(games, cfg.intervals, f"{base}/success_evolution.pdf")
    log_lines = plot_switch_vs_success(games, f"{base}/switch_vs_success.pdf")

    for label, r in games.items():
        plot_individual_heatmap(r, label, 1000,
                                f"{base}/{label.lower()}_heatmap.pdf")
        plot_top_bottom(r, label, top_n=3, interval=500,
                        save_path=f"{base}/{label.lower()}_top_bottom.pdf")

    stats_lines = print_attendance_stats(games)
    logger.log_params({"m": cfg.m, "s": cfg.s, "N": cfg.N, "rounds": cfg.rounds,
                       "intervals": cfg.intervals})
    logger.close()
    print(f"Done. Run folder: {base}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ScaledMG vs BinaryMG payoff schemes."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config. Uses defaults if omitted.")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else CompareScaledBinaryConfig()
    run_compare(cfg)


if __name__ == "__main__":
    main()
