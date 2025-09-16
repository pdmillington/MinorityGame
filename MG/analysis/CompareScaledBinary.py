#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:43:02 2025

@author: petermillington
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, pearsonr
from datetime import datetime
from core.game import Game
from payoffs.mg import ScaledMGPayoff, BinaryMGPayoff
from utils.logger import log_simulation  # Moved logging utility to reusable utils

def run_comparative_games(m, s, N, rounds):
    games = {}
    for label, PayoffClass in {"Scaled": ScaledMGPayoff, "Binary": BinaryMGPayoff}.items():
        game = Game(
            num_players=N,
            memory=m,
            num_strategies=s,
            rounds=rounds,
            payoff_scheme=PayoffClass()
        )
        game.run()
        games[label] = game
    return games

def compare_attendance_series(games, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    for label, game in games.items():
        plt.plot(game.actions, label=f"{label} Payoff", alpha=0.8)
    plt.title("Attendance Time Series Comparison")
    plt.xlabel("Round")
    plt.ylabel("Total Action A(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def compare_attendance_histogram(games, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, game in games.items():
        plt.hist(game.actions, bins=80, alpha=0.6, label=f"{label} Payoff", edgecolor='black')
    plt.title("Attendance Distribution Comparison")
    plt.xlabel("Total Action A(t)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def compare_strategy_switching(games, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, game in games.items():
        switches = [getattr(p, 'strategy_switches', 0) for p in game.players]
        plt.hist(switches, bins=30, alpha=0.6, edgecolor='black', label=f"{label} Payoff")
    plt.title("Strategy Switching Frequency")
    plt.xlabel("Switches")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def compare_average_rewards(games, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, game in games.items():
        rewards = [np.mean(p.wins_per_round) for p in game.players]
        plt.hist(rewards, bins=30, alpha=0.6, edgecolor='black', label=f"{label} Payoff")
    plt.title("Average Success Rate per Player")
    plt.xlabel("Average Success Rate")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def compare_success_evolution(games, interval_lengths, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, game in games.items():
        rounds = len(game.players[0].wins_per_round)

        for interval in interval_lengths:
            segment_means = []
            for p in game.players:
                wins = np.array(p.wins_per_round)
                full_intervals = rounds // interval
                wins = wins[:full_intervals * interval]
                reshaped = wins.reshape(-1, interval)
                interval_means = reshaped.mean(axis=1)
                segment_means.append(interval_means)

            avg_over_players = np.mean(segment_means, axis=0)
            ax.plot(np.arange(len(avg_over_players)) * interval, avg_over_players,
                    label=f"{label} Δt={interval}")

    ax.set_title("Evolution of Average Success Rate")
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg Success Rate")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def plot_top_and_bottom_agents(game, top_n=3, interval=500, save_path="plots/top_bottom_success_evolution.pdf"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    avg_wins = [np.mean(p.wins_per_round) for p in game.players]
    top_indices = np.argsort(avg_wins)[-top_n:]
    bottom_indices = np.argsort(avg_wins)[:top_n]
    points = len(game.players[0].wins_per_round)//interval
    x = np.arange(points) * interval

    plt.figure(figsize=(10, 6))
    for idx in top_indices:
        plt.plot(x, [np.mean(game.players[idx].wins_per_round[i * interval:(i+1) * interval]) for i in range(points)], label=f"Top {idx}", linestyle="-", linewidth=2)
    for idx in bottom_indices:
        plt.plot(x, [np.mean(game.players[idx].wins_per_round[i * interval:(i+1) * interval]) for i in range(points)], label=f"Bottom {idx}", linestyle="--")

    plt.title("Success Evolution of Best and Worst Agents")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Success")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    
def plot_individual_success_heatmap(game, interval=1000, save_path="plots/individual_success_heatmap.pdf"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rounds = len(game.players[0].wins_per_round)
    num_players = len(game.players)
    full_intervals = rounds // interval

    success_matrix = np.zeros((num_players, full_intervals))
    avg_success = []

    for i, p in enumerate(game.players):
        wins = np.array(p.wins_per_round[:full_intervals * interval])
        reshaped = wins.reshape(full_intervals, interval)
        interval_means = reshaped.mean(axis=1)
        success_matrix[i, :] = interval_means
        avg_success.append(np.mean(interval_means))

    # Sort players by their average success rate
    sorted_indices = np.argsort(avg_success)[::-1]  # descending
    sorted_matrix = success_matrix[sorted_indices, :]

    plt.figure(figsize=(10, 6))
    plt.imshow(sorted_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Success Rate")
    plt.xlabel("Interval")
    plt.ylabel("Sorted Player Index (by avg success)")
    plt.title(f"Individual Success Rate Evolution (interval={interval})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_switch_vs_success(games, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    log_lines = []
    for label, game in games.items():
        switches = np.array([p.strategy_switches for p in game.players])
        success = np.array([np.mean(p.wins_per_round) for p in game.players])
        plt.scatter(switches, success, alpha=0.6, label=f"{label} Payoff")

        if len(switches) > 1:
            corr, _ = pearsonr(switches, success)
            log_lines.append(f"{label} Payoff - Correlation between switches and success: {corr:.3f}")

    plt.title("Strategy Switching vs. Success Rate")
    plt.xlabel("Number of Strategy Switches")
    plt.ylabel("Average Success Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_path)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return log_lines

def print_attendance_statistics(games):
    lines = ["\nAttendance Series Summary:\n"]
    for label, game in games.items():
        actions = np.array(game.actions)
        mean = np.mean(actions)
        std = np.std(actions)
        sk = skew(actions)
        kurt = kurtosis(actions)
        lines.append(f"{label} Payoff:")
        lines.append(f"  Mean A(t):      {mean:.3f}")
        lines.append(f"  Std Dev A(t):   {std:.3f}")
        lines.append(f"  Skewness A(t):  {sk:.3f}")
        lines.append(f"  Kurtosis A(t):  {kurt:.3f}\n")
    print("\n".join(lines))
    return lines

def compare_scaled_vs_binary(m=8, s=5, N=1001, rounds=40000, intervals=[500, 1000, 2000, 5000]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = f"plots/compare/m{m}_s{s}_r{rounds}_{timestamp}"

    series_path = f"{base_path}_attendance_series.pdf"
    hist_path = f"{base_path}_attendance_histogram.pdf"
    switches_path = f"{base_path}_switches.pdf"
    rewards_path = f"{base_path}_avg_rewards.pdf"
    success_path = f"{base_path}_success_evolution.pdf"
    switch_vs_success_path = f"{base_path}_switch_vs_success.pdf"

    games = run_comparative_games(m, s, N, rounds)
    compare_attendance_series(games, save_path=series_path)
    compare_attendance_histogram(games, save_path=hist_path)
    compare_strategy_switching(games, save_path=switches_path)
    compare_average_rewards(games, save_path=rewards_path)
    compare_success_evolution(games, interval_lengths=intervals, save_path=success_path)
    switch_success_logs=plot_switch_vs_success(games, save_path=switch_vs_success_path)
    
    heatmap_paths = {}
    for label, game in games.items():
        heatmap_path = f"{base_path}_{label.lower()}_individual_heatmap.pdf"
        plot_individual_success_heatmap(game, interval=1000, save_path=heatmap_path)
        heatmap_paths[label] = heatmap_path
        
    for label, game in games.items():
        plot_top_and_bottom_agents(game, save_path=f"{base_path}_{label.lower()}_topbottom_success.pdf")
    
    log_lines = print_attendance_statistics(games)

    metadata = [
        f"Memory: {m}",
        f"Strategies: {s}",
        f"Rounds: {rounds}",
        f"Timestamp: {timestamp}",
        f"Saved Attendance Series: {series_path}",
        f"Saved Histogram: {hist_path}",
        f"Saved Strategy Switches: {switches_path}",
        f"Saved Avg Rewards: {rewards_path}",
        f"Saved Success Evolution: {success_path}"
    ] + [f"Saved {label} Success Heatmap: {path}" for label, path in heatmap_paths.items()] + \
        [f"Saved {label} Top/Bottom Agents Plot: {base_path}_{label.lower()}_topbottom_succes.pdf" for label in games]+ \
        log_lines + switch_success_logs

    log_simulation(metadata)

if __name__ == "__main__":
    compare_scaled_vs_binary()
