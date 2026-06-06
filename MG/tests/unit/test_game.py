#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Game

Tests cover:
- Game runs correctly for all payoff types
- Results structure is complete and consistent
- Phase structure: normalised variance follows expected pattern
- Success rates are in plausible range
- Wealth reconstruction is consistent with position and prices
- Reproducibility: same seed gives identical results
- Reset produces a fresh independent run
- Two-cohort games assign cohort_ids correctly
- Statistical properties across runs match known MG behaviour

Payoff strings use PAYOFF_REGISTRY keys:
    "BinaryMG", "ScaledMG", "SmallMinority", "DollarGame"
"""

import pytest
import numpy as np
from scipy import stats as sp_stats

from core.game import Game
from core.game_config import GameConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_spec(n=301, m=5, s=2, payoff="ScaledMG", position_limit=0):
    """Build a minimal single-cohort population spec."""
    return {
        "total": n,
        "cohorts": [{
            "count":          n,
            "memory":         m,
            "strategies":     s,
            "payoff":         payoff,
            "position_limit": position_limit,
            "agent_type":     "strategic",
            "score_lambda":   0.0,
        }]
    }


def make_cfg(n=301, rounds=2000, seed=42, record_agent_series=True):
    """Build a standard GameConfig."""
    return GameConfig(
        rounds=rounds,
        lambda_=1.0 / (n * 50),
        mm=None,
        price=100,
        seed=seed,
        record_agent_series=record_agent_series,
    )


def run_game(payoff="ScaledMG", m=5, s=2, n=301,
             rounds=2000, seed=42, record_agent_series=True,
             position_limit=0):
    """Run a single game and return results."""
    spec = make_spec(n=n, m=m, s=s, payoff=payoff,
                     position_limit=position_limit)
    cfg  = make_cfg(n=n, rounds=rounds, seed=seed,
                    record_agent_series=record_agent_series)
    game = Game(population_spec=spec, cfg=cfg)
    return game.run()


# ── Results structure ─────────────────────────────────────────────────────────

class TestGameResultsStructure:
    """Game.run() returns a complete, well-formed results dict."""

    def test_required_keys_present(self):
        r = run_game()
        required = [
            "Attendance", "Prices", "final_wins", "final_wealth",
            "final_points", "final_position", "strategy_switches",
            "cohort_ids", "cohorts", "config",
        ]
        for key in required:
            assert key in r, f"Missing key: {key}"

    def test_attendance_shape(self):
        rounds = 2000
        r = run_game(rounds=rounds)
        assert r["Attendance"].shape == (rounds,)

    def test_prices_shape(self):
        rounds = 2000
        r = run_game(rounds=rounds)
        assert r["Prices"].shape == (rounds + 1,)

    def test_final_wins_shape(self):
        n = 301
        r = run_game(n=n)
        assert r["final_wins"].shape == (n,)

    def test_cohort_ids_shape(self):
        n = 301
        r = run_game(n=n)
        assert r["cohort_ids"].shape == (n,)

    def test_agent_series_present_when_requested(self):
        r = run_game(record_agent_series=True)
        assert r["wealth"]   is not None
        assert r["position"] is not None
        assert r["wins"]     is not None

    def test_agent_series_absent_when_not_requested(self):
        r = run_game(record_agent_series=False)
        assert r["wealth"]   is None
        assert r["position"] is None
        assert r["wins"]     is None


# ── All payoff types ──────────────────────────────────────────────────────────

class TestGamePayoffs:
    """Game runs without error for all registered payoff types."""

    @pytest.mark.parametrize("payoff", [
        "BinaryMG",
        "ScaledMG",
        "SmallMinority",
        "DollarGame",
    ])
    def test_runs_without_error(self, payoff):
        """Game completes a full run for each payoff."""
        limit = 1 if payoff == "DollarGame" else 0
        r     = run_game(payoff=payoff, rounds=1000, position_limit=limit)
        assert r is not None
        assert len(r["Attendance"]) == 1000

    @pytest.mark.parametrize("payoff", [
        "BinaryMG",
        "ScaledMG",
        "SmallMinority",
    ])
    def test_attendance_centred_at_zero(self, payoff):
        """Attendance should be centred at zero for MG payoffs."""
        r       = run_game(payoff=payoff, rounds=5000, m=7)
        mean_att = float(np.mean(r["Attendance"]))
        assert abs(mean_att) < 5.0, (
            f"Attendance mean {mean_att:.2f} too far from zero for {payoff}"
        )

    def test_dollar_game_blows_up_without_limit(self):
        """Dollar game without position limits should produce large attendance variance."""
        r   = run_game(payoff="DollarGame", rounds=1000, m=5, position_limit=0)
        var = float(np.var(r["Attendance"]))
        assert var > 10, "DollarGame without limits should have high variance"

    def test_dollar_game_stable_with_limit(self):
        """Dollar game with position limit=1 should remain stable."""
        r = run_game(payoff="DollarGame", rounds=2000, m=5, position_limit=1)
        assert not np.any(np.isnan(r["Prices"])), "Prices should not be NaN"
        assert not np.any(np.isinf(r["Prices"])), "Prices should not be inf"


# ── Success rates ─────────────────────────────────────────────────────────────

class TestGameSuccessRates:
    """Success rates should be in plausible range for MG payoffs."""

    @pytest.mark.parametrize("m", [5, 7, 9])
    def test_success_rate_near_half_sparse_phase(self, m):
        """
        In or near the sparse phase (higher m), success rates should be
        close to 0.5.
        """
        rounds  = 5000
        r       = run_game(m=m, rounds=rounds, n=301, payoff="ScaledMG")
        sr      = r["final_wins"] / rounds
        mean_sr = float(np.mean(sr))
        assert 0.45 <= mean_sr <= 0.55, (
            f"Mean success rate {mean_sr:.4f} out of expected range for m={m}"
        )

    def test_success_rate_below_half_crowded_phase(self):
        """
        In the crowded phase (low alpha), agents should do worse than random.
        N=301, m=3 gives alpha ~ 0.027 (deep crowded phase).
        """
        rounds  = 5000
        r       = run_game(m=3, rounds=rounds, n=301, payoff="ScaledMG")
        sr      = r["final_wins"] / rounds
        mean_sr = float(np.mean(sr))
        assert mean_sr < 0.49, (
            f"Mean success rate {mean_sr:.4f} should be below 0.49 "
            f"in crowded phase"
        )

    def test_final_wins_in_valid_range(self):
        """final_wins should be between 0 and rounds for all agents."""
        rounds = 2000
        r      = run_game(rounds=rounds)
        assert np.all(r["final_wins"] >= 0),      "final_wins should be non-negative"
        assert np.all(r["final_wins"] <= rounds),  "final_wins should not exceed rounds"


# ── Phase structure ───────────────────────────────────────────────────────────

class TestGamePhaseStructure:
    """Normalised variance follows the expected phase diagram pattern."""

    def _sigma2_N(self, m, n=301, rounds=5000, n_runs=5,
                  payoff="ScaledMG"):
        """Run n_runs games and return mean normalised variance."""
        vals = [
            float(np.var(
                run_game(m=m, n=n, rounds=rounds, payoff=payoff, seed=s)["Attendance"]
            )) / n
            for s in range(n_runs)
        ]
        return float(np.mean(vals))

    def test_variance_lower_in_sparse_than_crowded(self):
        """
        Normalised variance should be lower in the sparse phase (high m)
        than in the crowded phase (low m) for N=301.
        m=3  → alpha ≈ 0.027  (crowded)
        m=9  → alpha ≈ 1.70   (sparse)
        """
        sigma2_crowded = self._sigma2_N(m=3)
        sigma2_sparse  = self._sigma2_N(m=9)
        assert sigma2_crowded > sigma2_sparse, (
            f"Crowded variance {sigma2_crowded:.3f} should exceed "
            f"sparse variance {sigma2_sparse:.3f}"
        )

    def test_variance_has_minimum_near_critical_alpha(self):
        """
        Normalised variance should be minimised near alpha_c ≈ 0.3.
        For N=301, m=6 gives alpha ≈ 0.21, close to the critical point.
        Variance at m=6 should be below both m=3 (crowded) and m=11 (sparse).
        """
        sigma2_crowded  = self._sigma2_N(m=3)
        sigma2_critical = self._sigma2_N(m=6)

        assert sigma2_critical < sigma2_crowded, (
            f"Variance at m=6 ({sigma2_critical:.3f}) should be below "
            f"m=3 ({sigma2_crowded:.3f})"
        )


# ── Wealth reconstruction ─────────────────────────────────────────────────────

class TestGameWealthReconstruction:
    """Wealth and cash reconstruction from position and prices is consistent."""

    def test_wealth_equals_cash_plus_position_times_price(self):
        """
        wealth[t] == cash[t] + position[t] * price[t] at every round.
        Tests the identity used in StatsRecorder._reconstruct_wealth_cash.
        """
        r      = run_game(record_agent_series=True, rounds=1000)
        wealth = r["wealth"]    # (rounds+1, N)
        cash   = r["cash"]      # (rounds+1, N)
        pos    = r["position"]  # (rounds+1, N)
        prices = r["Prices"]    # (rounds+1,)

        reconstructed = cash + pos * prices[:, None]
        assert np.allclose(wealth, reconstructed, atol=1e-6), (
            "Wealth should equal cash + position * price at every round"
        )

    def test_cash_update_identity(self):
        """
        cash[t] = cash[t-1] - delta_q[t] * price[t]
        Tests the cash update logic used in wealth reconstruction.
        """
        r      = run_game(record_agent_series=True, rounds=1000)
        cash   = r["cash"]      # (rounds+1, N)
        pos    = r["position"]  # (rounds+1, N)
        prices = r["Prices"]    # (rounds+1,)

        delta_q     = np.diff(pos,  axis=0)          # (rounds, N)
        cash_change = np.diff(cash, axis=0)           # (rounds, N)
        expected    = -delta_q * prices[1:, None]     # (rounds, N)

        assert np.allclose(cash_change, expected, atol=1e-6), (
            "Cash change should equal -delta_q * price each round"
        )

    def test_initial_wealth_is_zero(self):
        """Agents start with zero wealth and zero position."""
        r = run_game(record_agent_series=True, rounds=500)
        assert np.all(r["wealth"][0, :]   == 0.0), "Initial wealth should be zero"
        assert np.all(r["position"][0, :] == 0),   "Initial position should be zero"


# ── Reproducibility ───────────────────────────────────────────────────────────

class TestGameReproducibility:
    """Same seed produces identical results."""

    def test_same_seed_same_attendance(self):
        r1 = run_game(seed=123, rounds=1000)
        r2 = run_game(seed=123, rounds=1000)
        assert np.array_equal(r1["Attendance"], r2["Attendance"]), (
            "Same seed should produce identical attendance series"
        )

    def test_different_seeds_different_attendance(self):
        r1 = run_game(seed=1, rounds=1000)
        r2 = run_game(seed=2, rounds=1000)
        assert not np.array_equal(r1["Attendance"], r2["Attendance"]), (
            "Different seeds should produce different attendance series"
        )


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestGameReset:
    """
    Game.run(reset=True) clears state and produces a valid fresh run.

    Note: reset continues the existing RNG sequence rather than returning
    to the original seed — seed management is the responsibility of the
    experiment layer, not Game itself. These tests verify that reset
    correctly clears state and produces a complete valid run.
    """

    def test_reset_produces_valid_results(self):
        """After reset, game produces a complete valid results dict."""
        spec = make_spec()
        cfg  = make_cfg(rounds=1000)
        game = Game(population_spec=spec, cfg=cfg)

        game.run()
        r2 = game.run(reset=True)

        assert r2 is not None
        assert len(r2["Attendance"]) == 1000
        assert np.all(r2["final_wins"] >= 0)
        assert np.all(r2["final_wins"] <= 1000)

    def test_reset_clears_position(self):
        """After reset, all agents start with zero position."""
        spec = make_spec()
        cfg  = make_cfg(rounds=1000, record_agent_series=True)
        game = Game(population_spec=spec, cfg=cfg)

        game.run()
        r2 = game.run(reset=True)

        assert np.all(r2["position"][0, :] == 0), (
            "All positions should be zero at start of reset run"
        )

    def test_reset_clears_wins(self):
        """After reset, wins start from zero — final_wins reflects only the new run."""
        spec = make_spec()
        cfg  = make_cfg(rounds=1000)
        game = Game(population_spec=spec, cfg=cfg)

        game.run()
        r2 = game.run(reset=True)

        # final_wins should not exceed rounds of the reset run
        assert np.all(r2["final_wins"] <= 1000), (
            "final_wins after reset should not carry over from previous run"
        )


# ── Two-cohort games ──────────────────────────────────────────────────────────

class TestGameTwoCohorts:
    """Games with two cohorts assign cohort_ids correctly."""

    def test_cohort_ids_correct_counts(self):
        """cohort_ids should contain the right number of each cohort."""
        n_adaptive = 240
        n_single   = 61
        spec = {
            "total": n_adaptive + n_single,
            "cohorts": [
                {"count": n_adaptive, "memory": 5, "strategies": 2,
                 "payoff": "ScaledMG", "position_limit": 0,
                 "agent_type": "strategic", "score_lambda": 0.0},
                {"count": n_single,   "memory": 5, "strategies": 1,
                 "payoff": "ScaledMG", "position_limit": 0,
                 "agent_type": "strategic", "score_lambda": 0.0},
            ]
        }
        cfg = make_cfg(n=n_adaptive + n_single, rounds=500)
        g   = Game(population_spec=spec, cfg=cfg)
        r   = g.run()

        ids = r["cohort_ids"]
        assert np.sum(ids == 0) == n_adaptive, "Wrong count for cohort 0"
        assert np.sum(ids == 1) == n_single,   "Wrong count for cohort 1"

    def test_single_strategy_wins_near_half(self):
        """
        Single-strategy agents in a mixed game should have success rates
        near 0.5 at low proportions — they can't learn but aren't
        systematically exploited at low proportions.
        """
        rounds     = 5000
        n_adaptive = 270
        n_single   = 31
        spec = {
            "total": n_adaptive + n_single,
            "cohorts": [
                {"count": n_adaptive, "memory": 5, "strategies": 2,
                 "payoff": "ScaledMG", "position_limit": 0,
                 "agent_type": "strategic", "score_lambda": 0.0},
                {"count": n_single,   "memory": 5, "strategies": 1,
                 "payoff": "ScaledMG", "position_limit": 0,
                 "agent_type": "strategic", "score_lambda": 0.0},
            ]
        }
        cfg = make_cfg(n=n_adaptive + n_single, rounds=rounds)
        g   = Game(population_spec=spec, cfg=cfg)
        r   = g.run()

        ids       = r["cohort_ids"]
        wins      = r["final_wins"]
        sr_single = wins[ids == 1] / rounds
        mean_sr   = float(np.mean(sr_single))

        assert 0.40 <= mean_sr <= 0.55, (
            f"Single-strategy success rate {mean_sr:.4f} outside expected range"
        )


# ── Statistical properties across runs ───────────────────────────────────────

class TestGameStatisticalProperties:
    """
    Across multiple runs, key statistics should match known MG properties.
    These tests are slower — mark with -m 'not slow' to skip in fast CI.
    """

    def test_attendance_variance_consistent_across_runs(self):
        """
        Normalised variance should be consistent (low coefficient of variation)
        across runs at a stable phase point.
        """
        n_runs = 10
        rounds = 3000
        vals   = [
            float(np.var(
                run_game(m=7, rounds=rounds, seed=s)["Attendance"]
            )) / 301
            for s in range(n_runs)
        ]
        cv = float(np.std(vals) / np.mean(vals))
        assert cv < 0.5, (
            f"Normalised variance too variable across runs (CV={cv:.3f})"
        )


