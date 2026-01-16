#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 18:42:46 2026

@author: petermillington

Integration tests for complete game workflows

These tests verify that components work together correctly for real simulations.
They test end-to-end workflows rather than individual components.
"""

import pytest
import numpy as np
from core.game import Game
from core.game_config import GameConfig
from payoffs.mg import PAYOFF_REGISTRY


class TestBasicGameWorkflow:
    """Test that a complete game can run from start to finish"""

    def test_simple_game_completes(self):
        """Test that a basic game runs without crashing"""
        population_spec = {
            "total": 10,
            "cohorts": [
                {
                    "count": 10,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Basic sanity checks
        assert results is not None
        assert "Attendance" in results
        assert "Prices" in results
        assert len(results["Attendance"]) == 100
        assert len(results["Prices"]) == 101  # T+1 including initial
    
    def test_game_with_market_maker(self):
        """Test that game works with market maker enabled"""
        population_spec = {
            "total": 50,
            "cohorts": [
                {
                    "count": 50,
                    "memory": 4,
                    "payoff": "ScaledMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=100, lambda_=0.01, mm=True, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Verify market maker tracked
        assert game.mm is not None
        assert len(game.mm.position_history) == 100
        assert len(game.mm.wealth_history) == 100
    
    def test_game_produces_valid_statistics(self):
        """Test that game produces valid final statistics"""
        population_spec = {
            "total": 20,
            "cohorts": [
                {
                    "count": 20,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Check final statistics exist and are valid
        assert "final_wealth" in results
        assert "final_wins" in results
        assert "final_points" in results
        
        assert len(results["final_wealth"]) == 20
        assert len(results["final_wins"]) == 20
        assert all(isinstance(w, (int, float, np.integer, np.floating)) 
                   for w in results["final_wealth"])
        assert all(isinstance(w, (int, np.integer)) 
                   for w in results["final_wins"])


class TestHeterogeneousPopulations:
    """Test games with mixed agent types"""
    
    def test_multiple_cohorts_strategic(self):
        """Test game with multiple strategic cohorts"""
        population_spec = {
            "total": 30,
            "cohorts": [
                {
                    "count": 10,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                },
                {
                    "count": 10,
                    "memory": 5,
                    "payoff": "ScaledMG",
                    "strategies": 4,
                    "agent_type": "strategic"
                },
                {
                    "count": 10,
                    "memory": 7,
                    "payoff": "BinaryMG",
                    "strategies": 6,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Verify cohort tracking
        assert "cohort_ids" in results
        assert len(results["cohort_ids"]) == 30
        assert len(np.unique(results["cohort_ids"])) == 3  # 3 cohorts
    
    def test_strategic_and_noise_traders(self):
        """Test game with both strategic and noise traders"""
        population_spec = {
            "total": 30,
            "cohorts": [
                {
                    "count": 20,
                    "memory": 4,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                },
                {
                    "count": 10,
                    "agent_type": "noise",
                    "memory": 1
                }
            ]
        }
        
        cfg = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Verify mixed population works
        assert len(game.players) == 30
        assert results["final_wealth"] is not None
        assert len(results["final_wealth"]) == 30


class TestPositionLimits:
    """Test that position limits are enforced during games"""
    
    def test_position_limits_enforced(self):
        """Test that agents respect position limits"""
        population_spec = {
            "total": 10,
            "cohorts": [
                {
                    "count": 10,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic",
                    "position_limit": 5
                }
            ]
        }
        
        cfg = GameConfig(
            rounds=100, 
            lambda_=0.01, 
            mm=None, 
            price=100.0,
            record_agent_series=True
        )
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Check that no agent exceeded position limit
        if results.get("position") is not None:
            max_positions = np.max(np.abs(results["position"]), axis=0)
            assert all(max_positions <= 5), "Some agent exceeded position limit"


class TestPriceStability:
    """Test that prices behave reasonably"""
    
    def test_prices_dont_explode_binary(self):
        """Test that prices remain stable with BinaryMG payoff"""
        population_spec = {
            "total": 100,
            "cohorts": [
                {
                    "count": 100,
                    "memory": 5,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=500, lambda_=0.001, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        prices = results["Prices"]
        
        # Prices should stay in reasonable range
        assert np.all(prices > 1.0), "Price went too low"
        assert np.all(prices < 1000.0), "Price exploded"
        assert not np.any(np.isnan(prices)), "Price became NaN"
        assert not np.any(np.isinf(prices)), "Price became infinite"
    
    def test_prices_dont_explode_scaled(self):
        """Test that prices remain stable with ScaledMG payoff"""
        population_spec = {
            "total": 100,
            "cohorts": [
                {
                    "count": 100,
                    "memory": 5,
                    "payoff": "ScaledMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=500, lambda_=0.001, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        prices = results["Prices"]
        
        assert np.all(prices > 1.0), "Price went too low"
        assert np.all(prices < 1000.0), "Price exploded"


class TestPayoffSchemes:
    """Test that different payoff schemes produce valid results"""
    
    @pytest.mark.parametrize("payoff_name", ["BinaryMG", "ScaledMG", "DollarGame"])
    def test_payoff_scheme_completes(self, payoff_name):
        """Test that each payoff scheme can run a complete game"""
        population_spec = {
            "total": 20,
            "cohorts": [
                {
                    "count": 20,
                    "memory": 4,
                    "payoff": payoff_name,
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Basic sanity checks
        assert results is not None
        assert len(results["Attendance"]) == 100
        assert len(results["final_wealth"]) == 20
        
        # Check that points were accumulated
        assert "final_points" in results
        # For BinaryMG/ScaledMG, some agents should have positive points
        if payoff_name in ["BinaryMG", "ScaledMG"]:
            assert np.any(np.array(results["final_points"]) != 0), \
                f"{payoff_name} should produce non-zero points"


class TestReproducibility:
    """Test that games with same seed produce same results"""
    
    def test_seed_reproducibility(self):
        """Test that same seed produces identical results"""
        population_spec = {
            "total": 20,
            "cohorts": [
                {
                    "count": 20,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg1 = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0, seed=42)
        cfg2 = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0, seed=42)
        
        game1 = Game(population_spec=population_spec, cfg=cfg1)
        results1 = game1.run()
        
        game2 = Game(population_spec=population_spec, cfg=cfg2)
        results2 = game2.run()
        
        # Should produce identical results
        np.testing.assert_array_equal(
            results1["Attendance"], 
            results2["Attendance"],
            err_msg="Same seed should produce identical attendance"
        )
        np.testing.assert_array_almost_equal(
            results1["Prices"], 
            results2["Prices"],
            err_msg="Same seed should produce identical prices"
        )
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different results"""
        population_spec = {
            "total": 20,
            "cohorts": [
                {
                    "count": 20,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg1 = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0, seed=42)
        cfg2 = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0, seed=99)
        
        game1 = Game(population_spec=population_spec, cfg=cfg1)
        results1 = game1.run()
        
        game2 = Game(population_spec=population_spec, cfg=cfg2)
        results2 = game2.run()
        
        # Should produce different results
        assert not np.array_equal(results1["Attendance"], results2["Attendance"]), \
            "Different seeds should produce different results"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_very_short_game(self):
        """Test game with very few rounds"""
        population_spec = {
            "total": 10,
            "cohorts": [
                {
                    "count": 10,
                    "memory": 2,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=5, lambda_=0.01, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        assert len(results["Attendance"]) == 5
        assert len(results["Prices"]) == 6
    
    def test_small_population(self):
        """Test game with very small population"""
        population_spec = {
            "total": 3,
            "cohorts": [
                {
                    "count": 3,
                    "memory": 2,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=50, lambda_=0.01, mm=None, price=100.0)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        assert len(game.players) == 3
        assert results is not None


class TestStatisticsRecording:
    """Test that statistics are recorded correctly during games"""
    
    def test_agent_series_recorded(self):
        """Test that per-agent time series are recorded when enabled"""
        population_spec = {
            "total": 10,
            "cohorts": [
                {
                    "count": 10,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(
            rounds=50, 
            lambda_=0.01, 
            mm=None, 
            price=100.0,
            record_agent_series=True
        )
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Check that time series were recorded
        assert "wealth" in results
        assert results["wealth"] is not None
        assert results["wealth"].shape == (51, 10)  # rounds+1 x N
    
    def test_agent_series_not_recorded_when_disabled(self):
        """Test that per-agent series are NOT recorded when disabled"""
        population_spec = {
            "total": 10,
            "cohorts": [
                {
                    "count": 10,
                    "memory": 3,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(
            rounds=50, 
            lambda_=0.01, 
            mm=None, 
            price=100.0,
            record_agent_series=False
        )
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Time series should be None
        assert results["wealth"] is None
        
        # But final stats should still exist
        assert "final_wealth" in results
        assert results["final_wealth"] is not None
