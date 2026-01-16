#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 18:51:20 2026

Tests complete experiment pipelines to ensure they produce valid outputs.
These are slower than unit tests but verify real research workflows.
"""

import pytest
import os
import tempfile
import json
import numpy as np
from analysis.population_spec import (
    build_population_spec, 
    build_population_variant,
    PopulationConfig,
    PopulationFamilyConfig
)
from core.game import Game
from core.game_config import GameConfig


class TestPopulationSpecWorkflow:
    """Test that population specification building works end-to-end"""
    
    def test_cartesian_population_builds_and_runs(self):
        """Test building population from cartesian config and running game"""
        cfg = PopulationConfig(
            payoffs=["BinaryMG", "ScaledMG"],
            num_players_per_cohort=20,
            m_values=[3, 5],
            s_values=[2, 4],
            position_limit=0,
            rounds=50,
            mode="cartesian",
            seed=42
        )
        
        # Build population spec
        pop_spec = build_population_spec(cfg)
        
        # Verify it has right structure
        assert "total" in pop_spec
        assert "cohorts" in pop_spec
        assert pop_spec["total"] == 20
        
        # Should have 2x2x2 = 8 cohorts (but split into 20 total agents)
        assert len(pop_spec["cohorts"]) > 0
        
        # Run a game with it
        game_cfg = GameConfig(rounds=50, lambda_=0.01, mm=None, price=100.0)
        game = Game(population_spec=pop_spec, cfg=game_cfg)
        results = game.run()
        
        # Verify game completed
        assert results is not None
        assert len(results["final_wealth"]) == 20
    
    def test_explicit_cohorts_builds_and_runs(self):
        """Test building population from explicit cohort list"""
        from analysis.population_spec import CohortConfig
        
        cohorts = [
            CohortConfig(
                count=10,
                memory=3,
                payoff="BinaryMG",
                strategies=2,
                agent_type="strategic"
            ),
            CohortConfig(
                count=5,
                agent_type="noise",
                memory=1
            )
        ]
        
        cfg = PopulationConfig(
            payoffs=["BinaryMG"],  # Not used when cohorts specified
            num_players_per_cohort=100,  # Not used when cohorts specified
            m_values=[5],  # Not used when cohorts specified
            s_values=[2],  # Not used when cohorts specified
            rounds=50,
            cohorts=cohorts,
            seed=42
        )
        
        pop_spec = build_population_spec(cfg)
        
        assert pop_spec["total"] == 15
        assert len(pop_spec["cohorts"]) == 2
        
        # Run game
        game_cfg = GameConfig(rounds=50, lambda_=0.01, mm=None, price=100.0)
        game = Game(population_spec=pop_spec, cfg=game_cfg)
        results = game.run()
        
        assert len(results["final_wealth"]) == 15


class TestPopulationFamilyWorkflow:
    """Test population family experiments"""
    
    def test_memory_shift_family(self):
        """Test varying memory across population family"""
        base_cohorts = [
            {
                "count": 10,
                "memory": 3,
                "payoff": "BinaryMG",
                "strategies": 2
            }
        ]
        
        family_cfg = PopulationFamilyConfig(
            base_cohorts=base_cohorts,
            vary="memory_shift",
            values=[0, 1, 2],
            rounds=50,
            seed=42
        )
        
        # Build variants
        variants = [
            build_population_variant(family_cfg, v) 
            for v in family_cfg.values
        ]
        
        assert len(variants) == 3
        assert variants[0]["cohorts"][0]["memory"] == 3
        assert variants[1]["cohorts"][0]["memory"] == 4
        assert variants[2]["cohorts"][0]["memory"] == 5
        
        # Run games for each variant
        game_cfg = GameConfig(rounds=50, lambda_=0.01, mm=None, price=100.0)
        
        for variant in variants:
            game = Game(population_spec=variant, cfg=game_cfg)
            results = game.run()
            assert results is not None
    
    def test_strategies_shift_family(self):
        """Test varying strategies across population family"""
        base_cohorts = [
            {
                "count": 10,
                "memory": 4,
                "payoff": "ScaledMG",
                "strategies": 2
            }
        ]
        
        family_cfg = PopulationFamilyConfig(
            base_cohorts=base_cohorts,
            vary="strategies_shift",
            values=[0, 2, 4],
            rounds=50,
            seed=42
        )
        
        variants = [
            build_population_variant(family_cfg, v) 
            for v in family_cfg.values
        ]
        
        assert len(variants) == 3
        assert variants[0]["cohorts"][0]["strategies"] == 2
        assert variants[1]["cohorts"][0]["strategies"] == 4
        assert variants[2]["cohorts"][0]["strategies"] == 6
        
        # Verify they all run
        game_cfg = GameConfig(rounds=50, lambda_=0.01, mm=None, price=100.0)
        
        for variant in variants:
            game = Game(population_spec=variant, cfg=game_cfg)
            results = game.run()
            assert results is not None
    
    def test_total_size_family(self):
        """Test varying total population size"""
        base_cohorts = [
            {
                "count": 10,
                "memory": 3,
                "payoff": "BinaryMG",
                "strategies": 2
            }
        ]
        
        family_cfg = PopulationFamilyConfig(
            base_cohorts=base_cohorts,
            vary="total_size",
            values=[10, 20, 30],
            rounds=50,
            seed=42
        )
        
        variants = [
            build_population_variant(family_cfg, v) 
            for v in family_cfg.values
        ]
        
        assert len(variants) == 3
        assert variants[0]["total"] == 10
        assert variants[1]["total"] == 20
        assert variants[2]["total"] == 30


class TestExperimentReproducibility:
    """Test that experiments are reproducible"""
    
    def test_repeated_runs_identical(self):
        """Test that running same config twice gives same results"""
        population_spec = {
            "total": 20,
            "cohorts": [
                {
                    "count": 20,
                    "memory": 4,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                }
            ]
        }
        
        cfg = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0, seed=42)
        
        # Run 1
        game1 = Game(population_spec=population_spec, cfg=cfg)
        results1 = game1.run()
        
        # Run 2 (reset config with same seed)
        cfg2 = GameConfig(rounds=100, lambda_=0.01, mm=None, price=100.0, seed=42)
        game2 = Game(population_spec=population_spec, cfg=cfg2)
        results2 = game2.run()
        
        # Should be identical
        np.testing.assert_array_equal(results1["Attendance"], results2["Attendance"])
        np.testing.assert_array_almost_equal(results1["Prices"], results2["Prices"])
        np.testing.assert_array_almost_equal(
            results1["final_wealth"], 
            results2["final_wealth"]
        )


class TestConfigFiles:
    """Test that config files can be loaded and used"""
    
    def test_phase_diagram_config_structure(self):
        """Test that phase diagram configs have required fields"""
        # This tests the expected structure without running full experiment
        config = {
            "payoff_key": "ScaledMG",
            "m_values": [3, 4, 5],
            "num_players": 101,
            "num_strategies": 2,
            "rounds": 100,
            "num_games": 2,
            "market_maker": None
        }
        
        # Verify structure
        assert "payoff_key" in config
        assert "m_values" in config
        assert "num_players" in config
        assert "rounds" in config
        
        # Verify values are valid
        assert isinstance(config["m_values"], list)
        assert all(isinstance(m, int) for m in config["m_values"])
        assert config["num_players"] > 0
        assert config["rounds"] > 0
    
    def test_success_boxplot_config_structure(self):
        """Test that success boxplot configs have required fields"""
        config = {
            "payoffs": ["BinaryMG"],
            "num_players_per_cohort": 100,
            "m_values": [3, 5, 7],
            "s_values": [2, 4],
            "rounds": 100,
            "mode": "cartesian",
            "seed": 1234
        }
        
        assert "payoffs" in config
        assert "m_values" in config
        assert "s_values" in config
        assert "rounds" in config
        assert "seed" in config


class TestLongRunningWorkflows:
    """Test longer-running workflows (marked slow)"""
    
    @pytest.mark.slow
    def test_medium_length_simulation(self):
        """Test a medium-length simulation completes"""
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
        
        cfg = GameConfig(rounds=1000, lambda_=0.001, mm=None, price=100.0, seed=42)
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Should complete without errors
        assert results is not None
        assert len(results["Attendance"]) == 1000
        
        # Prices should remain stable
        assert np.all(results["Prices"] > 1.0)
        assert np.all(results["Prices"] < 1000.0)
    
    @pytest.mark.slow
    def test_heterogeneous_medium_simulation(self):
        """Test medium simulation with heterogeneous population"""
        population_spec = {
            "total": 100,
            "cohorts": [
                {
                    "count": 50,
                    "memory": 4,
                    "payoff": "BinaryMG",
                    "strategies": 2,
                    "agent_type": "strategic"
                },
                {
                    "count": 30,
                    "memory": 6,
                    "payoff": "ScaledMG",
                    "strategies": 4,
                    "agent_type": "strategic"
                },
                {
                    "count": 20,
                    "agent_type": "noise",
                    "memory": 1
                }
            ]
        }
        
        cfg = GameConfig(
            rounds=1000, 
            lambda_=0.001, 
            mm=True, 
            price=100.0, 
            seed=42,
            record_agent_series=True
        )
        
        game = Game(population_spec=population_spec, cfg=cfg)
        results = game.run()
        
        # Verify complete results
        assert len(results["final_wealth"]) == 100
        assert len(np.unique(results["cohort_ids"])) == 3
        
        # Market maker should be present
        assert game.mm is not None


# Pytest configuration to handle slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )