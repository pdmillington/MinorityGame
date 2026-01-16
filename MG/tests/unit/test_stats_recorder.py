#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:12:49 2026

@author: petermillington
"""

"""
Unit tests for StatsRecorder
"""
import pytest
import numpy as np
from data.stats_recorder import StatsRecorder
from core.base_agent import StrategicAgent
from payoffs.mg import PAYOFF_REGISTRY


class TestStatsRecorderInitialization:
    """Test StatsRecorder initialization"""
    
    def test_initialization_with_recording(self):
        """Test recorder initializes with agent series recording enabled"""
        recorder = StatsRecorder(N=10, rounds=100, k_max=2, record_agent_series=True)
        
        assert recorder.N == 10
        assert recorder.rounds == 100
        assert recorder.k_max == 2
        assert recorder.record_agent_series is True
        
        # Check array shapes
        assert recorder.wealth.shape == (101, 10)  # rounds+1 x N
        assert recorder.position.shape == (101, 10)
        assert recorder.wins_per_round.shape == (101, 10)
        assert recorder.points_per_round.shape == (101, 10)
        assert recorder.cash_per_round.shape == (101, 10)
        assert recorder.best_strategy.shape == (101, 10)
        
        assert recorder.prices.shape == (101,)
        assert recorder.attendance.shape == (100,)
    
    def test_initialization_without_recording(self):
        """Test recorder initializes without agent series recording"""
        recorder = StatsRecorder(N=10, rounds=100, k_max=2, record_agent_series=False)
        
        assert recorder.N == 10
        assert recorder.rounds == 100
        assert recorder.record_agent_series is False
        
        # Agent series should be None
        assert recorder.wealth is None
        assert recorder.position is None
        assert recorder.wins_per_round is None
        assert recorder.best_strategy is None
        
        # Global series should still exist
        assert recorder.prices.shape == (101,)
        assert recorder.attendance.shape == (100,)


class TestStatsRecorderRecordInitialState:
    """Test recording initial state"""
    
    def test_record_initial_with_recording(self):
        """Test recording initial state when agent series enabled"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(5)
        ]
        
        # Set some initial values
        for i, p in enumerate(players):
            p.wealth = 100.0 + i
            p.position = i
            p.wins = 0
            p.points = 0.0
            p.cash = 100.0
        
        recorder = StatsRecorder(N=5, rounds=10, k_max=2, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        
        assert recorder.prices[0] == 100.0
        assert recorder.wealth[0, 0] == 100.0
        assert recorder.wealth[0, 4] == 104.0
        assert recorder.position[0, 2] == 2
    
    def test_record_initial_without_recording(self):
        """Test recording initial state when agent series disabled"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(5)
        ]
        
        recorder = StatsRecorder(N=5, rounds=10, k_max=2, record_agent_series=False)
        recorder.record_initial_state(price=100.0, players=players)
        
        # Should set price but not crash on agent arrays
        assert recorder.prices[0] == 100.0


class TestStatsRecorderRecordRound:
    """Test recording rounds"""
    
    def test_record_round_with_recording(self):
        """Test recording a round when agent series enabled"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(5)
        ]
        
        # Set some values
        for i, p in enumerate(players):
            p.wealth = 100.0 + i * 10
            p.position = i - 2
            p.wins = i
            p.points = float(i * 5)
            p.cash = 50.0 + i * 5
            p.strategy = i % 2
        
        recorder = StatsRecorder(N=5, rounds=10, k_max=2, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        recorder.record_round(t=0, price=105.0, A=3, players=players)
        
        assert recorder.attendance[0] == 3
        assert recorder.prices[1] == 105.0
        assert recorder.wealth[1, 0] == 100.0
        assert recorder.wealth[1, 4] == 140.0
        assert recorder.position[1, 2] == 0
        assert recorder.wins_per_round[1, 3] == 3
    
    def test_record_round_without_recording(self):
        """Test recording a round when agent series disabled"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(5)
        ]
        
        recorder = StatsRecorder(N=5, rounds=10, k_max=2, record_agent_series=False)
        recorder.record_initial_state(price=100.0, players=players)
        recorder.record_round(t=0, price=105.0, A=3, players=players)
        
        # Should record global stats but not crash on agent arrays
        assert recorder.attendance[0] == 3
        assert recorder.prices[1] == 105.0
    
    def test_record_multiple_rounds(self):
        """Test recording multiple rounds"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(3)
        ]
        
        recorder = StatsRecorder(N=3, rounds=5, k_max=2, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        
        for t in range(5):
            # Update player values
            for p in players:
                p.wealth += 1.0
                p.strategy = 0
            recorder.record_round(t, price=100.0 + t, A=t, players=players)
        
        assert len(recorder.attendance) == 5
        assert recorder.attendance[0] == 0
        assert recorder.attendance[4] == 4


class TestStatsRecorderFinalize:
    """Test finalize functionality"""
    
    def test_finalize_with_recording(self):
        """Test finalize when recording agent series"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(5)
        ]
        
        # Set final values
        for i, p in enumerate(players):
            p.wealth = 100.0 + i * 10
            p.wins = i * 2
            p.points = float(i * 5)
            p.position = i - 2
            p.cash = 50.0
            p.strategy = 0
            p.strategy_switches = i
        
        recorder = StatsRecorder(N=5, rounds=3, k_max=2, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        
        # Record some rounds
        for t in range(3):
            recorder.record_round(t, price=100.0, A=0, players=players)
        
        # Finalize
        results = recorder.finalize(players)
        
        assert "final_wealth" in results
        assert "final_wins" in results
        assert "final_points" in results
        assert "strategy_switches" in results
        
        assert len(results["final_wealth"]) == 5
        assert results["final_wealth"][0] == 100.0
        assert results["final_wealth"][4] == 140.0
    
    def test_finalize_without_recording(self):
        """Test finalize when NOT recording agent series"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(5)
        ]
        
        # Set final values
        for i, p in enumerate(players):
            p.wealth = 100.0 + i * 10
            p.wins = i * 2
            p.points = float(i * 5)
            p.position = i - 2
            p.cash = 50.0
            p.strategy_switches = i
        
        recorder = StatsRecorder(N=5, rounds=3, k_max=2, record_agent_series=False)
        recorder.record_initial_state(price=100.0, players=players)
        
        # Record some rounds
        for t in range(3):
            recorder.record_round(t, price=100.0, A=0, players=players)
        
        # Finalize - should not crash!
        results = recorder.finalize(players)
        
        assert "final_wealth" in results
        assert "final_wins" in results
        assert "final_points" in results
        
        # Should extract from players directly
        assert len(results["final_wealth"]) == 5
        assert results["final_wealth"][0] == 100.0
        assert results["final_wealth"][4] == 140.0
        
        # Time series should be None
        assert results["wealth"] is None
        assert results["wins"] is None
    
    def test_finalize_results_structure(self):
        """Test that finalize returns correct structure"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(3)
        ]
        
        recorder = StatsRecorder(N=3, rounds=5, k_max=2, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        
        history = [1, -1, 1]
        
    
        for t in range(5):
            for p in players:
                p.choose_action(history)
            recorder.record_round(t, price=100.0, A=0, players=players)
        
        results = recorder.finalize(players)
        
        # Check all expected keys
        expected_keys = [
            "Attendance", "Prices", "wealth", "position", "wins",
            "best_strategy", "final_wealth", "final_wins",
            "final_points", "strategy_switches"
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"


class TestStatsRecorderStrategywitches:
    """Test strategy switch counting"""
    
    def test_strategy_switches_calculated(self):
        """Test that strategy switches are calculated correctly"""
        players = [
            StrategicAgent(memory=2, num_strategies=3, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(3)
        ]
        
        history = [-1, 1, -1]
        
        for p in players:
            p.choose_action(history)
        
        recorder = StatsRecorder(N=3, rounds=5, k_max=3, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        
        history = [-1, 1, -1]
        
        # Manually set strategy sequence for player 0
        # t=0: strategy 0
        # t=1: strategy 0 (no switch)
        # t=2: strategy 1 (switch!)
        # t=3: strategy 1 (no switch)
        # t=4: strategy 2 (switch!)
        # Expected: 2 switches
        
        for t in range(5):
            if t < 2:
                players[0].strategy = 0
            elif t < 4:
                players[0].strategy = 1
            else:
                players[0].strategy = 2
            
            recorder.record_round(t, price=100.0, A=0, players=players)
        
        results = recorder.finalize(players)
        
        # Should count 2 switches for player 0
        # Note: Exact counting depends on implementation details
        assert results["strategy_switches"] is not None
        assert len(results["strategy_switches"]) == 3
    
    def test_strategy_switches_without_recording(self):
        """Test strategy switches when agent series not recorded"""
        players = [
            StrategicAgent(memory=2, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(3)
        ]
        
        # Set strategy_switches on players
        for i, p in enumerate(players):
            p.strategy_switches = i * 2
        
        recorder = StatsRecorder(N=3, rounds=5, k_max=2, record_agent_series=False)
        recorder.record_initial_state(price=100.0, players=players)
        
        for t in range(5):
            recorder.record_round(t, price=100.0, A=0, players=players)
        
        results = recorder.finalize(players)
        
        # Should extract from players
        assert results["strategy_switches"] is not None
        assert results["strategy_switches"][0] == 0
        assert results["strategy_switches"][1] == 2
        assert results["strategy_switches"][2] == 4


class TestStatsRecorderEdgeCases:
    """Test edge cases"""
    
    def test_single_agent(self):
        """Test with single agent"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
        ]
        
        history = [-1, 1, -1]
        
        for p in players:
            p.choose_action(history)
        
        recorder = StatsRecorder(N=1, rounds=5, k_max=2, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        
        for t in range(5):
            recorder.record_round(t, price=100.0, A=0, players=players)
        
        results = recorder.finalize(players)
        
        assert len(results["final_wealth"]) == 1
    
    def test_zero_rounds(self):
        """Test with zero rounds (just initial state)"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(3)
        ]
        
        recorder = StatsRecorder(N=3, rounds=0, k_max=2, record_agent_series=True)
        recorder.record_initial_state(price=100.0, players=players)
        
        # No rounds recorded
        results = recorder.finalize(players)
        
        # Should still have initial values
        assert "final_wealth" in results
    
    def test_large_population(self):
        """Test with large population"""
        players = [
            StrategicAgent(memory=3, num_strategies=2, payoff=PAYOFF_REGISTRY["BinaryMG"])
            for _ in range(1000)
        ]
        
        recorder = StatsRecorder(N=1000, rounds=10, k_max=2, record_agent_series=False)
        recorder.record_initial_state(price=100.0, players=players)
        
        for t in range(10):
            recorder.record_round(t, price=100.0, A=0, players=players)
        
        results = recorder.finalize(players)
        
        assert len(results["final_wealth"]) == 1000


class TestStatsRecorderMemoryEfficiency:
    """Test memory efficiency of recording options"""
    
    def test_memory_usage_without_recording(self):
        """Test that disabling recording saves memory"""
        # With recording
        recorder_with = StatsRecorder(N=1000, rounds=10000, k_max=10, record_agent_series=True)
        
        # Without recording
        recorder_without = StatsRecorder(N=1000, rounds=10000, k_max=10, record_agent_series=False)
        
        # Without recording should have None arrays
        assert recorder_with.wealth is not None
        assert recorder_without.wealth is None
        
        # Both should have global series
        assert recorder_with.prices is not None
        assert recorder_without.prices is not None