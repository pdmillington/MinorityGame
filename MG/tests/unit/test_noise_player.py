#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:00:23 2026

@author: petermillington
"""

import pytest
import numpy as np
from core.noise_player import NoisePlayer


class TestNoisePlayer:
    """Test noise trader functionality"""
    
    def test_random_actions(self):
        """Test that noise player produces random actions"""
        rng = np.random.default_rng(42)
        player = NoisePlayer(rng=rng)
        
        actions = [player.choose_action([1, -1]) for _ in range(100)]
        
        # Should have both +1 and -1
        assert 1 in actions, "Should produce +1 actions"
        assert -1 in actions, "Should produce -1 actions"
        
        # Should be roughly balanced (within reason for 100 samples)
        count_plus = sum(1 for a in actions if a == 1)
        count_minus = sum(1 for a in actions if a == -1)
        assert 30 < count_plus < 70, "Should be roughly 50/50 split"
        assert 30 < count_minus < 70, "Should be roughly 50/50 split"
    
    def test_no_action_option(self):
        """Test noise player with allow_no_action=True"""
        rng = np.random.default_rng(42)
        player = NoisePlayer(allow_no_action=True, rng=rng)
        
        actions = [player.choose_action([1]) for _ in range(100)]
        
        # Should have -1, 0, and +1
        assert -1 in actions
        assert 0 in actions
        assert 1 in actions
    
    def test_position_limit_respected(self):
        """Test that noise player respects position limits"""
        rng = np.random.default_rng(42)
        player = NoisePlayer(position_limit=5, rng=rng)
        
        player.position = 5
        actions = [player.choose_action([1]) for _ in range(20)]
        
        # Should not produce +1 when at limit
        assert 1 not in actions, "Should not buy when at position limit"
        assert -1 in actions or 0 in actions, "Should be able to sell or do nothing"
    
    def test_wins_tracked(self):
        """Test that wins are tracked correctly"""
        rng = np.random.default_rng(42)
        player = NoisePlayer(rng=rng)
        
        initial_wins = player.wins
        
        # Choose action
        player._action = -1
        
        # Update with flow indicating minority was -1
        player.update(N=100, flow=10, price=100.0, lambda_value=0.01)
        
        # Should have incremented wins
        assert player.wins == initial_wins + 1, "Should track wins correctly"
    
    def test_wealth_updated(self):
        """Test that wealth is updated on trades"""
        rng = np.random.default_rng(42)
        player = NoisePlayer(initial_cash=1000.0, rng=rng)
        
        player._action = 1  # Buy
        player.update(N=100, flow=5, price=100.0, lambda_value=0.01)
        
        assert player.position == 1
        assert player.cash == 900.0
        assert player.wealth == 1000.0  # 1*100 + 900