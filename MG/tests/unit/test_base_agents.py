# -*- coding: utf-8 -*-

import pytest
import numpy as np
from core.base_agent import BaseAgent, StrategicAgent
from payoffs.mg import PAYOFF_REGISTRY


class TestBaseAgent:
    """Test abstract base agent functionality"""
    
    def test_position_limit_enforcement(self):
        """Test that position limits are enforced correctly"""
        # Create a mock agent with position limit
        agent = StrategicAgent(
            memory=3, 
            num_strategies=2, 
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            position_limit=5
        )
        
        # Test that action within limit is allowed
        agent.position = 3
        action = agent._enforce_position_limit(1)  # Would result in position 4
        assert action == 1, "Action within limit should be allowed"
        
        # Test that action exceeding limit is blocked
        agent.position = 5
        action = agent._enforce_position_limit(1)  # Would result in position 6
        assert action == 0, "Action exceeding limit should be blocked"
        
        # Test negative position
        agent.position = -4
        action = agent._enforce_position_limit(-1)  # Would result in position -5
        assert action == -1, "Negative position within limit should be allowed"
        
        agent.position = -5
        action = agent._enforce_position_limit(-1)  # Would result in position -6
        assert action == 0, "Negative action exceeding limit should be blocked"
    
    def test_no_position_limit(self):
        """Test agents with no position limit"""
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            position_limit=None  # No limit
        )
        
        agent.position = 1000
        action = agent._enforce_position_limit(1)
        assert action == 1, "No limit should allow any position"
    
    def test_apply_trade(self):
        """Test that trades update position, cash, and wealth correctly"""
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            initial_cash=1000.0
        )
        
        # Buy 1 unit at price 100
        agent._apply_trade(action=1, price=100.0)
        assert agent.position == 1
        assert agent.cash == 900.0  # 1000 - 100
        assert agent.wealth == 1000.0  # 1*100 + 900
        
        # Sell 2 units at price 110
        agent._apply_trade(action=-2, price=110.0)
        assert agent.position == -1  # 1 - 2
        assert agent.cash == 1120.0  # 900 + 2*110
        assert agent.wealth == 1010.0  # -1*110 + 1120
    
    def test_strategy_selection(self):
        """Test that best strategy is selected correctly"""
        agent = StrategicAgent(
            memory=2,
            num_strategies=3,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            rng=np.random.default_rng(42)
        )
        
        # Set scores manually
        agent.scores = np.array([1.0, 5.0, 3.0])
        
        # Choose action (should pick strategy 1 with highest score)
        history = [1, -1]
        action = agent.choose_action(history)
        
        assert agent.strategy == 1, "Should select strategy with highest score"
        assert action in [-1, 1], "Action should be valid"
    
    def test_strategy_switches_counted(self):
        """Test that strategy switches are tracked"""
        agent = StrategicAgent(
            memory=2,
            num_strategies=3,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            rng=np.random.default_rng(42)
        )
        
        initial_switches = agent.strategy_switches
        
        # First choice
        agent.scores = np.array([5.0, 1.0, 1.0])
        agent.choose_action([1, -1])
        first_strategy = agent.strategy
        switches_after_first = agent.strategy_switches
        
        # Second choice with same best strategy
        agent.choose_action([1, 1])
        assert agent.strategy_switches == switches_after_first, "No switch if same strategy"
        
        # Third choice with different best strategy
        agent.scores = np.array([1.0, 5.0, 1.0])
        agent.choose_action([-1, 1])
        assert agent.strategy != first_strategy, "Strategy should change"
        assert agent.strategy_switches == switches_after_first + 1, "Switch should be counted"


class TestStrategicAgentPayoffs:
    """Test that different payoff schemes work correctly"""
    
    @pytest.mark.parametrize("payoff_name", ["BinaryMG", "ScaledMG", "DollarGame"])
    def test_payoff_schemes_dont_crash(self, payoff_name):
        """Test that each payoff scheme can be used"""
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY[payoff_name]
        )
        
        # Choose action
        history = [1, -1, 1]
        action = agent.choose_action(history)
        
        # Update with some flow
        agent.update(N=100, flow=10, price=100.0, lambda_value=0.01)
        
        # Should not crash
        assert isinstance(agent.points, float)
        assert isinstance(agent.wins, int)