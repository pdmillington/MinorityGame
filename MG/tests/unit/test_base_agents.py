# -*- coding: utf-8 -*-

import pytest
import numpy as np
from core.base_agent import BaseAgent, StrategicAgent
from payoffs.mg import PAYOFF_REGISTRY


class TestBaseAgent:
    """Test abstract base agent functionality"""

    def test_position_limit_enforcement(self):
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            position_limit=5
        )
        agent.position = 3
        assert agent._enforce_action(1, current_round=0) == 1

        agent.position = 5
        assert agent._enforce_action(1, current_round=0) == 0

        agent.position = -4
        assert agent._enforce_action(-1, current_round=0) == -1

        agent.position = -5
        assert agent._enforce_action(-1, current_round=0) == 0

    def test_no_position_limit(self):
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            position_limit=None
        )
        agent.position = 1000
        assert agent._enforce_action(1, current_round=0) == 1

    def test_dormant_agent(self):
        """Agent is passive before active_from round."""
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            active_from=10
        )
        assert agent._enforce_action(1, current_round=5) == 0
        assert agent._enforce_action(1, current_round=10) == 1

    def test_apply_trade(self):
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            initial_cash=1000.0
        )
        agent._apply_trade(action=1, price=100.0)
        assert agent.position == 1
        assert agent.cash == 900.0
        assert agent.wealth == 1000.0

        agent._apply_trade(action=-2, price=110.0)
        assert agent.position == -1
        assert agent.cash == 1120.0
        assert agent.wealth == 1010.0

    def test_strategy_selection(self):
        agent = StrategicAgent(
            memory=2,
            num_strategies=3,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            rng=np.random.default_rng(42)
        )
        agent.scores = np.array([1.0, 5.0, 3.0])
        action = agent.choose_action([1, -1], current_round=0)
        assert agent.strategy == 1
        assert action in [-1, 1]

    def test_strategy_switches_counted(self):
        agent = StrategicAgent(
            memory=2,
            num_strategies=3,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            rng=np.random.default_rng(42)
        )
        initial_switches = agent.strategy_switches

        agent.scores = np.array([5.0, 1.0, 1.0])
        agent.choose_action([1, -1], current_round=0)
        first_strategy = agent.strategy
        switches_after_first = agent.strategy_switches

        agent.choose_action([1, 1], current_round=1)
        assert agent.strategy_switches == switches_after_first

        agent.scores = np.array([1.0, 5.0, 1.0])
        agent.choose_action([-1, 1], current_round=2)
        assert agent.strategy != first_strategy
        assert agent.strategy_switches == switches_after_first + 1


class TestStrategicAgentPayoffs:
    """Test that different payoff schemes work correctly"""

    @pytest.mark.parametrize("payoff_name", ["BinaryMG", "ScaledMG", "DollarGame"])
    def test_payoff_schemes_dont_crash(self, payoff_name):
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY[payoff_name]
        )
        action = agent.choose_action([1, -1, 1], current_round=0)
        agent.update(N=100, flow=10, price=100.0, lambda_value=0.01)
        assert isinstance(agent.points, float)
        assert isinstance(agent.wins, int)
        
class TestScoreDecay:
    """Test that score decay follows U_t = r_t + (1 - lambda) * U_{t-1}"""

    def _expected_score(self, rewards, lam):
        """Compute expected score analytically for a known reward sequence."""
        U = 0.0
        for r in rewards:
            U = (1.0 - lam) * U + r
        return U

    def test_no_decay(self):
        """lambda=0.0 gives pure accumulation — scores are simple sum."""
        rewards = [1.0, 1.0, 1.0]
        agent = StrategicAgent(
            memory=1,
            num_strategies=1,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            score_lambda=0.0
        )
        agent.scores = np.zeros(1)
        for r in rewards:
            agent.scores = (1.0 - agent.score_lambda) * agent.scores + r

        assert np.isclose(agent.scores[0], 3.0), \
            "lambda=0 should give pure sum of rewards"

    def test_full_decay(self):
        """lambda=1.0 gives pure last-round scoring."""
        rewards = [1.0, 2.0, 3.0]
        agent = StrategicAgent(
            memory=1,
            num_strategies=1,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            score_lambda=1.0
        )
        agent.scores = np.zeros(1)
        for r in rewards:
            agent.scores = (1.0 - agent.score_lambda) * agent.scores + r

        assert np.isclose(agent.scores[0], 3.0), \
            "lambda=1 should retain only last reward"

    def test_partial_decay(self):
        """lambda=0.5 follows geometric discounting correctly."""
        rewards = [1.0, 1.0, 1.0]
        lam = 0.5
        # U_1=1.0, U_2=1+0.5*1=1.5, U_3=1+0.5*1.5=1.75
        agent = StrategicAgent(
            memory=1,
            num_strategies=1,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            score_lambda=lam
        )
        agent.scores = np.zeros(1)
        for r in rewards:
            agent.scores = (1.0 - agent.score_lambda) * agent.scores + r

        expected = self._expected_score(rewards, lam)
        assert np.isclose(agent.scores[0], expected), \
            f"Expected {expected}, got {agent.scores[0]}"

    def test_mixed_rewards(self):
        """Decay works correctly with alternating positive and negative rewards."""
        rewards = [1.0, -1.0, 1.0]
        lam = 0.5
        # U_1=1.0, U_2=-1+0.5*1=-0.5, U_3=1+0.5*(-0.5)=0.75
        agent = StrategicAgent(
            memory=1,
            num_strategies=1,
            payoff=PAYOFF_REGISTRY["BinaryMG"],
            score_lambda=lam
        )
        agent.scores = np.zeros(1)
        for r in rewards:
            agent.scores = (1.0 - agent.score_lambda) * agent.scores + r

        expected = self._expected_score(rewards, lam)
        assert np.isclose(agent.scores[0], expected), \
            f"Expected {expected}, got {agent.scores[0]}"

    def test_default_score_lambda_is_zero(self):
        """score_lambda defaults to 0.0 — backward compatibility."""
        agent = StrategicAgent(
            memory=3,
            num_strategies=2,
            payoff=PAYOFF_REGISTRY["BinaryMG"]
        )
        assert agent.score_lambda == 0.0, \
            "Default score_lambda should be 0.0"