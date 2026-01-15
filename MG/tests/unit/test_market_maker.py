#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:10:56 2026

@author: petermillington
"""

"""
Unit tests for MarketMaker
"""
import pytest
from core.m_maker import MarketMaker


class TestMarketMakerInitialization:
    """Test MarketMaker initialization"""
    
    def test_default_initialization(self):
        """Test market maker initializes with default values"""
        mm = MarketMaker()
        
        assert mm.position == 0
        assert mm.cash == 0.0
        assert mm.wealth == 0.0
        assert len(mm.position_history) == 0
        assert len(mm.cash_history) == 0
        assert len(mm.wealth_history) == 0
    
    def test_initialization_with_params(self):
        """Test market maker initialization with parameters"""
        mm = MarketMaker(initial_position=10, initial_cash=1000.0)
        
        assert mm.position == 10
        assert mm.cash == 1000.0
        assert mm.wealth == 1000.0


class TestMarketMakerUpdate:
    """Test MarketMaker update functionality"""
    
    def test_update_opposing_flow_positive(self):
        """Test that MM takes opposite side when flow is positive"""
        mm = MarketMaker()
        
        # Agents buy (+5), MM sells (-5)
        mm.update(price=100.0, flow=5)
        
        assert mm.position == -5, "MM should take opposite position (sell)"
        assert mm.cash == 500.0, "MM should receive cash from selling"
        assert mm.wealth == 0.0, "Wealth should be -5*100 + 500 = 0"
    
    def test_update_opposing_flow_negative(self):
        """Test that MM takes opposite side when flow is negative"""
        mm = MarketMaker()
        
        # Agents sell (-5), MM buys (+5)
        mm.update(price=100.0, flow=-5)
        
        assert mm.position == 5, "MM should take opposite position (buy)"
        assert mm.cash == -500.0, "MM should pay cash for buying"
        assert mm.wealth == 0.0, "Wealth should be 5*100 - 500 = 0"
    
    def test_update_history_tracking(self):
        """Test that update tracks history"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=5)
        
        assert len(mm.position_history) == 1
        assert len(mm.cash_history) == 1
        assert len(mm.wealth_history) == 1
        assert mm.position_history[0] == -5
        assert mm.cash_history[0] == 500.0
        assert mm.wealth_history[0] == 0.0
    
    def test_update_multiple_rounds(self):
        """Test multiple rounds of updates"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=5)
        mm.update(price=101.0, flow=-2)
        mm.update(price=99.0, flow=3)
        
        assert len(mm.position_history) == 3
        assert mm.position == -5 + 2 - 3  # -6
        assert len(mm.cash_history) == 3
        assert len(mm.wealth_history) == 3


class TestMarketMakerPnL:
    """Test MarketMaker P&L calculations"""
    
    def test_pnl_profit_from_spread(self):
        """Test P&L from selling high and buying low"""
        mm = MarketMaker()
        
        # Sell at 100
        mm.update(price=100.0, flow=5)  # MM position = -5, cash = 500
        
        # Buy back at 95
        mm.update(price=95.0, flow=-3)  # MM position = -2, cash = 500 - 3*95 = 215
        
        # Wealth = -2*95 + 215 = 25
        pnl = mm.get_pnl()
        assert pnl == 25.0, "Should profit from selling high and buying low"
    
    def test_pnl_loss_from_adverse_prices(self):
        """Test P&L from buying high and selling low"""
        mm = MarketMaker()
        
        # Buy at 100
        mm.update(price=100.0, flow=-5)  # MM position = 5, cash = -500
        
        # Sell at 95
        mm.update(price=95.0, flow=3)  # MM position = 2, cash = -500 + 3*95 = -215
        
        # Wealth = 2*95 - 215 = -25
        pnl = mm.get_pnl()
        assert pnl == -25.0, "Should lose from buying high and selling low"
    
    def test_pnl_flat_position(self):
        """Test P&L when position is flat"""
        mm = MarketMaker()
        
        # Buy and sell back at same price
        mm.update(price=100.0, flow=-5)  # MM position = 5, cash = -500
        mm.update(price=100.0, flow=5)   # MM position = 0, cash = 0
        
        pnl = mm.get_pnl()
        assert pnl == 0.0, "Should have zero P&L when flat at same price"


class TestMarketMakerReset:
    """Test MarketMaker reset functionality"""
    
    def test_reset_clears_state(self):
        """Test that reset clears all state"""
        mm = MarketMaker()
        
        # Build up some state
        mm.update(price=100.0, flow=5)
        mm.update(price=101.0, flow=-2)
        mm.update(price=99.0, flow=3)
        
        # Reset
        mm.reset()
        
        assert mm.position == 0, "Position should be reset"
        assert mm.cash == 0.0, "Cash should be reset"
        assert mm.wealth == 0.0, "Wealth should be reset"
        assert len(mm.position_history) == 0, "Position history should be cleared"
        assert len(mm.cash_history) == 0, "Cash history should be cleared"
        assert len(mm.wealth_history) == 0, "Wealth history should be cleared"
    
    def test_reset_allows_reuse(self):
        """Test that MM can be reused after reset"""
        mm = MarketMaker()
        
        # First run
        mm.update(price=100.0, flow=5)
        first_wealth = mm.wealth
        
        # Reset and second run
        mm.reset()
        mm.update(price=100.0, flow=5)
        second_wealth = mm.wealth
        
        assert first_wealth == second_wealth, "Should behave the same after reset"


class TestMarketMakerStatistics:
    """Test MarketMaker statistics calculations"""
    
    def test_average_position_empty(self):
        """Test average position with no history"""
        mm = MarketMaker()
        
        avg_pos = mm.get_average_position()
        assert avg_pos == 0, "Average position should be 0 with no history"
    
    def test_average_position_single_round(self):
        """Test average position with one round"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=5)
        
        avg_pos = mm.get_average_position()
        assert avg_pos == 5, "Average absolute position should be 5"
    
    def test_average_position_multiple_rounds(self):
        """Test average position over multiple rounds"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=5)   # position = -5
        mm.update(price=100.0, flow=-3)  # position = -2
        mm.update(price=100.0, flow=2)   # position = -4
        
        # Average of abs(-5, -2, -4) = (5 + 2 + 4) / 3 = 3.67
        avg_pos = mm.get_average_position()
        assert abs(avg_pos - 11/3) < 0.01, "Average position should be ~3.67"
    
    def test_max_position_empty(self):
        """Test max position with no history"""
        mm = MarketMaker()
        
        max_pos = mm.get_max_position()
        assert max_pos == 0, "Max position should be 0 with no history"
    
    def test_max_position_single_round(self):
        """Test max position with one round"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=5)
        
        max_pos = mm.get_max_position()
        assert max_pos == 5, "Max absolute position should be 5"
    
    def test_max_position_multiple_rounds(self):
        """Test max position over multiple rounds"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=2)   # position = -2
        mm.update(price=100.0, flow=-8)  # position = 6
        mm.update(price=100.0, flow=3)   # position = 3
        
        max_pos = mm.get_max_position()
        assert max_pos == 6, "Max absolute position should be 6"
    
    def test_current_exposure(self):
        """Test current exposure property"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=5)
        assert mm.current_exposure == 5, "Current exposure should be absolute position"
        
        mm.update(price=100.0, flow=-8)
        assert mm.current_exposure == 3, "Current exposure should update with position"


class TestMarketMakerEdgeCases:
    """Test edge cases for MarketMaker"""
    
    def test_zero_flow(self):
        """Test behavior with zero flow"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=0)
        
        assert mm.position == 0, "Position should remain 0 with zero flow"
        assert mm.cash == 0.0, "Cash should remain 0 with zero flow"
    
    def test_large_flows(self):
        """Test with large flow values"""
        mm = MarketMaker()
        
        mm.update(price=100.0, flow=1000)
        
        assert mm.position == -1000, "Should handle large positions"
        assert mm.cash == 100000.0, "Should handle large cash values"
    
    def test_price_at_zero(self):
        """Test behavior when price is zero (edge case)"""
        mm = MarketMaker()
        
        mm.update(price=0.0, flow=5)
        
        assert mm.position == -5
        assert mm.cash == 0.0  # -5 * 0
        assert mm.wealth == 0.0
    
    def test_negative_prices(self):
        """Test behavior with negative prices (shouldn't happen but test robustness)"""
        mm = MarketMaker()
        
        mm.update(price=-100.0, flow=5)
        
        # Should still compute, even if economically nonsensical
        assert mm.position == -5
        assert mm.cash == -500.0
        assert mm.wealth == -1000.0


class TestMarketMakerRepresentation:
    """Test MarketMaker string representation"""
    
    def test_repr(self):
        """Test __repr__ method"""
        mm = MarketMaker(initial_position=10, initial_cash=500.0)
        
        repr_str = repr(mm)
        
        assert "MarketMaker" in repr_str
        assert "position=10" in repr_str
        assert "cash=500.00" in repr_str
        assert "wealth=" in repr_str