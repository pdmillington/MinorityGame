#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:37:22 2025

@author: petermillington
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass
class Agent:
    """Agent in the dollar game"""
    id: int
    wealth: float
    position: int = 0  # shares held (can be negative for short)
    strategy: str = "random"  # strategy type
    memory: List = None  # for storing historical data
    
    def __post_init__(self):
        if self.memory is None:
            self.memory = []

class DollarGame:
    """
    Single-period dollar game implementation.
    Agents trade a single asset with random price movements.
    """
    
    def __init__(self, n_agents=100, initial_wealth=1000, 
                 price_volatility=0.02, transaction_cost=0.001):
        self.n_agents = n_agents
        self.initial_wealth = initial_wealth
        self.price_volatility = price_volatility
        self.transaction_cost = transaction_cost
        
        # Initialize agents
        self.agents = [Agent(i, initial_wealth) for i in range(n_agents)]
        
        # Market variables
        self.price = 100.0  # initial asset price
        self.price_history = [self.price]
        self.wealth_history = []
        self.total_volume = 0
        
        # Results storage
        self.results = {
            'wealth_evolution': [],
            'price_evolution': [],
            'volume_evolution': [],
            'gini_evolution': [],
            'positions': []
        }
    
    def agent_decision(self, agent: Agent) -> int:
        """
        Agent makes trading decision. Returns desired position change.
        Positive = buy, Negative = sell, 0 = hold
        """
        if agent.strategy == "random":
            # Random trading with slight bias
            return np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        
        elif agent.strategy == "momentum":
            # Simple momentum strategy
            if len(self.price_history) < 2:
                return 0
            price_change = self.price_history[-1] - self.price_history[-2]
            if price_change > 0:
                return 1
            elif price_change < 0:
                return -1
            return 0
        
        elif agent.strategy == "contrarian":
            # Contrarian strategy
            if len(self.price_history) < 2:
                return 0
            price_change = self.price_history[-1] - self.price_history[-2]
            if price_change > 0:
                return -1
            elif price_change < 0:
                return 1
            return 0
        
        elif agent.strategy == "mean_reversion":
            # Mean reversion strategy
            if len(self.price_history) < 10:
                return 0
            recent_mean = np.mean(self.price_history[-10:])
            if self.price > recent_mean * 1.02:
                return -1
            elif self.price < recent_mean * 0.98:
                return 1
            return 0
    
    def execute_trades(self):
        """Execute all agent trades and update market price"""
        valid_trades = []
        
        for agent in self.agents:
            desired_change = self.agent_decision(agent)
            
            # Check if agent can afford the trade
            trade_cost = abs(desired_change) * self.price * (1 + self.transaction_cost)
            
            if desired_change > 0:  # Buying
                if agent.wealth >= trade_cost:
                    valid_trades.append((agent, desired_change))
            elif desired_change < 0:  # Selling
                if agent.position >= abs(desired_change):  # Can only sell what you own
                    valid_trades.append((agent, desired_change))
            else:  # No trade
                valid_trades.append((agent, 0))
        
        # Calculate actual net demand from valid trades only
        net_demand = sum(trade[1] for trade in valid_trades)
        
        # Update price based on actual net demand
        price_impact = net_demand / self.n_agents * 0.1
        noise = np.random.normal(0, self.price_volatility)
        self.price = self.price * (1 + price_impact + noise)
        self.price = max(self.price, 1.0)  # Prevent negative prices
        
        # Execute valid trades
        total_volume = 0
        for agent, trade_size in valid_trades:
            if trade_size != 0:
                trade_value = abs(trade_size) * self.price
                cost = trade_value * self.transaction_cost
                
                # Update agent position and wealth
                agent.position += trade_size
                agent.wealth -= trade_size * self.price + cost
                total_volume += abs(trade_size)
        
        self.total_volume = total_volume
        self.price_history.append(self.price)
    
    def calculate_gini_coefficient(self) -> float:
        """Calculate Gini coefficient for wealth distribution"""
        total_wealth = [agent.wealth + agent.position * self.price for agent in self.agents]
        total_wealth.sort()
        n = len(total_wealth)
        cumsum = np.cumsum(total_wealth)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def run_simulation(self, n_periods=1000, strategy_mix=None):
        """Run the simulation for n_periods"""
        if strategy_mix is None:
            strategy_mix = {"random": 0.5, "momentum": 0.2, "contrarian": 0.2, "mean_reversion": 0.1}
        
        # Assign strategies
        strategies = []
        for strategy, proportion in strategy_mix.items():
            strategies.extend([strategy] * int(self.n_agents * proportion))
        
        # Fill remaining with random
        while len(strategies) < self.n_agents:
            strategies.append("random")
        
        np.random.shuffle(strategies)
        for i, strategy in enumerate(strategies):
            self.agents[i].strategy = strategy
        
        # Run simulation
        for period in range(n_periods):
            # Execute trades
            self.execute_trades()
            
            # Record statistics
            total_wealth = [agent.wealth + agent.position * self.price for agent in self.agents]
            self.results['wealth_evolution'].append(total_wealth.copy())
            self.results['price_evolution'].append(self.price)
            self.results['volume_evolution'].append(self.total_volume)
            self.results['gini_evolution'].append(self.calculate_gini_coefficient())
            self.results['positions'].append([agent.position for agent in self.agents])
    
    def plot_results(self):
        """Generate comprehensive plots of simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Price evolution
        axes[0, 0].plot(self.results['price_evolution'])
        axes[0, 0].set_title('Price Evolution')
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].set_ylabel('Price')
        
        # Wealth distribution over time (heatmap)
        wealth_matrix = np.array(self.results['wealth_evolution']).T
        im = axes[0, 1].imshow(wealth_matrix, aspect='auto', cmap='viridis')
        axes[0, 1].set_title('Wealth Evolution (Agents x Time)')
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].set_ylabel('Agent ID')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Gini coefficient evolution
        axes[0, 2].plot(self.results['gini_evolution'])
        axes[0, 2].set_title('Wealth Inequality (Gini Coefficient)')
        axes[0, 2].set_xlabel('Period')
        axes[0, 2].set_ylabel('Gini Coefficient')
        
        # Final wealth distribution
        final_wealth = self.results['wealth_evolution'][-1]
        axes[1, 0].hist(final_wealth, bins=30, alpha=0.7)
        axes[1, 0].set_title('Final Wealth Distribution')
        axes[1, 0].set_xlabel('Wealth')
        axes[1, 0].set_ylabel('Frequency')
        
        # Trading volume
        axes[1, 1].plot(self.results['volume_evolution'])
        axes[1, 1].set_title('Trading Volume')
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].set_ylabel('Volume')
        
        # Price vs Volume scatter
        axes[1, 2].scatter(self.results['volume_evolution'], 
                          self.results['price_evolution'], alpha=0.6)
        axes[1, 2].set_title('Price vs Volume')
        axes[1, 2].set_xlabel('Volume')
        axes[1, 2].set_ylabel('Price')
        
        plt.tight_layout()
        plt.show()
    
    def get_efficiency_metrics(self) -> dict:
        """Calculate market efficiency metrics"""
        prices = np.array(self.results['price_evolution'])
        returns = np.diff(np.log(prices))
        
        return {
            'price_volatility': np.std(returns),
            'final_gini': self.results['gini_evolution'][-1],
            'initial_gini': self.results['gini_evolution'][0],
            'avg_volume': np.mean(self.results['volume_evolution']),
            'price_trend': (prices[-1] - prices[0]) / prices[0],
            'return_autocorr': np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        }

# Example usage and analysis
if __name__ == "__main__":
    # Run simulation
    game = DollarGame(n_agents=200, initial_wealth=1000, price_volatility=0.01)
    
    # Define strategy mix
    strategy_mix = {
        "random": 0.4,
        "momentum": 0.2, 
        "contrarian": 0.2,
        "mean_reversion": 0.2
    }
    
    game.run_simulation(n_periods=500, strategy_mix=strategy_mix)
    
    # Display results
    game.plot_results()
    
    # Print efficiency metrics
    metrics = game.get_efficiency_metrics()
    print("\nMarket Efficiency Metrics:")
    print("-" * 30)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Analyze wealth concentration
    final_wealth = game.results['wealth_evolution'][-1]
    print(f"\nWealth Analysis:")
    print(f"Richest agent: {max(final_wealth):.2f}")
    print(f"Poorest agent: {min(final_wealth):.2f}")
    print(f"Wealth ratio (rich/poor): {max(final_wealth)/min(final_wealth):.2f}")
    print(f"Agents with negative wealth: {sum(1 for w in final_wealth if w < 0)}")