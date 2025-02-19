#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:14:27 2024

@author: petermillington
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

class Player:
    def __init__(self, m, s, histories, use_new_payoff):
        self.m = m
        self.s = s
        self.points = 0
        self.points_per_round = []
        self.histories = histories
        self.strategies = [self.generate_random_strategy() for _ in range(s)]
        self.use_new_payoff = use_new_payoff

    def generate_random_strategy(self):
        """Generate a random strategy with actions for each of the 2**m histories and initialize its score."""
        return {
            "actions": [random.choice([0, 1]) for _ in range(2 ** self.m)],
            "score": 0
        }

    def choose_action(self, history, random_flag):
        # Find the index of the current history in the shared histories list
        index = self.histories.index(history)
        # Select the strategy with the highest score (break ties randomly)
        if random_flag == 0:
            best_strategies = [strategy for strategy in self.strategies if strategy["score"] == max(s["score"] for s in self.strategies)]
            selected_strategy = random.choice(best_strategies)
        else:
            selected_strategy = random.choice(self.strategies)
        return selected_strategy["actions"][index]

    def update_strategy_scores(self, history, minority_action, actions, N):
        """Update the score of each strategy based on whether it predicted the minority action correctly."""
        # Count how many players chose action 1 and action 0
        count_1 = sum(actions)
        minority_count = count_1 if minority_action == 1 else (N-count_1)
        
        # Apply the payoff system based on the flag
        if self.use_new_payoff:
            reward = ((N / minority_count) - 2)
            #penalty = (2 - (N / minority_count))
            penalty = 0
        else:
            # Standard payoff: 1 point for choosing the minority
            reward = 1
            penalty = 0
        #print(f'{count_1}')
        #print(f'{penalty:.3f}') 
        index = self.histories.index(history)
        for strategy in self.strategies:
            if strategy["actions"][index] == minority_action:
                strategy["score"] += reward  # Reward correct prediction
            else:
                strategy["score"] += penalty
        
                
    def update_points(self, minority_action, action, attendance, N):
        minority_count = attendance if minority_action == 1 else N - attendance
        
        if self.use_new_payoff:
            reward = (N/minority_count) - 2
            #penalty = 2 - (N / minority_count)
            penalty = 0
        else: 
            reward = 1
            penalty = 0
            
        if action == minority_action:
            self.points += reward
        else:
            self.points += penalty
            
        self.points_per_round.append(self.points)



class MinorityGame:
    def __init__(self, n, m, rounds, s, random_flag, use_new_payoff):
        self.n = n        # Number of players
        self.m = m        # Memory size
        self.s = s        # Number of strategies per player
        self.rounds = rounds
        self.random_flag = random_flag
        self.use_new_payoff = use_new_payoff
        # Generate all possible histories of length m
        self.histories = [bin(i)[2:].zfill(m) for i in range(2 ** m)]
        print(self.histories)
        # Initialize players with the shared list of histories
        self.players = [Player(m, s, self.histories, self.use_new_payoff) for _ in range(n)]
        # Initialize a random history of length m for the first round
        self.history = ''.join(random.choice(['0', '1']) for _ in range(m))
        # Track attendance for each round to calculate variance later
        self.attendance = []

    def play_round(self):
        # Each player chooses an action based on the current history by using their strategies
        actions = [player.choose_action(self.history, self.random_flag) for player in self.players]
        count_1 = sum(actions)
        attendance = count_1  # Attendance
        minority_action = 1 if count_1 <= self.n / 2 else 0

        # Store attendance for variance calculation
        self.attendance.append(attendance)

        # Reward players who chose the minority action and update their strategies
        for player, action in zip(self.players, actions):
            player.update_points(minority_action, action, attendance, self.n)
            player.update_strategy_scores(self.history, minority_action, actions, self.n)

        # Update history for the next round
        self.history = self.history[1:] + str(minority_action)

    def run(self):
        # Run the game for the specified number of rounds
        for _ in range(self.rounds):
            self.play_round()

    def get_variance_normalized(self):
        """Calculate and return the normalized variance of attendance."""
        variance = np.var(self.attendance)
        return variance / self.n
