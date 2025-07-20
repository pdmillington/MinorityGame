#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 21:13:05 2025

@author: petermillington
"""

import numpy as np
import random

class Player:
    def __init__(self, m, s, use_new_payoff):
        self.m = m  # memory size
        self.s = s  # number of strategies
        self.use_new_payoff = use_new_payoff
        self.points = 0
        self.points_per_round = []

        # Strategies: each is a row of 0s or 1s, shape (s, 2**m)
        self.strategies = np.random.randint(0, 2, size=(s, 2 ** m))
        self.scores = np.zeros(s)

    def choose_action(self, global_history, random_flag):
        index = int(global_history[-self.m:], 2)
        if random_flag:
            return random.choice([0, 1])
        best_ids = np.flatnonzero(self.scores == self.scores.max())
        chosen = np.random.choice(best_ids)
        return self.strategies[chosen, index]

    def update_strategy_scores(self, global_history, minority_action, N, attendance):
        index = int(global_history[-self.m:], 2)
        minority_count = attendance if minority_action == 1 else N - attendance
        reward = N - minority_count if self.use_new_payoff else 1

        for i in range(self.s):
            if self.strategies[i, index] == minority_action:
                self.scores[i] += reward

    def update_points(self, minority_action, action, attendance, N):
        minority_count = attendance if minority_action == 1 else N - attendance
        reward = N - minority_count if self.use_new_payoff else 1

        if action == minority_action:
            self.points += reward

        self.points_per_round.append(self.points)


class MinorityGame:
    def __init__(self, n, min_m, max_m, rounds, s, random_flag, use_new_payoff, random_history):
        self.n = n
        self.rounds = rounds
        self.s = s
        self.random_flag = random_flag
        self.use_new_payoff = use_new_payoff
        self.random_history = random_history

        self.memory_assignments = self.assign_memory_groups(n, min_m, max_m)
        self.players = [
            Player(m_i, s, use_new_payoff)
            for m_i in self.memory_assignments
        ]

        # Start with a global history of max_m 0s and 1s
        self.max_m = max_m
        self.history = ''.join(random.choice(['0', '1']) for _ in range(max_m))
        self.attendance = []
        self.price = []
        self.price.append(10)

    def assign_memory_groups(self, n, min_m, max_m):
        memory_levels = list(range(min_m, max_m + 1))

        group_count = len(memory_levels)
        base_size = n // group_count
        remainder = n % group_count

        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(group_count)]
        memory_assignments = []

        for m, size in zip(memory_levels, group_sizes):
            memory_assignments.extend([m] * size)

        return memory_assignments

    def play_round(self):
        actions = [p.choose_action(self.history, self.random_flag) for p in self.players]
        attendance = sum(actions)
        minority_action = 1 if attendance < self.n / 2 else 0
        self.attendance.append(attendance)

        for player, action in zip(self.players, actions):
            player.update_points(minority_action, action, attendance, self.n)
            player.update_strategy_scores(self.history, minority_action, self.n, attendance)
            
        if self.random_history:
            self.history = ''.join(random.choice(['0', '1']) for _ in range(self.max_m))
        else:
            self.history = (self.history + str(minority_action))[-self.max_m:]

    def run(self):
        for _ in range(self.rounds):
            self.play_round()

    def get_variance_normalized(self):
        variance = np.var(self.attendance)
        return variance / self.n if self.n > 0 else 0
