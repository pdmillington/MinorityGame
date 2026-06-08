#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:57:58 2025

@author: petermillington
"""

from typing import List, Dict, Any, Optional
import numpy as np
from core.base_agent import BaseAgent, StrategicAgent
from core.noise_player import NoisePlayer
from core.m_maker import MarketMaker
from core.players_factory import build_players
from core.game_config import GameConfig
from data.stats_recorder import StatsRecorder
from payoffs.mg import PAYOFF_REGISTRY


SAFE_MAX = 1e200
SAFE_MIN = - SAFE_MAX

# ── Payoff dispatch tables derived from PAYOFF_REGISTRY ───────────────────────
# _PAYOFF_CODE   : class name  -> integer code
# _DELAYED_CODES : set of codes where mode == "delayed"
# _PAYOFF_SCORE_FN : code -> get_reward_vector callable
# _PAYOFF_WIN_FN   : code -> get_win callable

_PAYOFF_CODE = {
    type(v).__name__: i
    for i, v in enumerate(PAYOFF_REGISTRY.values())
    }

_DELAYED_CODES: set = {
    _PAYOFF_CODE[type(v).__name__]
    for v in PAYOFF_REGISTRY.values()
    if getattr(v, "mode", "immediate") == "delayed"
    }

_PAYOFF_SCORE_FN: Dict[int, Any] = {
    _PAYOFF_CODE[type(v).__name__]: v.get_reward_vector
    for v in PAYOFF_REGISTRY.values()
}
 
# Win condition classification derived from registry.
# "minority" payoffs: win if action == -sign(A)  (MG variants)
# "majority" payoffs: win if action == +sign(A)  (DollarGame)
# Determined by whether get_win checks a_i == minority or a_i == majority.
# We detect this from mode: delayed payoffs are majority-style.
_WIN_IS_MAJORITY: set = _DELAYED_CODES.copy()


class Game:
    """
    Main Game Engine for Minority Game and Dollar Game
    Coordinates agents, market maker and price dynamics
    
    Responsibilities:
        - initialize population from specification
        - Run game rounds
        - update global states (history, price)
        - coordinate agent and MM updates
        - record statistics
    Virtual strategy scoring uses standard annealed approximation:
        each virtual strategy is scored against actual aggregate flow
        ignoring the correction that would happen if the agent played the virtual
        strategy instead.  The correction is negligible for large N, but grows
        in importance near the phase change.

    Parameters
    ----------
    population_spec : dict
        Specification for population composition
    cgf : GameConfig
        Game configuration(rounds, lambda, MM, etc.)
    agent_class_map : dict, optional
        Mapping from agent_type str to class
    activation_schedule: dict, optional
    Returns
    -------
    results: dict
        Dictionary of results
    """
    def __init__(self,
                 population_spec: Dict[str, Any],
                 cfg: GameConfig,
                 agent_class_map: Optional[Dict[str, type]] = None,
                 activation_schedule: Dict[int, List[int]] = None):
        """
        Initialize game with population and configuration
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Default agent types
        if agent_class_map is None:
            agent_class_map = {
                "strategic": StrategicAgent,
                "noise": NoisePlayer
            }
        # Build population
        self.players, self.meta, self.cohort_id = build_players(
            population_spec,
            agent_class_map=agent_class_map,
            rng=self.rng,
            shuffle=True,
            per_player_seeds=False,
            master_seed=cfg.seed
            )
        
        # Apply activation schedule if it has been passed
        if activation_schedule:
            self._apply_activation_schedule(activation_schedule)
            
        # Game state
        self.n = len(self.players)
        self.rounds = cfg.rounds
        self.lambda_value = cfg.lambda_ or (1 / (50 * self.n))
        self.price = float(self.cfg.price)
        self.current_round = 0
        # History
        self.max_memory = max(p.memory for p in self.players)
        self.history=list(self.rng.choice([-1, +1], size=self.max_memory))
        # Aggregate flow
        self.A = 0
        # Market Maker
        self.mm = MarketMaker() if cfg.mm else None
        # Statistics recorder
        k_max = max(p.num_strategies for p in self.players
                    if hasattr(p, 'num_strategies'))
        self.stats = StatsRecorder(
            N = self.n,
            rounds = self.rounds,
            k_max = k_max,
            record_agent_series=cfg.record_agent_series,
            record_strategies=cfg.record_strategies,
            )
        
        # Store results
        self.results: Optional[Dict[str, Any]] = None
        
        # Build vectorised arrays from agent objects at init
        self._build_fast_arrays()
        
    # Fast array construction
    def _build_fast_arrays(self) -> None:
        """
        Extract per-agent statet into numpy arrays for the vectorised loop.
        Called at init and every reset.
        Fast Arrays
        ----
        _fa_strategic    bool  (N,)
        _fa_memory       int32 (N,)
        _fa_n_strats     int32 (N,)
        _fa_strategies   int8  (N, S_max, 2^M_max)
        _fa_scores       f64   (N, S_max)
        _fa_pos_lim      int32 (N,)
        _fa_paycode      int32 (N,)         -1 for noise agents
        _fa_score_lam    f64   (N,)
        _fa_active_from  int32 (N,)
        _fa_position     int32 (N,)
        _fa_wins         int32 (N,)
        _fa_points       f64   (N,)
        _fa_best_strat   int16 (N,)
        _fa_prev_action  int8  (N,)
        _fa_prev_A       int
        """
        N = self.n
        S_max = max(
            p.num_strategies for p in self.players
            if hasattr(p, "num_strategies")
            )
        M_max = self.max_memory
        
        self._fa_strategic = np.array(
            [isinstance(p, StrategicAgent) for p in self.players], dtype=bool
        )
        self._fa_memory      = np.array(
            [p.memory if hasattr(p, "memory") else M_max for p in self.players],
            dtype=np.int32,
        )
        self._fa_n_strats    = np.array(
            [p.num_strategies if hasattr(p, "num_strategies") else 1
             for p in self.players],
            dtype=np.int32,
        )
        self._fa_pos_lim     = np.array(
            [p.position_limit if p.position_limit is not None else 0
             for p in self.players],
            dtype=np.int32,
        )
        self._fa_score_lam   = np.array(
            [p.score_lambda if hasattr(p, "score_lambda") else 0.0
             for p in self.players],
            dtype=np.float64,
        )
        self._fa_active_from = np.array(
            [p.active_from if hasattr(p, "active_from") else 0
             for p in self.players],
            dtype=np.int32,
        )
        self._fa_always_trade = np.array(
            [p.always_trade if hasattr(p, "always_trade") else False
             for p in self.players],
            dtype=bool,
        )
        
        # Payiff codes from PAYOFF_REGISTRY class names
        pcodes = np.full(N, -1, dtype=np.int32)
        for i, p in enumerate(self.players):
            if isinstance(p, StrategicAgent):
                pcodes[i] = _PAYOFF_CODE.get(type(p.payoff).__name__, -1)
        self._fa_paycode = pcodes
        
        # Strategy tables (N, S_max, 2^M_max)
        table_cols = 2 ** M_max
        strat = np.zeros((N, S_max, table_cols), dtype=np.int8)
        for i, p in enumerate(self.players):
            if isinstance(p, StrategicAgent):
                s = p.num_strategies
                cols = 2 ** p.memory
                strat[i, :s, :cols] = p.strategies.astype(np.int8)
        self._fa_strategies = strat
        
        # Scores (N, S_max)
        scores = np.zeros((N, S_max), dtype=np.float64)
        for i, p in enumerate(self.players):
            if isinstance(p, StrategicAgent):
                scores[i, :p.num_strategies] = p.scores
        self._fa_scores = scores
 
        # State arrays
        self._fa_position    = np.zeros(N, dtype=np.int32)
        self._fa_wins        = np.zeros(N, dtype=np.int32)
        self._fa_points      = np.zeros(N, dtype=np.float64)
        self._fa_best_strat  = np.full(N, -1, dtype=np.int16)
        self._fa_prev_action = np.zeros(N, dtype=np.int8)
        self._fa_prev_A      = 0
 
        self._S_max = S_max
        self._M_max = M_max
        
        # Pre-compute power of 2 vectors for history to index conversion
        self._fa_h_powers: Dict[int, np.ndarray] =  {}
        for m in np.unique(self._fa_memory):
            self._fa_h_powers[int(m)] = (
                2 ** np.arange(int(m) - 1, -1, -1, dtype=np.int32)
            )

    def _sync_arrays_to_agents(self) -> None:
        """
        Write scalar finals from fast arrays back to agent objects.
        Called before finalize so StatsRecorder.finalize sees correct values.
        """
        for i, p in enumerate(self.players):
            p.position = int(self._fa_position[i])
            p.wins     = int(self._fa_wins[i])
            p.points   = float(self._fa_points[i])
            if isinstance(p, StrategicAgent):
                s    = p.num_strategies
                cols = 2 ** p.memory
                p.scores     = self._fa_scores[i, :s].copy()
                p.strategies = self._fa_strategies[i, :s, :cols].astype(np.int8)
                best          = int(self._fa_best_strat[i])
                p.strategy    = best if best >= 0 else None

    # Vectorised helpers

    def _history_index(self, h_full: np.ndarray) -> np.ndarray:
        """
        Converts global history to per agent strategy table column indices.
        
        Parameters
        -------
        h_full: (M_max,) int8   values in {-1, +1}
        
        Returns
        ------
        (N,) int32
        """
        h_bin = ((h_full + 1) // 2).astype(np.int32)
        idx_out = np.empty(self.n, dtype=np.int32)
        
        for m, pows in self._fa_h_powers.items():
            mask = self._fa_memory == m
            bits = h_bin[self._M_max - m:]
            idx_out[mask] = int(np.dot(bits, pows))
        
        return idx_out
    
    def _choose_strategies(self, h_indices: np.ndarray):
        """
        Select best strategy per agent and return chosen actions.
 
        Single-strategy agents always use index 0 — no RNG call needed.
        Multi-strategy agents: argmax of active scores, random tie-breaking.
 
        Returns
        -------
        chosen   : (N,) int8
        best_idx : (N,) int32
        """
        N        = self.n
        scores   = self._fa_scores
        n_strats = self._fa_n_strats
        strats   = self._fa_strategies
        
        mask_unused = np.arange(self._S_max)[None, :] >= n_strats[:, None]
        masked = scores.copy()
        masked[mask_unused] = -np.inf
 
        best_idx = np.argmax(masked, axis=1).astype(np.int32)
        
        max_scores = scores[np.arange(N), best_idx]
        active_mask = ~mask_unused
        is_tied = (
            ((scores==max_scores[:, None]) & active_mask)
            .sum(axis=1) > 1
            )
 
        # Random tie breaking only for tied agents
        if np.any(is_tied):
            perm = self.rng.permutation(self._S_max)
            masked_perm = masked[:, perm]
            best_perm = np.argmax(masked_perm, axis=1)
            best_idx[is_tied] = perm[best_perm[is_tied]]
                
        chosen = strats[np.arange(N), best_idx, h_indices].astype(np.int8)
        return chosen, best_idx

    def _enforce_position_limits(self, chosen: np.ndarray) -> np.ndarray:
        """Block actions that would breach position limits."""
        lim = self._fa_pos_lim
        if not np.any(lim > 0):
            return chosen
        out      = chosen.copy()
        new_pos  = self._fa_position + chosen
        exceeded = (lim > 0) & (np.abs(new_pos) > lim)
        out[exceeded] = 0
        return out
        
    def _update_scores_wins_points(
        self,
        h_indices: np.ndarray,
        chosen:    np.ndarray,
        A:         int,
    ) -> None:
        """
        Fully vectorised update of strategy scores, win counts, and points.
 
        Win conditions (derived from mg.py):
        - All MG variants: win if chosen == -sign(A)  (in the minority)
        - DollarGame:      win if prev_action == +sign(A_prev)  (in the majority)
 
        Points use get_reward_vector applied to a (N,1) chosen-action array,
        which is equivalent to get_reward but vectorised — no scalar loop needed.
 
        Delayed payoffs (DollarGame) settle the previous round's action
        against the previous round's A.
        """
        N        = self.n
        A_prev   = self._fa_prev_A
        prev_act = self._fa_prev_action
        strats   = self._fa_strategies
        score_lam = self._fa_score_lam
        pcodes   = self._fa_paycode
 
        # Virtual actions for all strategies: (N, S_max)
        virt = strats[
            np.arange(N)[:, None],
            np.arange(self._S_max)[None, :],
            h_indices[:, None],
        ].astype(np.float64)
 
        # Score decay
        new_scores = (1.0 - score_lam[:, None]) * self._fa_scores
        
        # Precompute sign values once
        sign_A = int(np.sign(A)) if A != 0 else 0
        sign_A_prev = int(np.sign(A_prev)) if A_prev != 0 else 0
 
        for code, score_fn in _PAYOFF_SCORE_FN.items():
            mask = pcodes == code
            if not np.any(mask):
                continue
 
            # Use A_prev for delayed payoffs, A for immediate
            delayed = code in _DELAYED_CODES
            A_use = A_prev if delayed else A
 
            # Vectorised score update using the payoff's own get_reward_vector
            new_scores[mask] += score_fn(
                virt[mask], A_use, self.n, self.lambda_value
            )
 
            # Points.  Apply get_reward to chosen action
            if delayed:
                # Delayed: settle previous action against previous A
                act_for_points = prev_act[mask].astype(np.float64)[:, None]
            else:
                # Immediate: settle current action against current A
                act_for_points = chosen[mask].astype(np.float64)[:, None]
                
                
            points_delta = score_fn(
                act_for_points, A_use, self.n, self.lambda_value
                )
            self._fa_points[mask] += points_delta[:, 0]
                
            # Wins
            if delayed:    
                if A_prev != 0:
                    if code in _WIN_IS_MAJORITY:
                        # Dollar game: win if prev_action==sign(A_prev)
                        self._fa_wins[mask] += (
                            prev_act[mask] == sign_A_prev
                            ).astype(np.int32)
                    else:
                        self._fa_wins[mask] +=(
                            prev_act[mask] == -sign_A_prev
                            ).astype(np.int32)
            else:
                if A != 0:
                    if code in _WIN_IS_MAJORITY:
                        self._fa_wins[mask] += (
                            chosen[mask] == sign_A
                            ).astype(np.int32)
                    else:
                        self._fa_wins[mask] += (
                            chosen[mask] == -sign_A
                            ).astype(np.int32)
 
        self._fa_scores = new_scores
 
    def _reset_for_run(self) -> None:
        """
        Resets all states for a fresh simulation run.
        Automatically called by run() if there are multiple runs.
        """
        #Global state
        self.history = list(self.rng.choice([-1, 1], size=self.max_memory))
        self.price = self.cfg.price
        self.A = 0
        self.current_round = 0

        if self.mm:
            self.mm.reset()
        # Reset for all agents
        for p in self.players:
            if isinstance(p, StrategicAgent):
                p.strategies = self.rng.choice(
                    [-1, 1],
                    size=(p.num_strategies, 2 ** p.memory))
                p.scores = np.zeros(p.num_strategies)
                p.strategy = None
                p.strategy_switches = 0
                p._pend_idx = p._pend_virt = p._pend_action = p._pend_chosen = None
                p._prev_idx = p._prev_virt = p._prev_action = p._prev_chosen = None

            p.position = 0
            p.cash = p.initial_cash if hasattr(p, 'initial_cash') else 0.0
            p.wealth = p.cash
            p.points = 0
            p.wins = 0
        
        self._build_fast_arrays()
            
    def _apply_activation_schedule(self, schedule: Dict[int, List[int]]) -> None:
        """
        Assign active from values to agent by cohort index
        schedule = {cohort_index: [active_from, active_from ...]}
        """
        for cohort_idx, activation_list in schedule.items():
            cohort_agents = [p for p in self.players
                             if p.cohort_id==cohort_idx]
            if len(cohort_agents) != len(activation_list):
                raise ValueError(
                    f"Cohort {cohort_idx}:"
                    f"{len(activation_list)} activation values "
                    f"for {len(cohort_agents)} agents"
                    )
            for agent, active_from in zip(cohort_agents, activation_list):
                agent.active_from = active_from

    def _clip_or_nan(self, x:float) -> float:
        """Ensures that prices do not blow up in large numbers of rounds"""
        if np.isnan(x) or np.isinf(x):
            return np.nan
        if x > SAFE_MAX: return SAFE_MAX
        if x < SAFE_MIN: return SAFE_MIN
        return float(x)

    def play_round(self) -> None:
        """
        Play a single round of a game - fully vectorised across agents.
        1. History → per-agent column indices
        2. Strategy selection (vectorised, random tie-break)
        3. Activation mask
        4. Position limit enforcement
        5. Aggregate flow A
        6. Price update
        7. Score / win / point update
        8. Position update
        9. Best strategy tracking
        10. History update
        11. Buffer A and actions for delayed payoff
        """
        # Update current round counter
        self.current_round += 1
        
        h_full = np.array(self.history, dtype=np.int8)
        h_indices = self._history_index(h_full)
        
        chosen, best_idx = self._choose_strategies(h_indices)
        
        # Inactive agents: actions set to 0
        inactive = self.current_round <= self._fa_active_from
        chosen[inactive] = 0

        # Position limits
        chosen = self._enforce_position_limits(chosen)
        
        # Agents sit out if score below threshold
        if self.cfg.grand_canonical:
            best_scores = self._fa_scores[np.arange(self.n), best_idx]
            frozen = (best_scores < self.cfg.gc_threshold) & ~self._fa_always_trade
            chosen[frozen] = 0
        
        # Aggregate flow
        self.A = int(chosen.sum())
        if self.A == 0:
            self.A = int(self.rng.choice([-1,+1]))
        
        # Price 
        mm_position = self.mm.position if self.mm else 0
        r_t =  self.A * self.lambda_value - mm_position * self.lambda_value
        self.price = self._clip_or_nan(self.price * np.exp(r_t))

        if self.mm:
            self.mm.update(self.price, self.A)

        self._update_scores_wins_points(h_indices, chosen, self.A)
        self._fa_position += chosen

        # Strategy tracking for switch counting
        self._fa_best_strat = best_idx.astype(np.int16)

        # Update “public info” bit for the MG history. Define it on the same 'flow'
        minority_action = -1 if self.A > 0 else 1
        self.history = (self.history + [minority_action])[-self._M_max:]

        # Buffer for delayed payoff
        self._fa_prev_action = chosen.copy()
        self._fa_prev_A = self.A


    def run(self, reset: bool = False) -> Dict[str, Any]:
        """
        Run a simulation and reset for a new simulation run,
        
        Parameter
        reset: bool
            If True, resets all states before running
        
        Returns
        dict
            Results dictionary with price series, agent stats etc.
            
        """
        if reset:
            self._reset_for_run()

        self._fa_prev_A = 0
        self._fa_prev_action = np.zeros(self.n, dtype=np.int8)

        initial_cash = np.array(
            [p.cash if hasattr(p, "cash") else 0.0 for p in self.players],
            dtype=np.float64,
            )

        self.stats.record_initial_state(
            self.price, self.players,
            position_arr=self._fa_position,
            wins_arr= self._fa_wins,
            points_arr=self._fa_points,
            initial_cash_arr=initial_cash,
            )

        # Run all rounds
        for t in range(self.rounds):
            self.play_round()
            self.stats.record_round(
                t, self.price, self.A, self.players,
                position_arr=self._fa_position,
                wins_arr=self._fa_wins,
                points_arr=self._fa_points,
                best_strategy_arr=self._fa_best_strat,
                )

        self._sync_arrays_to_agents()

        # Finalize stats
        results = self.stats.finalize(
            self.players,
            fast_path=True,
            initial_cash_arr=initial_cash,
            )

        # Add metadata
        results["cohort_ids"] = np.array(self.cohort_id, dtype=int)
        results["cohorts"] = self.meta["cohorts"]
        results["config"] = self.cfg

        self.results = results
        return results

    @property
    def is_stable(self, max_price: float= 10.0, min_price: float = 0.1) -> bool:
        """
        Check on stability of prices
        Params
        max_price: float
            Upper limit for price stability
        min_price: float
            Lower limit for price stability
            
        Returns
            True if prices are within limits
        """
        if self.results is None:
            raise ValueError("No results available, run simulation!")

        prices = self.results.get("prices")
        if prices is None:
            return False

        return (prices.max() < max_price and
                prices.min() > min_price and
                not np.any(np.isnan(prices)))

    def time_to_explosion(self, threshold: float = 10000.0) -> int:
        """
        Calculate number of rounds until price explosion
        Parameter
            threshold: price threshold for explosion
        Returns
            Number of rounds until explosion (or total number of rounds if no explosion)
        """
        if self.results is None:
            raise ValueError("No results available.  Please run simulation!")

        prices = self.results.get("prices")
        if prices is None:
            return 0
        
        for t, price in enumerate(prices):
            if price > threshold or price < 1 / threshold:
                return t
        return len(prices)
