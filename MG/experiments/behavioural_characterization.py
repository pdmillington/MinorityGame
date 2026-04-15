#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioural Strategy Characterization Experiment
=================================================

Distinguishes mean-reverting (contrarian) from trend-following (momentum)
behaviour at TWO levels:

  Individual level:
      rho_i^w = Corr( a_i(t),  R_t^w )

  Aggregate level:
      rho_agg^w = Corr( A_t,  R_t^w )

where the trend signal is the log price return over window w:

      R_t^w = log(P_{t-1}) - log(P_{t-1-w})

This measures the cumulative return over the w rounds immediately
preceding round t — the trend observable by agents when acting at t.

Using price returns rather than the attendance rolling mean T_t^w:
  - is economically interpretable (R > 0 = price rose, agent buys = TF)
  - gives the MG correlation measure actual power, since log price is a
    random walk with non-trivial variance even when A_t is near-uncorrelated

The phase indicator sigma_A/sqrt(N) and attendance stationarity tests
(ADF, KPSS) are reported separately in the attendance statistics table.
Log price stationarity is also tested there.

Key findings from pure populations:
  - MG:  rho_agg ~ 0, rho_i ~ 0  (no trend-following at either level)
         stationarity: attendance stationary, log price non-stationary
  - DG:  rho_agg > 0, rho_i > 0  (trend-following at both levels)
         log price collapses — non-stationary and degenerate

Agent actions a_i(t) are recovered from the position series:
    a_i(t) = position[t+1] - position[t]

Output (PDF):
    Page 1 : A_t time series (sample run, one panel per payoff)
    Page 2 : Attendance + price stationarity statistics table
    Page 3 : Correlation summary table  rho_agg and mean rho_i vs R_t^w
    Page 4 : Mean rho_i^w vs window size w
"""

import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from datetime import datetime
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller, kpss

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from core.game import Game
from core.game_config import GameConfig
from analysis.cohort_utils import cohort_map_from_spec, summarize_population, format_population_summary
from analysis.plot_utils import create_grid_figure
from analysis.population_spec import PopulationFamilyConfig, build_population_variant
from utils.logger import RunLogger


@dataclass
class BehaviouralCharConfig:
    # --- Mode ---
    mode: Literal["pure", "family"] = "family"

    # --- Population (shared) ---
    N:            int = 300
    memory:       int = 4
    n_strategies: int = 2

    # --- Pure mode ---
    pure_payoffs: List[str] = field(default_factory=lambda: [
        "BinaryMG", "ScaledMG", "DollarGame"
    ])

    # --- Family mode ---
    target_payoff:  str        = "DollarGame"
    non_target_payoff: str     = "BinaryMG"   # makes extension to 3-way splits easier later
    dg_shares: List[float]     = field(default_factory=lambda: [
        0.0, 0.15, 0.30, 0.40, 0.45, 0.47, 0.48, 0.49,
        0.50, 0.51, 0.52, 0.53, 0.55, 0.60, 0.70, 0.85, 1.0
    ])

    # --- Run settings ---
    rounds:       int   = 10_000
    n_runs:       int   = 20
    seed:         int   = 42
    max_workers:  int   = 10
    lambda_:      float = None
    market_maker: bool  = False

    # --- Analysis ---
    w_paper: int       = 4
    windows: List[int] = None

    # --- Output ---
    save_dir: str = "simulation_runs"

    def __post_init__(self):
        if self.windows is None:
            m = self.memory
            self.windows = sorted({1, max(1, m // 2), m, 2 * m})
        if self.w_paper not in self.windows:
            self.windows = sorted(set(self.windows) | {self.w_paper})

    def base_cohorts(self) -> List[dict]:
        """
        Derive BASE_COHORTS from config.
        Equal split by default — vary_payoff_weights rescales from here.
        """
        half = self.N // 2
        remainder = self.N - 2 * half
        return [
            {
                "count":          half + remainder,  # non-target gets any odd agent
                "memory":         self.memory,
                "payoff":         self.non_target_payoff,
                "strategies":     self.n_strategies,
                "position_limit": 0,
                "agent_type":     "strategic",
            },
            {
                "count":          half,
                "memory":         self.memory,
                "payoff":         self.target_payoff,
                "strategies":     self.n_strategies,
                "position_limit": 0,
                "agent_type":     "strategic",
            },
        ]

    @property
    def n_tasks(self) -> int:
        if self.mode == "pure":
            return len(self.pure_payoffs) * self.n_runs
        return len(self.dg_shares) * self.n_runs

    @property
    def label(self) -> str:
        if self.mode == "pure":
            return f"pure_N{self.N}_m{self.memory}_runs{self.n_runs}"
        return (f"family_N{self.N}_m{self.memory}"
                f"_shares{len(self.dg_shares)}_runs{self.n_runs}")

# Config loader
def load_config(path:str) -> BehaviouralCharConfig:
    with open(path, "r") as f:
        data = json.load(f)
        config_data = {k: v for k, v in data.items() if not k.startswith('_')}
        
    return BehaviouralCharConfig(**config_data)

# Plotting helper
def _make_price_panel(colours: dict, adf_pvalues: dict):
    """Return a panel_fn for use with create_grid_figure."""
    def _fn(ax, title, y_data, stat_sum, x_data=None):
        colour = colours.get(title, "#333333")
        adf    = adf_pvalues.get(title, np.nan)
        ax.plot(np.asarray(y_data, float), color=colour, lw=0.5, alpha=0.85)
        ax.set_title(title, fontsize=7, fontweight="bold", color=colour)
        ax.text(
            0.03, 0.97,
            f"ADF $p$={adf:.3f}" if not np.isnan(adf) else "ADF p=n/a",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
        )
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
    return _fn

# Attendance statistics
def attendance_statistics(attendance: np.ndarray, N: int, memory: int) -> Dict:
    """
    Compute summary statistics for the attendance series A_t.

    Parameters
    ----------
    attendance : ndarray (T,)
    N          : number of agents
    memory     : agent memory m (used for AC at lag m)

    Returns
    -------
    dict of scalar statistics
    """
    T     = len(attendance)
    mu    = float(np.mean(attendance))
    sigma = float(np.std(attendance))
    var_N = sigma**2 / N                   # normalised variance (phase diagram y-axis)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p, _, _, _, _ = adfuller(attendance, autolag='AIC')
    except Exception:
        adf_p = np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p , _, _ = kpss(attendance, regression='c', nlags='auto')
    except Exception:
        kpss_p = np.nan

    def ac(lag):
        if T <= lag:
            return np.nan
        x = attendance[:-lag].astype(float)
        y = attendance[lag:].astype(float)
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "mean_A":           mu,
        "sigma_A":          sigma,
        "sigma2_N":         var_N,
        "AC(1)":            ac(1),
        f"AC({memory})":    ac(memory),
        "skew":             float(scipy_stats.skew(attendance)),
        "ex_kurt":          float(scipy_stats.kurtosis(attendance)),
        "sqrt_N":           float(np.sqrt(N)),
        "phase_ratio":      sigma / float(np.sqrt(N)),
        "ADF_p":            float(adf_p),
        "KPSS_p":           float(kpss_p)
    }

def price_statistics(prices: np.ndarray) -> Dict:
    """
    Stationarity tests on the log price series.

    ADF  : null = unit root (non-stationary). p ~ 0 -> stationary.
    KPSS : null = stationary. p > 0.05 -> consistent with stationarity.

    For MG we expect log price to be a random walk (non-stationary):
        ADF fails to reject (p large), KPSS rejects (p small).
    For DG the series is degenerate so tests are undefined.
    """
    valid = prices[np.isfinite(prices) & (prices > 0)]
    if len(valid) < 20:
        return {"ADF_p_price": np.nan, "KPSS_p_price": np.nan}

    log_p = np.log(valid)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            _, adf_p, _, _, _, _ = adfuller(log_p, autolag='AIC')
        except Exception:
            adf_p = np.nan
        try:
            _, kpss_p, _, _ = kpss(log_p, regression='c', nlags='auto')
        except Exception:
            kpss_p = np.nan

    return {"ADF_p_price": float(adf_p), "KPSS_p_price": float(kpss_p)}

# Price return signal construction
def build_price_return_signals(
    prices:  np.ndarray,
    windows: List[int]
) -> Dict[int, np.ndarray]:
    """
    Build log price return signal R_t^w for each window size.

    R_t^w = log(P_{t-1}) - log(P_{t-1-w})   for t > w, else NaN

    This measures the cumulative log return over the w rounds immediately
    preceding round t — the trend that agents could have observed and
    reacted to when choosing their action at t.

    Using price returns rather than the attendance rolling mean T_t^w has
    two advantages:
      1. Economically interpretable: R_t^w > 0 means price rose, < 0 fell.
      2. For the MG, log price is a random walk so R_t^w has non-trivial
         variance, giving the correlation measure actual power.

    Returns
    -------
    dict  {w: ndarray of length T}   where T = len(prices) - 1
    """
    # prices has length T+1 (includes t=0 initial price)
    # actions/attendance have length T
    T      = len(prices) - 1
    log_p  = np.where(prices > 0, np.log(prices), np.nan)

    signals = {}
    for w in windows:
        R = np.full(T, np.nan)
        # At round t (0-indexed), agent observes prices up to P_{t-1}
        # R_t^w = log(P_{t-1}) - log(P_{t-1-w})
        # valid for t >= w+1, i.e. t = w, w+1, ..., T-1 (0-indexed: t >= w)
        for t in range(w, T):
            R[t] = log_p[t] - log_p[t - w]
        signals[w] = R

    return signals

# Aggregate correlation   rho_agg^w = Corr(A_t, R_t^w)
def aggregate_correlation(
    attendance:     np.ndarray,
    price_signals:  Dict,
    windows:        List[int]
) -> pd.DataFrame:
    """
    Compute Pearson correlation between aggregate order flow A_t
    and the log price return signal R_t^w.

    rho_agg > 0  ->  order flow follows recent price rise  (trend-following/herding)
    rho_agg < 0  ->  order flow opposes recent price rise  (mean-reverting)
    """
    rows = []
    for w in windows:
        R     = price_signals[w]
        valid = ~np.isnan(R)
        if valid.sum() < 10:
            continue
        A_v = attendance[valid].astype(float)
        R_v = R[valid]
        if np.std(A_v) < 1e-10 or np.std(R_v) < 1e-10:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = scipy_stats.pearsonr(A_v, R_v)
        rows.append({
            "window":  w,
            "rho_agg": float(rho),
            "p_agg":   float(pval)
        })
    return pd.DataFrame(rows)


# Individual agent correlations   rho_i^w = Corr(a_i(t), R_t^w)
def individual_correlations(
    actions:        np.ndarray,    # (T, N)
    price_signals:  Dict,
    windows:        List[int],
    cohort_ids=None
) -> pd.DataFrame:
    """
    Compute Pearson correlation between each agent's action series
    and the log price return signal R_t^w.

    rho_i > 0  ->  agent buys after price rise   (trend-following)
    rho_i < 0  ->  agent sells after price rise  (mean-reverting/contrarian)
    """
    T, N = actions.shape
    if cohort_ids is None:
        cohort_ids = np.zeros(N, dtype=int)
    rows = []

    for w in windows:
        R     = price_signals[w]
        valid = ~np.isnan(R)
        if valid.sum() < 10:
            continue
        R_v = R[valid]

        for i in range(N):
            a_v = actions[valid, i]
            if np.std(a_v) < 1e-10:
                rho = np.nan
            else:
                rho, _ = scipy_stats.pearsonr(a_v, R_v)
            rows.append({
                "agent_id": i,
                "cohort_id":int(cohort_ids[i]),
                "window":   w,
                "rho_i":    float(rho)
            })

    return pd.DataFrame(rows)

# Run a single game -- return actions and attendance
def run_game_and_extract(
    n_agents:        int,
    rounds:          int,
    seed:            int,
    lambda_:         float = None,
    population_spec: dict = None,
    # original single-cohort parameters — used if population_spec is None
    payoff_name:     str  = None,
    memory:          int  = None,
    n_strategies:    int  = 2,
    position_limit:  int  = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one game and return (actions, attendance, prices).
    Pass either a pre-built population_spec or a single cohort
    actions    : ndarray (rounds, N)  -- position diffs
    attendance : ndarray (rounds,)
    """
    if population_spec is None:
        assert payoff_name is not None and memory is not None
        population_spec = {
            "total": n_agents,
            "cohorts": [{
                "count":          n_agents,
                "memory":         memory,
                "payoff":         payoff_name,
                "strategies":     n_strategies,
                "position_limit": position_limit,
                "agent_type":     "strategic"
            }]
        }

    cfg = GameConfig(
        rounds=rounds,
        lambda_= lambda_ if lambda_ is not None else 1.0 / (n_agents * 50),
        mm=None,
        price=100,
        seed=seed,
        record_agent_series=True
    )

    game    = Game(population_spec=population_spec, cfg=cfg)
    results = game.run()

    position   = results["position"]     # (rounds+1, N)
    attendance = results["Attendance"]   # (rounds,)
    prices     = results["Prices"]
    cohort_ids = results["cohort_ids"]
    cohort_meta= results["cohorts"]
    actions    = np.diff(position, axis=0).astype(float)   # (rounds, N)

    return actions, attendance, prices, cohort_ids, cohort_meta



# Summary helpers
def _agg_summary(data: Dict, windows: List[int]) -> pd.DataFrame:
    """Mean rho_agg and p-value across runs, per window x norm."""
    df   = data["agg_df"]
    rows = []
    for w in windows:
        sub  = df[df["window"] == w]
        rhos = sub["rho_agg"].dropna()
        if len(rhos) < 1:
            continue
        t, p = scipy_stats.ttest_1samp(rhos, 0) if len(rhos) > 1 else (np.nan, np.nan)
        rows.append({
            "window":       w,
            "mean_rho_agg": float(rhos.mean()),
            "std_rho_agg":  float(rhos.std()) if len(rhos) > 1 else 0.0,
            "p_agg":        float(p)
        })
    return pd.DataFrame(rows)


def _ind_summary(data: Dict, windows: List[int]) -> pd.DataFrame:
    """Mean rho_i across agents and runs, per window x norm."""
    df   = data["ind_df"]
    rows = []
    for w in windows:
        sub  = df[df["window"] == w]
        rhos = sub["rho_i"].dropna()
        if len(rhos) < 1:
            continue
        t, p = scipy_stats.ttest_1samp(rhos, 0) if len(rhos) > 1 else (np.nan, np.nan)
        rows.append({
            "window":      w,
            "mean_rho_i": float(rhos.mean()),
            "std_rho_i":  float(rhos.std()) if len(rhos) > 1 else 0.0,
            "n":          len(rhos),
            "p_ind":      float(p)
        })
    return pd.DataFrame(rows)


def _attend_mean(att_stats: List[Dict], memory: int) -> Dict:
    """Average attendance statistics across runs."""
    keys = ["mean_A", "sigma_A", "sigma2_N", "AC(1)",
            f"AC({memory})", "skew", "ex_kurt", "phase_ratio",
            "ADF_p", "KPSS_p", "ADF_p_price", "KPSS_p_price" ]
    out = {}
    for k in keys:
        vals = [s[k] for s in att_stats if k in s and not np.isnan(s.get(k, np.nan))]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out
    

def _cohort_label(cohort, cid: int) -> str:
    """Short human-readable label for one cohort."""
    if getattr(cohort, "agent_type", "strategic") == "noise":
        return f"C{cid}: Noise"
    payoff = getattr(cohort, "payoff", "?")
    mem    = getattr(cohort, "memory", "?")
    return f"C{cid}: {payoff}  m={mem}"


def _ind_summary_by_cohort(data: Dict, windows: List[int]) -> Dict[int, pd.DataFrame]:
    """
    Per-cohort version of _ind_summary.

    Returns
    -------
    dict  {cohort_id (int): summary DataFrame}
        Each DataFrame has the same columns as _ind_summary output.
        In a pure (single-cohort) population there will be exactly one entry.
    """
    df          = data["ind_df"]
    cohort_ids  = sorted(df["cohort_id"].unique())
    by_cohort   = {}
    for cid in cohort_ids:
        sub_data = {"ind_df": df[df["cohort_id"] == cid].copy()}
        by_cohort[cid] = _ind_summary(sub_data, windows)
    return by_cohort

# Table renderer
def _render_table(ax, col_labels, row_labels, cell_data,
                  col_widths=None,
                  title="Attendance Statistics (mean across runs"):
    """Render a clean table on a matplotlib Axes (axis turned off)."""
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    if col_widths is None:
        col_widths = [0.14] + [0.095] * len(col_labels)

    table = ax.table(
        cellText=[[rl] + row for rl, row in zip(row_labels, cell_data)],
        colLabels=[""] + col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="upper center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.9)

    # Header
    for j in range(n_cols + 1):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", weight="bold")

    # Alternating row shading
    for i in range(1, n_rows + 1):
        bg = "#f2f2f2" if i % 2 == 0 else "white"
        for j in range(n_cols + 1):
            table[i, j].set_facecolor(bg)

    return table

# Matplotlib global settings
# Use matplotlib's built-in mathtext engine with STIX fonts — gives proper
# LaTeX-quality rendering without requiring a local LaTeX installation.
plt.rcParams.update({
    "text.usetex":       False,         # use mathtext, not system LaTeX
    "mathtext.fontset":  "stix",        # STIX fonts ≈ Computer Modern look
    "font.family":       "STIXGeneral", # body text to match
})

# Maximum variant panels per page for time-series and rho plots.
# Tables are handled separately with MAX_TABLE_ROWS.
_MAX_PANELS_PER_PAGE = 4
_MAX_RHO_PANELS_PER_PAGE = 6
_MAX_TABLE_ROWS      = 16

# Main plotting function
COLOURS = {
    "Binary MG":   "#2ca02c",
    "Scaled MG":   "#17becf",
    "Dollar Game": "#ff7f0e"
}

_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

def _colour(label: str, idx: int) -> str:
    """Return a fixed colour for known labels or cycle through tab10."""
    return COLOURS.get(label, _CYCLE[idx % len(_CYCLE)])


def _chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def plot_results(
    all_results: Dict,
    cfg: BehaviouralCharConfig,
    save_dir:    str
) -> str:
    """
    Build a multi-page PDF.  Pages are automatically paginated so that no
    figure tries to squeeze more than ``_MAX_PANELS_PER_PAGE`` time-series
    or rho-vs-window panels into a single page.  Table pages are split at
    ``_MAX_TABLE_ROWS`` rows.

    Pages
    -----
    §0  Population summary
    §1  Attendance A_t time series  (one page per ≤4 variants)
    §2  Price P_t time series       (one page per ≤4 variants)
    §3  Attendance statistics table (split every _MAX_TABLE_ROWS rows)
    §4  Correlation summary table   (split every _MAX_TABLE_ROWS rows)
    §5  Mean rho_i vs window w      (one page per ≤4 variants)
    """
    # Use matplotlib's built-in mathtext engine — no external LaTeX needed.
    # STIX gives the closest look to real LaTeX for Greek letters and operators.
    plt.rcParams.update({
        "text.usetex":      False,
        "mathtext.fontset": "stix",
        "font.family":      "STIXGeneral",
    })
    
    os.makedirs(save_dir, exist_ok=True)
    pdf_path  = os.path.join(save_dir, f"behavioural_characterization_{cfg.label}.pdf")
    all_labels   = [k for k in all_results.keys() if not k.startswith("_")]
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}


    with PdfPages(pdf_path) as pdf:

        # Page 0: Experiment configuration summary table
        first = list(all_results.values())[0]
        n_runs  = len(first["attend_stats"])
        n_agents = first["n_agents"]
        rounds  = first["rounds"]
        memory = cfg.memory
        windows = cfg.windows
        
        # Cohort descriptions from cfg
        if cfg.mode == "family":
            cohort_lines = "   |   ".join(
                f"C{i}: {c['payoff']}  m={c['memory']}  s={c['strategies']}"
                for i, c in enumerate(cfg.base_cohorts())
                )
        else:
            cohort_lines = "   |   ".join(
                f"C{i}:  {p}  m={cfg.memory}   s={cfg.n_strategies}"
                for i, p in enumerate(cfg.pure_payoffs)
                )
        
        lam_str =(f"{cfg.lambda_:.2e}" if cfg.lambda_ is not None
                     else f"1/(50*N) = {1.0/(first['n_agents']*50):.2e}")
        
        # Header block above the table
        header = (
            f"EXPERIMENT: BEHAVIOURAL CHARACTERIZATION ANALYSIS\n"
            f"{'=' * 48}\n"
            f"Mode: {cfg.mode}     Sweep: payoff_weights\n"
            f"Cohorts: {cohort_lines}\n"
            f"N: {first['n_agents']}    Memory: m={memory}    "
            f"Strategies: {cfg.n_strategies}     lambda: {lam_str}\n"
            f"Rounds: {rounds}     Runs: {n_runs}\n"
            f"Seed: {first['seed']}     Workers: {cfg.max_workers}\n"
            f"{'=' * 48}"
        )
        
        # Table for front page
        target = cfg.target_payoff
        non_target = cfg.non_target_payoff
        cfg_col_labels = [f"N  {non_target}",  f"N  {target}"]
        cfg_row_labels = []
        cfg_cell_data  = []
        
        for plabel in all_labels:
            data     = all_results[plabel]
            pop_spec = data["population_spec"]
            cohorts  = pop_spec.get("cohorts", [])
            share = data.get("value", np.nan)
            
            n_target = sum(c["count"] for c in cohorts if c.get("payoff") == target)
            n_non_target = sum(c["count"] for c in cohorts if c.get("payoff") == non_target)
            
            cfg_row_labels.append(plabel)
            cfg_cell_data.append([
                str(n_non_target),
                str(n_target),
            ])
        
        
        
        n_rows  = len(cfg_row_labels)
        fig, axes_cfg = plt.subplots(
            2, 1,
            figsize=(10, 3.5 + 0.45 * n_rows),
            gridspec_kw={"height_ratios": [1, 4]}
        )
        
        # Top: header text
        axes_cfg[0].axis("off")
        axes_cfg[0].text(
            0.01, 0.5, header,
            transform=axes_cfg[0].transAxes,
            fontsize=9, va="center", family="monospace"
        )
        
        # Bottom: compact table
        _render_table(
            axes_cfg[1],
            cfg_col_labels,
            cfg_row_labels,
            cfg_cell_data,
            col_widths=[0.20, 0.12, 0.12, 0.12, 0.12],
            title="Population variants"
        )
        
        plt.tight_layout()
        _safe_save(pdf, fig, "header page")

        # Page 1: A_t time series
        for page_labels in _chunks(all_labels, _MAX_PANELS_PER_PAGE):
            n_panels = len(page_labels)
            fig, axes = plt.subplots(
                n_panels, 1,
                figsize=(12, 3.4 * n_panels),
                sharex=False
            )
            axes = np.atleast_1d(axes).flatten()
        

            fig.suptitle(r"Attendance $A_t$ time series  (sample run)",
                         fontsize=13, fontweight="bold")

            for ax, plabel in zip(axes, page_labels):
                attend = all_results[plabel]["sample_attend"]
                colour = _colour(plabel, label_to_idx[plabel])
                T      = len(attend)
    
                ax.plot(attend, color=colour, lw=0.6, alpha=0.85)
                ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    
                # Rolling mean overlay
                cs  = np.concatenate([[0.0], np.cumsum(attend, dtype=float)])
                rm  = np.full(T, np.nan)
                rm[memory:] = (cs[memory:T] - cs[:T - memory]) / memory
                ax.plot(rm, color="black", lw=1.2, alpha=0.6,
                        label=f"Rolling mean  w={memory}")

                ms   = _attend_mean(all_results[plabel]["attend_stats"], memory)
                info = (rf"$\sigma_A=${ms['sigma_A']:.2f}   "
                        rf"$\sigma^2/N=${ms['sigma2_N']:.3f}   "
                        rf"$\mathrm{{AC}}(1)=$"f"{ms['AC(1)']:+.3f}")
                ax.text(0.01, 0.97, info, transform=ax.transAxes,
                        fontsize=8, va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    
                ax.set_ylabel(r"$A_t$", fontsize=10)
                ax.set_title(plabel, fontsize=10, fontweight="bold", color=colour)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(alpha=0.25)

            axes[-1].set_xlabel(r"Round $t$", fontsize=10)
            plt.tight_layout()
            _safe_save(pdf, fig, "attendance time series")

        # Paragraph 2: P_t time series — single grid page (appendix)
        fig = create_grid_figure(
            items    = [all_results[l]["sample_price"] for l in all_labels],
            plot_fn  = _make_price_panel(
                colours     = {l: _colour(l, label_to_idx[l]) for l in all_labels},
                adf_pvalues = {l: _attend_mean(all_results[l]["attend_stats"], memory)
                                  .get("ADF_p_price", np.nan)
                               for l in all_labels},
            ),
            titles   = all_labels,
            suptitle = r"Price $P_t$ time series  (sample run)",
            xlabel   = r"Round $t$",
            ylabel   = r"$P_t$",
            figsize_per_panel = (4.5, 3.0),
        )
        _safe_save(pdf, fig, "price series")
        
        # Paragraph 2a: Normalised Variance and AC1 vs dollar game share
        if cfg.mode == "family":
            sigma2 = []
            ac1 = []
            shares = []
            for plabel in all_labels:
                ms = _attend_mean(all_results[plabel]["attend_stats"], memory)
                sigma2.append(ms.get("sigma2_N", np.nan))
                ac1.append(ms.get("AC(1)", np.nan))
                shares.append(all_results[plabel].get("value", np.nan))
            
            sigma2 = np.array(sigma2)
            ac1    = np.array(ac1)
            shares = np.array(shares)
            
            fig, ax1 = plt.subplots(figsize=(8,5))
            
            # Left axis: sigma^2/N
            colour_sigma = "#1f77b4"
            ax1.plot(shares, sigma2, color=colour_sigma, marker="o", linewidth=1.8,
                     label=r"$\sigma^2_A / N$")
            ax1.axhline(1.0, color=colour_sigma, lw=0.8, ls="--", alpha=0.5,
                        label=r"$\sigma^2/N = 1$ (disorder boundary)")
            ax1.set_xlabel("DollarGame share", fontsize=11)
            ax1.set_ylabel(r"$\sigma^2_A / N$", fontsize=11, color=colour_sigma)
            ax1.tick_params(axis="y", labelcolor=colour_sigma)
            ax1.set_xlim(-0.02, 1.02)
            
            # Right axis: AC(1)
            colour_ac = "#d62728"
            ax2 = ax1.twinx()
            ax2.plot(shares, ac1, color=colour_ac, marker="s", linewidth=1.8,
                     ls="--", label="AC(1)")
            ax2.axhline(0.0, color=colour_ac, lw=0.8, ls=":", alpha=0.5)
            ax2.set_ylabel("AC(1)", fontsize=11, color=colour_ac)
            ax2.tick_params(axis="y", labelcolor=colour_ac)
            
            # Combined legend from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       fontsize=8, loc="upper right")
            
            fig.suptitle(
                r"Phase diagram: $\sigma^2_A/N$ and AC(1) vs DollarGame share",
                fontsize=12, fontweight="bold"
            )
            plt.tight_layout()
            _safe_save(pdf, fig, "phase")
            
    # Paragraph 2c: Cohort correlation plot — rho vs share, split by cohort
        # One line per window per cohort. Only produced in family mode.
        if cfg.mode == "family":
            COLOUR_DG = "#d62728"
            COLOUR_MG = "#1f77b4"
            ALPHA_W   = [1.0, 0.75, 0.5, 0.3]  # darkest = w=1, lightest = w=8
        
            # Collect rho and SEM per (cohort, window) across all share values
            dg_rho  = {w: [] for w in windows}
            mg_rho  = {w: [] for w in windows}
            dg_sem  = {w: [] for w in windows}
            mg_sem  = {w: [] for w in windows}
            x_shares = []
        
            for plabel in all_labels:
                data        = all_results[plabel]
                cohort_meta = data.get("cohort_meta", [])
                by_cohort   = _ind_summary_by_cohort(data, windows)
                x_shares.append(data.get("value", np.nan))
        
                dg_cid = next((i for i, c in enumerate(cohort_meta)
                               if getattr(c, "payoff", "") == cfg.target_payoff), None)
                mg_cid = next((i for i, c in enumerate(cohort_meta)
                               if getattr(c, "payoff", "") == cfg.non_target_payoff), None)
        
                def _extract(cid, w):
                    if cid is None or cid not in by_cohort:
                        return np.nan, np.nan
                    row = by_cohort[cid]
                    row = row[row["window"] == w]
                    if row.empty:
                        return np.nan, np.nan
                    rho = float(row.iloc[0]["mean_rho_i"])
                    sem = float(row.iloc[0]["std_rho_i"]) / np.sqrt(
                          max(float(row.iloc[0]["n"]), 1))
                    return rho, sem
        
                for w in windows:
                    r, s = _extract(dg_cid, w)
                    dg_rho[w].append(r); dg_sem[w].append(s)
                    r, s = _extract(mg_cid, w)
                    mg_rho[w].append(r); mg_sem[w].append(s)
        
            x = np.array(x_shares)
        
            fig, ax = plt.subplots(figsize=(9, 5))
        
            for i, w in enumerate(sorted(windows)):
                alpha = ALPHA_W[i] if i < len(ALPHA_W) else 0.3
                lw    = 1.8 if w == cfg.w_paper else 1.0
                label_dg = (rf"$\bar{{\rho}}_{{DG}}$  $w={w}$"
                            + (" (*)" if w == cfg.w_paper else ""))
                label_mg = (rf"$\bar{{\rho}}_{{MG}}$  $w={w}$"
                            + (" (*)" if w == cfg.w_paper else ""))
        
                rho_dg = np.array(dg_rho[w])
                sem_dg = np.array(dg_sem[w])
                rho_mg = np.array(mg_rho[w])
                sem_mg = np.array(mg_sem[w])
        
                # DG cohort — solid
                valid = ~np.isnan(rho_dg)
                ax.plot(x[valid], rho_dg[valid],
                        color=COLOUR_DG, lw=lw, alpha=alpha,
                        marker="o", markersize=3, label=label_dg)
                ax.fill_between(x[valid],
                                rho_dg[valid] - 1.96 * sem_dg[valid],
                                rho_dg[valid] + 1.96 * sem_dg[valid],
                                color=COLOUR_DG, alpha=alpha * 0.15)
        
                # MG cohort — dashed
                valid = ~np.isnan(rho_mg)
                ax.plot(x[valid], rho_mg[valid],
                        color=COLOUR_MG, lw=lw, alpha=alpha, ls="--",
                        marker="s", markersize=3, label=label_mg)
                ax.fill_between(x[valid],
                                rho_mg[valid] - 1.96 * sem_mg[valid],
                                rho_mg[valid] + 1.96 * sem_mg[valid],
                                color=COLOUR_MG, alpha=alpha * 0.15)
        
            ax.axhline(0,    color="black", lw=0.8, ls="--", alpha=0.4)
            ax.axhline( 0.05, color="grey", lw=0.5, ls=":",  alpha=0.5)
            ax.axhline(-0.05, color="grey", lw=0.5, ls=":",  alpha=0.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_xlabel(f"{cfg.target_payoff} share", fontsize=11)
            ax.set_ylabel(r"Mean $\bar{\rho}_i^w$", fontsize=11)
            ax.legend(fontsize=7, ncol=2, loc="upper left")
            ax.grid(alpha=0.25)
        
            fig.suptitle(
                r"Cohort behavioural characterisation: $\bar{\rho}_i^w$ vs share"
                "\n"
                rf"Solid: {cfg.target_payoff} (trend-following)   "
                rf"Dashed: {cfg.non_target_payoff} (mean-reverting)   "
                r"(*) = paper window",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            _safe_save(pdf, fig, "cohort correlations")
        
        # Pure mode summary page
        if cfg.mode == "pure":
            COLOURS_PURE = {
                "BinaryMG":   "#1f77b4",
                "ScaledMG":   "#2ca02c",
                "DollarGame": "#d62728",
            }
        
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            fig.suptitle(
                "Pure Population Behavioural Characterisation\n"
                rf"N={cfg.N}  m={cfg.memory}  s={cfg.n_strategies}  "
                rf"rounds={cfg.rounds}  runs={cfg.n_runs}",
                fontsize=12, fontweight="bold"
            )
        
            # Panel (0,0): A_t time series — one line per payoff, sample run
            ax = axes[0, 0]
            for plabel in all_labels:
                attend = all_results[plabel]["sample_attend"]
                colour = COLOURS_PURE.get(plabel, "#333333")
                ax.plot(attend, color=colour, lw=0.5, alpha=0.7, label=plabel)
            ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
            ax.set_title(r"Attendance $A_t$ (sample run)", fontsize=10)
            ax.set_xlabel(r"Round $t$", fontsize=9)
            ax.set_ylabel(r"$A_t$", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        
            # Panel (0,1): rho_agg vs window — one line per payoff
            ax = axes[0, 1]
            for plabel in all_labels:
                colour  = COLOURS_PURE.get(plabel, "#333333")
                agg_smr = _agg_summary(all_results[plabel], windows)
                sub     = agg_smr.sort_values("window")
                ax.plot(sub["window"], sub["mean_rho_agg"],
                        color=colour, marker="o", lw=1.6, label=plabel)
                ax.fill_between(
                    sub["window"],
                    sub["mean_rho_agg"] - sub["std_rho_agg"],
                    sub["mean_rho_agg"] + sub["std_rho_agg"],
                    color=colour, alpha=0.12)
            ax.axhline(0,     color="black", lw=0.8, ls="--", alpha=0.4)
            ax.axhline( 0.05, color="grey",  lw=0.5, ls=":",  alpha=0.5)
            ax.axhline(-0.05, color="grey",  lw=0.5, ls=":",  alpha=0.5)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xticks(windows)
            ax.set_title(r"Aggregate correlation $\rho_{agg}^w$", fontsize=10)
            ax.set_xlabel(r"Window $w$", fontsize=9)
            ax.set_ylabel(r"$\rho_{agg}^w$", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        
            # Panel (1,0): rho_i boxplot — distribution across agents, at w_paper
            ax = axes[1, 0]
            box_data   = []
            box_labels = []
            box_colours = []
            for plabel in all_labels:
                ind_df = all_results[plabel]["ind_df"]
                rhos   = ind_df[ind_df["window"] == cfg.w_paper]["rho_i"].dropna().values
                box_data.append(rhos)
                box_labels.append(plabel)
                box_colours.append(COLOURS_PURE.get(plabel, "#333333"))
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                            medianprops=dict(color="black", lw=1.5),
                            whiskerprops=dict(lw=0.8),
                            flierprops=dict(marker=".", markersize=2, alpha=0.4))
            for patch, colour in zip(bp["boxes"], box_colours):
                patch.set_facecolor(colour)
                patch.set_alpha(0.6)
            ax.axhline(0,     color="black", lw=0.8, ls="--", alpha=0.4)
            ax.axhline( 0.05, color="grey",  lw=0.5, ls=":",  alpha=0.5)
            ax.axhline(-0.05, color="grey",  lw=0.5, ls=":",  alpha=0.5)
            ax.set_title(
                rf"Individual $\rho_i^{{w={cfg.w_paper}}}$ distribution",
                fontsize=10)
            ax.set_ylabel(r"$\rho_i^w$", fontsize=9)
            ax.grid(alpha=0.25, axis="y")
        
            # Panel (1,1): summary statistics table
            ax = axes[1, 1]
            ax.axis("off")
            col_labels = [r"$\sigma^2/N$", "AC(1)", r"$\rho_{agg}$", r"$\bar{\rho}_i$"]
            row_labels = []
            cell_data  = []
            for plabel in all_labels:
                ms      = _attend_mean(all_results[plabel]["attend_stats"], memory)
                agg_smr = _agg_summary(all_results[plabel], windows)
                ind_smr = _ind_summary(all_results[plabel], windows)
                row_w   = agg_smr[agg_smr["window"] == cfg.w_paper]
                ind_w   = ind_smr[ind_smr["window"] == cfg.w_paper]
                ra      = float(row_w.iloc[0]["mean_rho_agg"]) if not row_w.empty else np.nan
                ri      = float(ind_w.iloc[0]["mean_rho_i"])   if not ind_w.empty else np.nan
                row_labels.append(plabel)
                cell_data.append([
                    f"{ms.get('sigma2_N', np.nan):.3f}",
                    f"{ms.get('AC(1)',    np.nan):+.3f}",
                    f"{ra:+.3f}" if not np.isnan(ra) else "--",
                    f"{ri:+.3f}" if not np.isnan(ri) else "--",
                ])
            table = ax.table(
                cellText=[[rl] + row for rl, row in zip(row_labels, cell_data)],
                colLabels=[""] + col_labels,
                loc="center", cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.2)
            for j in range(len(col_labels) + 1):
                table[0, j].set_facecolor("#4472C4")
                table[0, j].set_text_props(color="white", weight="bold")
            for i in range(1, len(row_labels) + 1):
                colour = COLOURS_PURE.get(row_labels[i-1], "#333333")
                table[i, 0].set_text_props(color=colour, weight="bold")
            ax.set_title(
                rf"Summary  ($w={cfg.w_paper}$, mean across runs)",
                fontsize=10, fontweight="bold", pad=8)
        
            plt.tight_layout()
            _safe_save(pdf, fig, "pure population summary")
        
        # Paragraph 3: Combined summary table — one row per variant, w=4

        summary_col_labels = [
            r"$\sigma^2_A/N$",
            r"$\sigma_A/\sqrt{N}$",
            r"$\mathrm{AC}(1)$",
            "Ex. Kurt",
            rf"$\bar{{\rho}}^{{w={cfg.w_paper}}}_{{DG}}$",
            rf"$\bar{{\rho}}^{{w={cfg.w_paper}}}_{{MG}}$",
            "Character",
        ]

        summary_row_labels = []
        summary_cell_data = []
        
        for plabel in all_labels:
            data = all_results[plabel]
            ms = _attend_mean(data["attend_stats"], memory)
            by_cohort = _ind_summary_by_cohort(data, cfg.windows)
            cohort_meta = data.get("cohort_meta", [])
            
            # Identify DG and MG cohort IDs from metadata
            dg_cid = next((i for i, c in enumerate(cohort_meta)
                          if getattr(c, "payoff", "") == "DollarGame"), None)
            mg_cid = next((i for i, c in enumerate(cohort_meta)
                          if getattr(c, "payoff", "") in ("ScaledMG", "BinaryMG")), None)
            
            def _rho(cid):
                if cid is None or cid not in by_cohort:
                    return np.nan, np.nan
                row = by_cohort[cid]
                row = row[row["window"] == cfg.w_paper]
                if row.empty:
                    return np.nan, np.nan
                return float(row.iloc[0]["mean_rho_i"]), float(row.iloc[0]["p_ind"])
            
            rho_dg, p_dg = _rho(dg_cid)
            rho_mg, p_mg = _rho(mg_cid)
            
            sigma2_N   = ms.get("sigma2_N",    np.nan)
            phase      = ms.get("phase_ratio", np.nan)
            ac1_val    = ms.get("AC(1)",       np.nan)
            ex_kurt    = ms.get("ex_kurt",     np.nan)
            
            if ac1_val > 0.3:
                mkt_char = "Herding"
            elif sigma2_N < 0.5:
                mkt_char = "Anti-coord."
            elif sigma2_N > 1.5:
                mkt_char = "Turbulent"    # high variance, low AC — inefficient MG
            else:
                mkt_char = "Transition"
           
            summary_row_labels.append(plabel)
            summary_cell_data.append([
                f"{sigma2_N:.4f}",
                f"{phase:.3f}",
                f"{ac1_val:+.3f}",
                f"{ex_kurt:+.2f}",
                f"{rho_dg:+.3f}" if not np.isnan(rho_dg) else "--",
                f"{rho_mg:+.3f}" if not np.isnan(rho_mg) else "--",
                mkt_char,
            ])

        n_rows = len(summary_row_labels)
        fig, ax = plt.subplots(figsize=(13, 2.2 + 0.75 * n_rows))
        _render_table(
            ax,
            summary_col_labels,
            summary_row_labels,
            summary_cell_data,
            col_widths=[0.10] + [0.09] * len(summary_col_labels),
            title="Summary Statistics - mean across runs   "
                  rf"($\bar{{\rho}}$ at window $w = {cfg.w_paper}$)",
        )
        fig.text(
            0.5, 0.02,
            r"$\sigma^2/N < 1$: anti-coordinated (MG)   "
            r"$|\quad \approx 1$: disordered   "
            r"$|\quad > 1$: coordinated / herding (DG)"
            r"$|\quad \bar{\rho} > 0$: trend-following   "
            r"$|\quad \bar{\rho} < 0$: mean-reverting",
            ha="center", fontsize=8, style="italic"
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        _safe_save(pdf, fig, "combined summary table")

        # Paragraph 4: Correlation table with per cohort average individual correlations
        # Cohorts are labelled a, b, c, … for compactness in the header;
        # a legend at the foot of each page maps those letters to payoff/memory.
        _LETTERS = "abcdefghijklmnopqrstuvwxyz"
        
        max_cohorts = max(
            len(all_results[lbl].get("cohort_meta",[]))
            for lbl in all_labels
        )
        
        def char(rho, p):
            if np.isnan(rho) or np.isnan(p):
                return "?"
            if abs(rho) > 0.05 and p < 0.05:
                return "MR" if rho < 0 else "TF"
            return "~0"
        
        
        corr_col_labels = [
            r"$\rho_\mathrm{agg}$",
            r"$p_\mathrm{agg}$",
            r"$\bar{\rho}_i$ (pool)",
            r"$p$ (pool)",
            "Char (agg/pool)",
        ]

        for ci in range(max_cohorts):
            letter = _LETTERS[ci]
            corr_col_labels += [
                rf"$\bar{{\rho}}_i$ ({letter})",
                rf"$p$ ({letter})",
                rf"Char ({letter})",
            ]
            
        all_row_labels_cor = []
        all_cell_data_cor  = []
        cohort_legend_lines = {}
        
        for plabel in all_labels:
            data    = all_results[plabel]
            cohort_meta = data.get("cohort_meta", [])
            agg_smr = _agg_summary(data, windows)
            ind_smr = _ind_summary(data, windows)
            by_cohort   = _ind_summary_by_cohort(data, windows)
            cids        = sorted(by_cohort.keys())
            
            # Build legend entries for cohorts in this variant
            for cid, c in enumerate(cohort_meta):
                letter = _LETTERS[cid]
                desc   = _cohort_label(c, cid)
                # Record the most descriptive entry seen for this letter
                key = f"{plabel}|{letter}"
                cohort_legend_lines[key] = f"({letter}) {desc}"

            for w in windows:
                ar = agg_smr[agg_smr["window"] == w]
                ir = ind_smr[ind_smr["window"] == w]

                ra = ar.iloc[0]["mean_rho_agg"] if not ar.empty else np.nan
                pa = ar.iloc[0]["p_agg"]        if not ar.empty else np.nan
                ri = ir.iloc[0]["mean_rho_i"]   if not ir.empty else np.nan
                pi = ir.iloc[0]["p_ind"]        if not ir.empty else np.nan

                row_cells = [
                    f"{ra:+.3f}" if not np.isnan(ra) else "--",
                    f"{pa:.2e}"  if not np.isnan(pa) else "--",
                    f"{ri:+.3f}" if not np.isnan(ri) else "--",
                    f"{pi:.2e}"  if not np.isnan(pi) else "--",
                    f"agg:{char(ra,pa)} / pool:{char(ri,pi)}",
                ]

                for cid in range(max_cohorts):
                    if cid in by_cohort:
                        c_smr = by_cohort[cid]
                        crow  = c_smr[c_smr["window"] == w]
                        rc    = crow.iloc[0]["mean_rho_i"] if not crow.empty else np.nan
                        pc    = crow.iloc[0]["p_ind"]      if not crow.empty else np.nan
                        row_cells += [
                            f"{rc:+.3f}" if not np.isnan(rc) else "--",
                            f"{pc:.2e}"  if not np.isnan(pc) else "--",
                            char(rc, pc),
                        ]
                    else:
                        row_cells += ["--", "--", "--"]

                all_row_labels_cor.append(rf"{plabel}  $w={w}$")
                all_cell_data_cor.append(row_cells)

        # Dynamic column widths: row label + fixed cols + 3 cols per cohort
        fixed_cw  = [0.1, 0.07, 0.07, 0.08, 0.07, 0.10]
        cohort_cw = [0.07, 0.07, 0.06] * max_cohorts
        corr_col_widths = fixed_cw + cohort_cw

        # Legend string: unique descriptions across all variants
        # Group by letter: collect all unique descriptions per letter
        letter_descs: Dict[str, set] = {}
        for key, desc in cohort_legend_lines.items():
            letter = key.split("|")[1]
            letter_descs.setdefault(letter, set()).add(desc)
        legend_str = "   |   ".join(
            ",  ".join(sorted(descs))
            for letter, descs in sorted(letter_descs.items())
        )

        row_chunks = list(_chunks(list(range(len(all_row_labels_cor))), _MAX_TABLE_ROWS))
        for chunk_idx, row_idxs in enumerate(row_chunks):
            chunk_rows  = [all_row_labels_cor[i] for i in row_idxs]
            chunk_cells = [all_cell_data_cor[i]   for i in row_idxs]
            n_rows      = len(chunk_rows)
            # Wider page when there are many cohort columns
            fig_w = min(13 + 2.4 * max_cohorts, 22)
            fig, ax = plt.subplots(figsize=(fig_w, 2.2 + 0.95 * n_rows))
            page_tag = (f"  (page {chunk_idx+1}/{len(row_chunks)})"
                        if len(row_chunks) > 1 else "")
            fig.suptitle(
                r"Correlation with log price return   "
                r"$R_t^w = \log P_{t-1} - \log P_{t-1-w}$" + "\n"
                r"$\rho_\mathrm{agg} = \mathrm{Corr}(A_t,\,R_t^w)$"
                r"$\quad|\quad$"
                r"$\bar{\rho}_i$: mean individual correlation"
                + page_tag,
                fontsize=12, fontweight="bold"
            )
            _render_table(
                ax, corr_col_labels, chunk_rows, chunk_cells,
                col_widths=corr_col_widths,
                title=r"$R_t^w = \log P_{t-1} - \log P_{t-1-w}$",
            )
            # Two-line footnote: interpretation key + cohort legend
            fig.text(
                0.5, 0.03,
                r"$\rho > 0$: trend-following   "
                r"$|\quad \rho < 0$: mean-reverting   "
                r"$|\quad$ ~0: neutral  ($|\rho| \leq 0.05$ or $p \geq 0.05$)",
                ha="center", fontsize=7.5, style="italic"
            )
            fig.text(
                0.5, 0.005,
                "Cohort key:  " + legend_str,
                ha="center", fontsize=9, style="italic"
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            _safe_save(pdf, fig, "correlation table")

        # Paragraph 5: Mean rho_i^w vs window w split by cohort
        RHO_NCOLS = 3
        COLOUR_DG = "#d62728"   # red  — trend-following
        COLOUR_MG = "#1f77b4"   # blue — mean-reverting

        for page_idx, page_labels in enumerate(_chunks(all_labels, _MAX_RHO_PANELS_PER_PAGE)):
            n_panels = len(page_labels)
            ncols    = min(RHO_NCOLS, n_panels)
            nrows    = -(-n_panels // ncols)

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(5.2 * ncols, 4.8 * nrows),
                sharey=True,
                squeeze=False,
            )
            ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
            for spare in ax_flat[n_panels:]:
                spare.set_visible(False)

            fig.suptitle(
                r"Mean individual correlation $\bar{\rho}_i^w$ vs window $w$"
                "\n"
                r"$\rho_i^w = \mathrm{Corr}(a_i(t),\,R_t^w)$"
                rf"  |  solid: {cfg.target_payoff}   dashed: {cfg.non_target_payoff}",
                fontsize=11, fontweight="bold"
            )

            for col_pos, (ax, plabel) in enumerate(zip(ax_flat, page_labels)):
                data        = all_results[plabel]
                cohort_meta = data.get("cohort_meta", [])
                by_cohort   = _ind_summary_by_cohort(data, windows)

                # Identify cohort IDs from metadata
                dg_cid = next((i for i, c in enumerate(cohort_meta)
                               if getattr(c, "payoff", "") == cfg.target_payoff), None)
                mg_cid = next((i for i, c in enumerate(cohort_meta)
                               if getattr(c, "payoff", "") == cfg.non_target_payoff), None)

                def _plot_cohort(cid, colour, ls, label):
                    if cid is None or cid not in by_cohort:
                        return
                    smr = by_cohort[cid].sort_values("window")
                    sem = smr["std_rho_i"] / np.sqrt(smr["n"].clip(lower=1))
                    ax.errorbar(
                        smr["window"], smr["mean_rho_i"],
                        yerr=sem,
                        color=colour, marker="o", ls=ls,
                        capsize=3, linewidth=1.6, label=label
                    )

                _plot_cohort(dg_cid, COLOUR_DG, "-",  cfg.target_payoff)
                _plot_cohort(mg_cid, COLOUR_MG, "--", cfg.non_target_payoff)

                ax.axhline(0,     color="black", lw=0.8, ls="--", alpha=0.4)
                ax.axhline( 0.05, color="grey",  lw=0.5, ls=":",  alpha=0.4)
                ax.axhline(-0.05, color="grey",  lw=0.5, ls=":",  alpha=0.4)

                colour = _colour(plabel, label_to_idx[plabel])
                ax.set_title(plabel, fontsize=10, fontweight="bold", color=colour)
                ax.set_xlabel(r"Window $w$", fontsize=9)
                if col_pos % ncols == 0:
                    ax.set_ylabel(r"Mean $\bar{\rho}_i^w$", fontsize=9)
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.set_xticks(windows)
                ax.grid(alpha=0.3)

                ms = _attend_mean(data["attend_stats"], memory)
                ax.text(
                    0.97, 0.04,
                    r"$\sigma_A/\sqrt{N}=$" + f" {ms['phase_ratio']:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
                )

                # Legend only on first panel
                if col_pos == 0:
                    ax.legend(fontsize=7, loc="upper left")

            plt.tight_layout()
            _safe_save(pdf, fig, "mean rho by cohort graphs")

    print(f"\n  PDF saved: {pdf_path}")
    return pdf_path

def save_paper_figures(
    all_results: Dict,
    cfg: BehaviouralCharConfig,
    save_dir: str,
) -> None:
    """
    Save publication-quality standalone figures for inclusion in LaTeX.

    Figures produced:
      fig_phase_diagram.pdf       — sigma^2/N and AC(1) vs DG share
      fig_cohort_correlation.pdf  — rho_DG and rho_MG vs DG share (w=w_paper only)
    """
    if cfg.mode != "family":
        return

    plt.rcParams.update({
        "text.usetex":      False,
        "mathtext.fontset": "stix",
        "font.family":      "STIXGeneral",
        "axes.labelsize":   10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  8,
    })

    os.makedirs(save_dir, exist_ok=True)
    all_labels = [k for k in all_results if not k.startswith("_")]
    memory     = cfg.memory

    # Collect shares and stats
    shares, sigma2, ac1_vals = [], [], []
    for plabel in all_labels:
        ms = _attend_mean(all_results[plabel]["attend_stats"], memory)
        shares.append(all_results[plabel].get("value", np.nan))
        sigma2.append(ms.get("sigma2_N", np.nan))
        ac1_vals.append(ms.get("AC(1)", np.nan))

    shares    = np.array(shares)
    sigma2    = np.array(sigma2)
    ac1_vals  = np.array(ac1_vals)

    # ------------------------------------------------------------------ #
    # Figure 1: Phase diagram                                              #
    # ------------------------------------------------------------------ #
    fig, ax1 = plt.subplots(figsize=(7, 4))

    c_sigma = "#1f77b4"
    c_ac    = "#d62728"

    ax1.plot(shares, sigma2, color=c_sigma, marker="o",
             markersize=4, linewidth=1.6, label=r"$\sigma^2_A/N$")
    ax1.axhline(1.0, color=c_sigma, lw=0.9, ls="--", alpha=0.5,
                label=r"$\sigma^2_A/N = 1$")
    ax1.set_xlabel("DollarGame share", labelpad=4)
    ax1.set_ylabel(r"$\sigma^2_A/N$", color=c_sigma, labelpad=4)
    ax1.tick_params(axis="y", labelcolor=c_sigma)
    ax1.set_xlim(-0.02, 1.02)
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(shares, ac1_vals, color=c_ac, marker="s",
             markersize=4, linewidth=1.6, ls="--", label="AC(1)")
    ax2.axhline(0.0, color=c_ac, lw=0.8, ls=":", alpha=0.4)
    ax2.set_ylabel("AC(1)", color=c_ac, labelpad=4)
    ax2.tick_params(axis="y", labelcolor=c_ac)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="upper right", framealpha=0.85)

    fig.tight_layout()
    path1 = os.path.join(save_dir, "fig_phase_diagram.pdf")
    fig.savefig(path1, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ------------------------------------------------------------------ #
    # Figure 2: Cohort correlation (w_paper only, with 95% CI band)       #
    # ------------------------------------------------------------------ #
    COLOUR_DG = "#d62728"
    COLOUR_MG = "#1f77b4"

    rho_dg, rho_mg = [], []
    sem_dg, sem_mg = [], []

    for plabel in all_labels:
        data        = all_results[plabel]
        cohort_meta = data.get("cohort_meta", [])
        by_cohort   = _ind_summary_by_cohort(data, cfg.windows)

        dg_cid = next((i for i, c in enumerate(cohort_meta)
                       if getattr(c, "payoff", "") == cfg.target_payoff), None)
        mg_cid = next((i for i, c in enumerate(cohort_meta)
                       if getattr(c, "payoff", "") == cfg.non_target_payoff), None)

        def _get(cid):
            if cid is None or cid not in by_cohort:
                return np.nan, np.nan
            row = by_cohort[cid][by_cohort[cid]["window"] == cfg.w_paper]
            if row.empty:
                return np.nan, np.nan
            rho = float(row.iloc[0]["mean_rho_i"])
            sem = float(row.iloc[0]["std_rho_i"]) / np.sqrt(
                  max(float(row.iloc[0]["n"]), 1))
            return rho, sem

        r, s = _get(dg_cid); rho_dg.append(r); sem_dg.append(s)
        r, s = _get(mg_cid); rho_mg.append(r); sem_mg.append(s)

    rho_dg = np.array(rho_dg); sem_dg = np.array(sem_dg)
    rho_mg = np.array(rho_mg); sem_mg = np.array(sem_mg)

    fig, ax = plt.subplots(figsize=(7, 4))

    valid = ~np.isnan(rho_dg)
    ax.plot(shares[valid], rho_dg[valid], color=COLOUR_DG,
            marker="o", markersize=4, linewidth=1.6,
            label=rf"{cfg.target_payoff}  (trend-following)")
    ax.fill_between(shares[valid],
                    rho_dg[valid] - 1.96 * sem_dg[valid],
                    rho_dg[valid] + 1.96 * sem_dg[valid],
                    color=COLOUR_DG, alpha=0.12)

    valid = ~np.isnan(rho_mg)
    ax.plot(shares[valid], rho_mg[valid], color=COLOUR_MG,
            marker="s", markersize=4, linewidth=1.6, ls="--",
            label=rf"{cfg.non_target_payoff}  (mean-reverting)")
    ax.fill_between(shares[valid],
                    rho_mg[valid] - 1.96 * sem_mg[valid],
                    rho_mg[valid] + 1.96 * sem_mg[valid],
                    color=COLOUR_MG, alpha=0.12)

    ax.axhline(0,     color="black", lw=0.8, ls="--", alpha=0.4)
    ax.axhline( 0.05, color="grey",  lw=0.5, ls=":",  alpha=0.5)
    ax.axhline(-0.05, color="grey",  lw=0.5, ls=":",  alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"{cfg.target_payoff} share", labelpad=4)
    ax.set_ylabel(rf"$\bar{{\rho}}_i^{{w={cfg.w_paper}}}$", labelpad=4)
    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    path2 = os.path.join(save_dir, "fig_cohort_correlation.pdf")
    fig.savefig(path2, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}")

def save_data(all_results: Dict, cfg: BehaviouralCharConfig, save_dir: str) -> None:
    """
    Save all results to a single Excel workbook with four sheets:
      - summary        : one row per variant (attendance stats + rho at w_paper)
      - ind_rho        : all individual correlations stacked (label column added)
      - agg_rho        : all aggregate correlations stacked (label column added)
      - config         : experiment configuration key-value pairs
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"results_{cfg.label}.xlsx")

    labels = [k for k in all_results if not k.startswith("_")]

    # --- Sheet 1: summary ---
    summary_rows = []
    for plabel in labels:
        data        = all_results[plabel]
        ms          = _attend_mean(data["attend_stats"], cfg.memory)
        by_cohort   = _ind_summary_by_cohort(data, cfg.windows)
        cohort_meta = data.get("cohort_meta", [])

        def _rho_at_w(cid):
            if cid is None or cid not in by_cohort:
                return np.nan, np.nan
            row = by_cohort[cid][by_cohort[cid]["window"] == cfg.w_paper]
            if row.empty:
                return np.nan, np.nan
            return float(row.iloc[0]["mean_rho_i"]), float(row.iloc[0]["p_ind"])

        dg_cid = next((i for i, c in enumerate(cohort_meta)
                       if getattr(c, "payoff", "") == "DollarGame"), None)
        mg_cid = next((i for i, c in enumerate(cohort_meta)
                       if getattr(c, "payoff", "") in ("BinaryMG", "ScaledMG")), None)

        rho_dg, p_dg = _rho_at_w(dg_cid)
        rho_mg, p_mg = _rho_at_w(mg_cid)

        summary_rows.append({
            "label":       plabel,
            "value":       data.get("value"),
            "n_agents":    data.get("n_agents"),
            "sigma2_N":    ms.get("sigma2_N"),
            "phase_ratio": ms.get("phase_ratio"),
            "AC1":         ms.get("AC(1)"),
            "ex_kurt":     ms.get("ex_kurt"),
            "rho_DG":      rho_dg,
            "p_DG":        p_dg,
            "rho_MG":      rho_mg,
            "p_MG":        p_mg,
            "ADF_p":       ms.get("ADF_p"),
            "KPSS_p":      ms.get("KPSS_p"),
        })
    df_summary = pd.DataFrame(summary_rows)

    # --- Sheet 2: ind_rho (stacked) ---
    ind_frames = []
    for plabel in labels:
        df = all_results[plabel]["ind_df"].copy()
        df.insert(0, "label", plabel)
        ind_frames.append(df)
    df_ind = pd.concat(ind_frames, ignore_index=True)

    # --- Sheet 3: agg_rho (stacked) ---
    agg_frames = []
    for plabel in labels:
        df = all_results[plabel]["agg_df"].copy()
        df.insert(0, "label", plabel)
        agg_frames.append(df)
    df_agg = pd.concat(agg_frames, ignore_index=True)

    # --- Sheet 4: config ---
    from dataclasses import asdict
    cfg_rows = [{"parameter": k, "value": str(v)}
                for k, v in asdict(cfg).items()]
    df_cfg = pd.DataFrame(cfg_rows)

    # --- Write ---
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary",  index=False)
        df_ind.to_excel(    writer, sheet_name="ind_rho",  index=False)
        df_agg.to_excel(    writer, sheet_name="agg_rho",  index=False)
        df_cfg.to_excel(    writer, sheet_name="config",   index=False)

    print(f"  Results saved: {path}")

def _run_single_game(args):
    """One game, one run. Must be module-level for pickling."""
    label, pop_spec, n_agents, memory, rounds, windows, seed, run_idx, lambda_ = args
    
    actions, attendance, prices, cohort_ids, cohort_meta = run_game_and_extract(
        n_agents        = n_agents,
        rounds          = rounds,
        lambda_         = lambda_,
        seed            = seed + run_idx,
        population_spec = pop_spec,
    )
    astats = attendance_statistics(attendance, n_agents, memory)
    pstats = price_statistics(prices)
    astats.update(pstats)
    psigs  = build_price_return_signals(prices, windows)
    agg_df = aggregate_correlation(attendance, psigs, windows)
    ind_df = individual_correlations(actions, psigs, windows, cohort_ids=cohort_ids)
    agg_df["run"] = run_idx
    ind_df["run"] = run_idx

    return {
        "label":       label,
        "run_idx":     run_idx,
        "attend_stats": astats,
        "agg_df":      agg_df,
        "ind_df":      ind_df,
        "attendance":  attendance,
        "prices":      prices,
        "cohort_meta": cohort_meta,
        "pop_spec":    pop_spec,
    }

def run_pure_experiment_parallel(cfg: BehaviouralCharConfig) -> Dict:
    """
    Run behavioural characterization for each pure population in parallel.
    One task per (payoff, run_idx) - identical assembly logic to family runner.
    """
    # Build one single-cohort pop_spec per payoff
    variants = []
    for payoff in cfg.pure_payoffs:
        pop_spec = {
            "total": cfg.N,
            "cohorts": [{
                "count":          cfg.N,
                "memory":         cfg.memory,
                "payoff":         payoff,
                "strategies":     cfg.n_strategies,
                "position_limit": 0,
                "agent_type":     "strategic",
            }]
        }
        label = payoff   # e.g. "BinaryMG", "DollarGame"
        variants.append((label, pop_spec, cfg.N, None))  # value=None for pure

    # One task per (variant, run_idx)
    tasks = [
        (label, pop_spec, n_agents, cfg.memory,
         cfg.rounds, cfg.windows, cfg.seed, run_idx, cfg.lambda_)
        for label, pop_spec, n_agents, _ in variants
        for run_idx in range(cfg.n_runs)
    ]

    raw = defaultdict(list)
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(_run_single_game, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            raw[res["label"]].append(res)
            print(f"  Done: {res['label']}  run {res['run_idx']+1}/{cfg.n_runs}")

    all_results = {}
    for label, pop_spec, n_agents, _ in variants:
        runs   = sorted(raw[label], key=lambda r: r["run_idx"])
        sample = next(r for r in runs if r["run_idx"] == 0)

        all_results[label] = {
            "attend_stats":   [r["attend_stats"] for r in runs],
            "agg_df":         pd.concat([r["agg_df"] for r in runs], ignore_index=True),
            "ind_df":         pd.concat([r["ind_df"] for r in runs], ignore_index=True),
            "sample_attend":  sample["attendance"],
            "sample_price":   sample["prices"],
            "cohort_meta":    sample["cohort_meta"],
            "population_spec": pop_spec,
            "label":          label,
            "value":          None,   # no sweep value for pure mode
            "n_agents":       n_agents,
            "rounds":         cfg.rounds,
            "seed":           cfg.seed,
        }

    return all_results

def run_family_experiment_parallel(cfg: BehaviouralCharConfig,
                                   family_cfg: PopulationFamilyConfig) -> Dict:
    memory = cfg.memory
    windows = cfg.windows
    n_runs = cfg.n_runs
    seed = cfg.seed
    max_workers = cfg.max_workers
    
    # Build all (label, pop_spec) pairs
    variants = []
    for value in family_cfg.values:
        pop_spec = build_population_variant(family_cfg, value)
        label    = f"share={float(value):.2f}"
        variants.append((label, pop_spec, pop_spec["total"], value))

    # Build full task list — one task per (variant, run)
    tasks = [
        (label, pop_spec, n_agents, memory,
         family_cfg.rounds, windows, seed, run_idx, cfg.lambda_)
        for label, pop_spec, n_agents, _ in variants
        for run_idx in range(n_runs)
    ]

    # Collect results grouped by label
    raw = defaultdict(list)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_game, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            raw[res["label"]].append(res)
            print(f"  Done: {res['label']}  run {res['run_idx']+1}/{n_runs}")

    # Reassemble into the same dict structure _run_n_games returns
    all_results = {}
    for label, pop_spec, n_agents, value in variants:
        runs = sorted(raw[label], key=lambda r: r["run_idx"])

        # Sample run is run_idx==0
        sample = next(r for r in runs if r["run_idx"] == 0)

        all_results[label] = {
            "attend_stats":  [r["attend_stats"] for r in runs],
            "agg_df":        pd.concat([r["agg_df"] for r in runs], ignore_index=True),
            "ind_df":        pd.concat([r["ind_df"] for r in runs], ignore_index=True),
            "sample_attend": sample["attendance"],
            "sample_price":  sample["prices"],
            "cohort_meta":   sample["cohort_meta"],
            "population_spec": pop_spec,
            "label":         label,
            "value":         float(value),
            "n_agents":      n_agents,
            "rounds":        family_cfg.rounds,
            "seed":          seed
        }

    return all_results

def _safe_save(pdf, fig, label="unknown"):
    """Save a figure to PDF, printing any error without raising."""
    try:
        pdf.savefig(fig, bbox_inches="tight")
        print(f"  [OK] page: {label}")
    except Exception as e:
        print(f"  [FAIL] page: {label}  --  {e}")
    finally:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file for a behavioural characterization experiment"
        )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    logger = RunLogger(
    base_save_dir = cfg.save_dir,
    module        = "BehaviouralChar",
    run_id        = cfg.label,
    seed          = cfg.seed,
    )
    logger.log_config(cfg)
    
    if cfg.mode == 'family':
        family_cfg = PopulationFamilyConfig(
            base_cohorts=cfg.base_cohorts(),
            vary="payoff_weights",
            values=cfg.dg_shares,
            target_payoff=cfg.target_payoff,
            rounds=cfg.rounds)
    
    if cfg.mode == 'pure':
        results = run_pure_experiment_parallel(cfg)
    else:
        results = run_family_experiment_parallel(cfg, family_cfg)
    
    try:
        pdf_path = plot_results(
                results,
                cfg=cfg,
                save_dir=logger.get_dir(),
            )
    except Exception as e:
        import traceback
        print(f"ERROR in plot_results:{e}")
        traceback.print_exc()
        raise
        
    save_data(results, cfg=cfg, save_dir=logger.get_dir())
    save_paper_figures(results, cfg=cfg, save_dir=logger.get_dir())
    logger.close()

    print(f"\nDONE - {logger.get_dir()}")

if __name__ == "__main__":
    main()
