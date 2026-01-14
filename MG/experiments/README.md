# Running Experiments

## Quick Start

```bash
# From repository root
cd /path/to/MinorityGame

# Run a phase diagram
python MG/experiments/plot_phase_diagram.py --config MG/config/phase_diagrams/phase_scaledmg_small.json

# Run success analysis
python MG/experiments/success_boxplot.py --config MG/config/success_analysis/success_boxplot_test.json

# Run population family
python MG/experiments/population_family_experiment.py --config MG/config/population_families/population_family_test3.json
```

---

## Available Experiments

### 1. Phase Diagram (`plot_phase_diagram.py`)

**What it does:**
- Generates phase diagrams showing σ²/N vs α = 2^m/N
- Plots return variance and mean
- Runs multiple games for statistical reliability

**When to use:**
- Exploring phase transitions
- Comparing different payoff schemes
- Understanding collective behavior at different memory scales

**Output:**
- `plots/phase/phase_diagram_*.pdf` - Main phase diagram
- `plots/phase/return_var_*.pdf` - Return variance plot
- `plots/phase/return_mean_*.pdf` - Return mean plot
- `logs/simulation_log.txt` - Logged metadata

**Typical runtime:** 5-30 minutes (depends on m_values range and num_games)

---

### 2. Success Boxplot (`success_boxplot.py`)

**What it does:**
- Compares success rates across heterogeneous cohorts
- Generates boxplots showing distribution per cohort
- Creates wealth distribution plots
- Produces per-agent CSV data

**When to use:**
- Comparing strategies (different m, s combinations)
- Understanding which agent types perform best
- Analyzing heterogeneous populations

**Output:**
- `plots/success/*_success_boxplot.pdf` - Success rate by cohort
- `plots/success/*_wealth_boxplot.pdf` - Wealth distribution
- `plots/success/*_success_mean_line.pdf` - Mean success rates
- `plots/success/*_price.pdf` - Price evolution
- `simulation_runs/*/per_player_metrics.csv` - Detailed per-agent data

**Typical runtime:** 1-5 minutes

---

### 3. Success Evolution (`success_evolution.py`)

**What it does:**
- Tracks how success rates evolve over time
- Shows moving averages at different time scales
- Useful for understanding convergence

**When to use:**
- Checking if game has converged
- Understanding temporal dynamics
- Comparing adaptation speeds

**Output:**
- `plots/success/success_evolution_*.pdf` - Evolution plot

**Typical runtime:** 5-15 minutes (long games needed)

---

### 4. Population Family (`population_family_experiment.py`)

**What it does:**
- Compares related populations (e.g., varying memory)
- Generates comprehensive multi-page PDF report
- Shows multiple metrics (attendance, success, wealth, correlations)

**When to use:**
- Exploring parameter sensitivity
- Comparing competitive dynamics
- Understanding effects of population composition

**Output:**
- `simulation_runs/*/plots/*_family_report.pdf` - Multi-page report with:
  - Population descriptions
  - Attendance time series and distributions
  - Price evolution
  - Success rates and points distributions
  - Wealth analysis
  - Risk-return relationships
  - Cohort-level comparisons

**Typical runtime:** 2-10 minutes (multiple populations)

---

## Creating New Experiments

### Step 1: Choose Template

```bash
ls config/examples/TEMPLATE_*.json
```

Available templates:
- `TEMPLATE_phase_diagram.json`
- `TEMPLATE_success_boxplot.json`
- `TEMPLATE_success_evolution.json`
- `TEMPLATE_population_family.json`

### Step 2: Copy and Edit

```bash
# Copy template
cp config/examples/TEMPLATE_success_boxplot.json config/success_analysis/my_experiment.json

# Edit parameters
nano config/success_analysis/my_experiment.json
```

### Step 3: Run

```bash
python experiments/success_boxplot.py --config config/success_analysis/my_experiment.json
```

---

## Common Parameters

### Population Size
- **Small:** 101 players (quick tests)
- **Medium:** 301 players (standard)
- **Large:** 1001+ players (publication quality)

### Memory Values
- **Low:** m=3-5 (simple strategies)
- **Medium:** m=6-8 (complex strategies, phase transition region)
- **High:** m=9-12 (very complex, may not converge)

### Number of Strategies
- **Minimum:** s=2 (binary choice)
- **Typical:** s=2-6
- **High:** s=8+ (more exploration)

### Rounds
- **Quick test:** 1,000 rounds
- **Standard:** 5,000-10,000 rounds
- **Phase diagrams:** 10,000+ rounds (need convergence)
- **Publication:** 50,000+ rounds

### Lambda (Market Impact)
- `null` = Auto-calculated as 1/(N×50) (recommended)
- Explicit value for custom impact

---

## Understanding Output

### Phase Diagrams
- **Below critical α:** Oscillating phase (σ²/N ≈ 1)
- **Above critical α:** Frozen phase (σ²/N ≈ 0)
- **Transition:** Around α ≈ 0.3-0.4

### Success Boxplots
- **Wide boxes:** High variance in performance
- **Narrow boxes:** Consistent performance
- **Outliers:** Indicate interesting strategies or luck

### Population Families
- **Attendance:** Should stabilize (not explode)
- **Success rates:** Compare across cohorts
- **Wealth:** Not all cohorts should have same wealth (heterogeneity)

---

## Troubleshooting

### Experiment Hangs
- Reduce `rounds`
- Reduce `num_players`
- Check for infinite loops in code

### Prices Explode
- Check lambda value (may be too high)
- Ensure position limits are set if needed
- Verify payoff scheme is correct

### No Output Files
- Check `plots/` directory exists
- Verify path in config is correct
- Look in `simulation_runs/` for logs

### Import Errors
- Ensure you're running from correct directory
- Check Python path includes MG package
- Verify all dependencies installed

---

## Best Practices

### 1. Start Small
```json
{
  "rounds": 1000,
  "num_players": 101
}
```
Test with quick runs before committing to long experiments.

### 2. Use Descriptive Names
```bash
# Good
config/phase_diagrams/phase_scaledmg_m3to10_n301.json

# Bad
config/test.json
```

### 3. Document Your Configs
Add comments (using `_description` field) explaining:
- What you're testing
- Why these parameters
- Expected outcomes

### 4. Version Your Configs
```bash
config/success_analysis/success_memory_v1.json
config/success_analysis/success_memory_v2.json
```

### 5. Keep Templates Clean
Don't modify templates directly. Always copy first.

---

## Batch Running

Create simple scripts for running multiple experiments:

```bash
#!/bin/bash
# run_my_experiments.sh

echo "Running experiment suite..."

python experiments/success_boxplot.py --config config/success_analysis/exp1.json
python experiments/success_boxplot.py --config config/success_analysis/exp2.json
python experiments/plot_phase_diagram.py --config config/phase_diagrams/phase1.json

echo "Complete! Check plots/ and simulation_runs/"
```

Make executable:
```bash
chmod +x run_my_experiments.sh
./run_my_experiments.sh
```

---

## Output Organization

Results are organized by timestamp and experiment:

```
plots/
├── phase/
│   └── 20260114_153022_phase_diagram.pdf
├── success/
│   └── 20260114_153045_success_boxplot.pdf
└── family/

simulation_runs/
├── 20260114_153022_PopulationFamily_memory_shift/
│   ├── run_info.json
│   ├── config_used.json
│   ├── per_player_metrics.csv
│   └── plots/
│       └── 20260114_153022_family_report.pdf
└── 20260114_153045_HeteroCohorts_ScaledMG/
    ├── run_info.json
    ├── per_player_metrics.csv
    └── plots/
```

---

## Next Steps

1. **Copy a template:** Start with `TEMPLATE_success_boxplot.json`
2. **Adjust parameters:** Edit for your research question
3. **Run experiment:** `python experiments/success_boxplot.py --config your_config.json`
4. **Analyze results:** Check `plots/` and `simulation_runs/`
5. **Iterate:** Adjust config based on results

---

## Getting Help

- **Config questions:** See `config/README.md`
- **Code questions:** See source code docstrings
- **Bug reports:** Check code or ask for help

Happy experimenting! 🚀
