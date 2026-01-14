# Configuration Files

## Directory Structure

```
config/
├── phase_diagrams/       # Phase diagram experiment configs
├── success_analysis/     # Success rate analysis configs
├── population_families/  # Population family comparison configs
├── examples/            # Template configurations
│   ├── TEMPLATE_phase_diagram.json
│   ├── TEMPLATE_success_boxplot.json
│   ├── TEMPLATE_success_evolution.json
│   └── TEMPLATE_population_family.json
└── README.md            # This file
```

---

## Creating New Experiments

### 1. Copy a Template

```bash
# For phase diagram
cp config/examples/TEMPLATE_phase_diagram.json config/phase_diagrams/phase_myexperiment.json

# For success analysis
cp config/examples/TEMPLATE_success_boxplot.json config/success_analysis/success_myexperiment.json

# For population family
cp config/examples/TEMPLATE_population_family.json config/population_families/family_myexperiment.json
```

### 2. Edit Parameters

Open the file and adjust:
- Payoff schemes
- Memory values
- Number of players
- Rounds
- Output paths (change `CHANGEME` to descriptive names)

### 3. Run Experiment

```bash
python experiments/plot_phase_diagram.py --config config/phase_diagrams/phase_myexperiment.json
```

---

## Config File Naming Convention

Use descriptive names that indicate:
- **Experiment type**
- **Payoff scheme**
- **Key variant or parameter**

Examples:
- `phase_scaledmg_small.json` - Small phase diagram for ScaledMG
- `phase_dollargame_withnoise.json` - Dollar game with noise traders
- `success_boxplot_hetero_memory.json` - Success analysis varying memory
- `family_memory_scaledmg.json` - Family varying memory for ScaledMG

---

## Common Parameters Explained

### Payoff Schemes
- `"BinaryMG"` - Binary Minority Game (±1 payoff)
- `"ScaledMG"` - Scaled Minority Game (scaled by N)
- `"SmallMinority"` - Small minority variation
- `"DollarGame"` - Dollar game (price-based payoff)

### Memory Values
- Typical range: 3-10
- Lower memory = simpler strategies
- Higher memory = more complex strategies
- Phase transitions often occur around m=6-8

### Strategies
- Common values: 2, 4, 6, 8
- More strategies = more exploration
- Fewer strategies = more exploitation

### Position Limits
- `0` or `null` = No limit
- Positive integer = Maximum absolute position (e.g., `10` = positions from -10 to +10)

### Lambda
- `null` = Auto-calculated as 1/(N×50)
- Explicit value = Custom market impact parameter
- Higher λ = more price impact

### Rounds
- Phase diagrams: 10,000+ (need convergence)
- Success analysis: 5,000-10,000
- Quick tests: 1,000

---

## Quick Reference

| Experiment Type | Template | Script | Typical Time |
|----------------|----------|--------|--------------|
| Phase Diagram | `TEMPLATE_phase_diagram.json` | `plot_phase_diagram.py` | 5-30 min |
| Success Boxplot | `TEMPLATE_success_boxplot.json` | `success_boxplot.py` | 1-5 min |
| Success Evolution | `TEMPLATE_success_evolution.json` | `success_evolution.py` | 5-15 min |
| Population Family | `TEMPLATE_population_family.json` | `population_family_experiment.py` | 2-10 min |

*Times vary based on rounds, num_games, and population size*

---

## Tips

### For Phase Diagrams
- Use `num_games: 20` for reliable statistics
- Cover a wide range of m_values (e.g., 3-10)
- Compare different payoff schemes

### For Success Analysis
- Use `mode: "cartesian"` to test all combinations of m and s
- Include diverse memory values (e.g., [3, 5, 7, 9])
- Set `record_agent_series: true` for detailed analysis

### For Population Families
- Use `vary: "memory_shift"` to see effects of memory changes
- Use `vary: "payoff_weights"` to see competitive dynamics
- Keep `rounds` moderate (5000) since you're running multiple populations

---

## Troubleshooting

**Config not found?**
- Check path is relative to where you run the script
- Use full path if needed: `--config /full/path/to/config.json`

**Experiment takes too long?**
- Reduce `rounds`
- Reduce `num_games` (for phase diagrams)
- Reduce `num_players_per_cohort` (for success analysis)

**Output plots not appearing?**
- Check `plots/` directory exists
- Check output paths in config are correct
- Look in `simulation_runs/` for detailed logs

---

## Example Workflow

```bash
# 1. Copy template
cp config/examples/TEMPLATE_success_boxplot.json config/success_analysis/success_memory_comparison.json

# 2. Edit file (change m_values, adjust rounds, etc.)
nano config/success_analysis/success_memory_comparison.json

# 3. Run experiment
python experiments/success_boxplot.py --config config/success_analysis/success_memory_comparison.json

# 4. Check results
ls simulation_runs/  # Detailed logs
ls plots/success/    # Output plots
```

---

## Need Help?

See `experiments/README.md` for more information on running experiments.
