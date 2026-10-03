# Minority Game Research

Research code and generated results for the Minority Game strand of Peter
Millington's PhD on agent-based models of fixed-income markets.

The repository is intentionally separate from the LaTeX thesis repository.
Code, experiment configurations, tests, and reproducibility instructions live
here. Curated publication figures are copied into the thesis so that the thesis
continues to compile independently in Overleaf and local LaTeX environments.

## Repository layout

- `MG/core/`: game mechanics, agents, population construction, and market maker.
- `MG/payoffs/`: payoff definitions.
- `MG/experiments/`: executable research experiments.
- `MG/analysis/`: analysis and reporting utilities.
- `MG/config/`: tracked templates and example configurations.
- `MG/tests/`: unit and integration tests.
- `outputs/`, `plots/`, `results/`, `simulation_runs/`, and `logs/`: generated
  local research artifacts; these are excluded from Git.

Detailed experiment and configuration guidance is available in
`MG/experiments/README.md` and `MG/config/README.md`.

## Installation

Python 3.8 or newer is required. From the repository root, create and activate
a virtual environment, then install the project in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Verification

Run the test suite from the repository root:

```bash
pytest
```

## Running experiments

Experiment scripts are run from the repository root. For example:

```bash
python MG/experiments/plot_phase_diagram.py \
  --config MG/config/examples/TEMPLATE_phase_diagram.json
```

Start with a copied configuration and use a descriptive filename rather than
editing the templates directly. Generated results should remain in the ignored
output directories unless an artifact is deliberately selected for publication.

## Thesis workflow

The related thesis is maintained separately at `../thesis`. Do not reference
this repository through relative paths in LaTeX: doing so would make the thesis
dependent on the local directory layout and break Overleaf builds. Instead,
copy selected final figures into the thesis's `minority game/` directory and
record the experiment configuration or simulation run that produced them.
