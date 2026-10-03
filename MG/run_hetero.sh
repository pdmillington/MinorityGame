#!/bin/bash
set -e  # stop on first error

CONFIGS=(
    "config/behavioural_characterization/mg5_dg5.json"
    "config/behavioural_characterization/mg5_dg7.json"
    "config/behavioural_characterization/mg3_dg5.json"
    "config/behavioural_characterization/mg7_dg5.json"
)

for config in "${CONFIGS[@]}"; do
    echo "================================================"
    echo "Running: $config"
    echo "================================================"
    python experiments/behavioural_characterization.py --config "$config"
done

echo "All done."
