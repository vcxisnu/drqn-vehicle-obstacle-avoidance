#!/usr/bin/env bash
# Create "runs" for the random baseline (5 seeds = 5 different action RNG streams) so evaluate/aggregate treat it uniformly.
R=${1:-/workspace/thesis/results}
for s in 0 1 2 3 4; do
  python3 train.py --agent random --obs 5 --episodes 1 --seeds $s --out $R/runs/random_s10_k5_d0.2 > /dev/null
done
