#!/usr/bin/env bash
# Evaluate every run on the fixed held-out test set (500 maps) at densities 0.1/0.2/0.3, then aggregate, plot, analyse failures.
cd "$(dirname "$0")"
R=/workspace/thesis/results; N=${N:-500}
./make_random_runs.sh $R
for c in $R/runs/*/; do for s in $c/seed*; do for d in 0.1 0.2 0.3; do echo "$s $d"; done; done; done \
 | xargs -P 8 -L 1 sh -c 'python3 evaluate.py --run $0 --n_maps '$N' --density $1 >> '$R'/logs/eval.log 2>&1'
python3 aggregate.py --runs $R/runs --out $R/summary.csv
python3 plot.py --runs $R/runs --out $R/plots
python3 failure_analysis.py --runs $R/runs --out $R/failures
