#!/usr/bin/env bash
# Smoke test (10x10, density 0.2, 2 seeds, very short training, 100 test maps) of the whole pipeline:
# train -> greedy + eps-greedy eval -> CI/significance table -> plots -> failure analysis.
# Everything is written to $R (default ./smoke_results), NOT to the thesis results directory.
# Formal experiments: run_full.sh (5 seeds x 10000 episodes) + run_eval_all.sh (500 maps).
set -e
cd "$(dirname "$0")"
R=${R:-$PWD/smoke_results}; EP=${EP:-300}; N=${N:-100}
mkdir -p $R/logs
for s in 0 1; do
  python3 train.py --agent dqn  --obs 5 --episodes $EP --seeds $s --out $R/runs/dqn_s10_k5_d0.2 > $R/logs/dqn_k5_s$s.log 2>&1 &
  for k in 3 5 7; do
    python3 train.py --agent drqn --obs $k --episodes $EP --seeds $s --out $R/runs/drqn_s10_k${k}_d0.2 > $R/logs/drqn_k${k}_s$s.log 2>&1 &
  done
done
wait
for r in $R/runs/*/; do
  for d in 0.1 0.2 0.3; do python3 evaluate.py --run "$r" --n_maps $N --density $d > /dev/null & done
  python3 evaluate.py --run "$r" --n_maps $N --density 0.2 --eval_eps 0.01 > /dev/null &
done
wait
python3 aggregate.py --runs $R/runs --out $R/summary.csv
python3 plot.py --runs $R/runs --out $R/plots
python3 failure_analysis.py --runs $R/runs --out $R/failures --render_per_config 1
echo "smoke test OK -> $R/summary.md"
