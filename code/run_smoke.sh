#!/usr/bin/env bash
# Smoke test (10x10, density 0.2, 3 seeds, short training) -> eval on the fixed 300-map test set -> CI table -> plots.
# Formal experiments: use >=5 seeds and many more episodes.
set -e
cd "$(dirname "$0")"
mkdir -p logs
SEEDS="0 1 2"; EP_DQN=${EP_DQN:-2000}; EP_DRQN=${EP_DRQN:-1500}
for s in $SEEDS; do
  # Experiment 1: DQN (param-matched FC) vs DRQN (LSTM), k=5
  python3 train.py --agent dqn  --obs 5 --episodes $EP_DQN  --seeds $s > logs/dqn_k5_s$s.log 2>&1 &
  # Experiment 2: DRQN with k = 3 / 5 / 7
  for k in 3 5 7; do
    python3 train.py --agent drqn --obs $k --episodes $EP_DRQN --seeds $s > logs/drqn_k${k}_s$s.log 2>&1 &
  done
done
wait
for r in runs/*/; do
  for d in 0.1 0.2 0.3; do python3 evaluate.py --run "$r" --n_maps 300 --density $d > logs/eval_$(basename $r)_d$d.log 2>&1 & done
done
wait
python3 aggregate.py --runs runs --out results/summary.csv
python3 plot.py --runs runs --out plots
