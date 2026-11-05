#!/usr/bin/env bash
# Full experiments: N x N (default N=10), density 0.2, 5 seeds, EP episodes per run, all jobs in parallel (xargs -P).
# Thesis settings: DQN k5, DRQN k3/5/7. Same hyperparameters/network/rewards for every N; max_steps = 8N (env default),
# min start-goal distance 7, goal offset /(N-1) (consistent at train and test time).
#   N=10 (default): results go to <repo>/results/runs/<config>/ exactly as before (override root with R=/path).
#   N!=10         : results go to $R/runs_trained/N<N>/<config>/ and logs to $R/logs/trained_N<N>/
#                   (separate from the zero-shot transfer evaluations in $R/runs/N<N>/).
cd "$(dirname "$0")"
R=${R:-$(cd .. && pwd)/results}
EP=${EP:-10000}; P=${P:-8}; N=${N:-10}; SEEDS=${SEEDS:-"0 1 2 3 4"}
if [ "$N" = 10 ]; then RUNS="$R/runs"; LOGS="$R/logs"; else RUNS=${RUNS:-$R/runs_trained/N$N}; LOGS="$R/logs/trained_N$N"; fi
mkdir -p "$LOGS"
{
for s in $SEEDS; do
  echo "dqn 5 $s"; for k in 3 5 7; do echo "drqn $k $s"; done
done
} | xargs -P "$P" -L 1 sh -c 'python3 train.py --agent $0 --size '"$N"' --obs $1 --episodes '"$EP"' --seeds $2 --out "'"$RUNS"'/$0_s'"$N"'_k$1_d0.2" > "'"$LOGS"'/train_$0_k$1_s$2.log" 2>&1'
echo TRAIN_DONE N=$N "$(date)"
