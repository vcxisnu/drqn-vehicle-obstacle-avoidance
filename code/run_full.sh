#!/usr/bin/env bash
# Full experiments: 10x10, density 0.2, 5 seeds, EP episodes per run, all jobs in parallel (xargs -P).
cd "$(dirname "$0")"
R=/workspace/thesis/results; mkdir -p $R/logs
EP=${EP:-10000}; P=${P:-8}
{
for s in 0 1 2 3 4; do
  echo "dqn 5 $s"; for k in 3 5 7; do echo "drqn $k $s"; done
done
} | xargs -P $P -L 1 sh -c 'python3 train.py --agent $0 --obs $1 --episodes '$EP' --seeds $2 --out '$R'/runs/$0_s10_k$1_d0.2 > '$R'/logs/train_$0_k$1_s$2.log 2>&1'
echo TRAIN_DONE $(date)
