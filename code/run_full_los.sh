#!/usr/bin/env bash
# Formal N=10 line-of-sight (obs_mode=los) training.
# Same protocol as run_full.sh at N=10: DQN k5 and DRQN k3/k5/k7, seeds 0-4,
# 10000 episodes, density 0.2, min Manhattan distance 7, max_steps=8N=80,
# gamma 0.99, PBRS v2, epsilon 1.0 -> 0.01 over 60% of episodes, lr 5e-4,
# batch 32, DRQN seq_len 16, burn-in 8, train_every 4, target_update 1000,
# learn_start 1000, hidden 128.
# Writes ONLY to results/runs_los/ (never results/runs). Does not train N=15/20.
# A seed is skipped only when its train_log has 10000 rows, model.pt exists, and
# config.json records obs_mode=los and episodes=10000. Anything else is retrained
# from episode 0 (train.py does not resume mid-run).
set -euo pipefail
cd "$(dirname "$0")"
R=${R:-$(cd .. && pwd)/results}
EP=10000
N=10
SEEDS="0 1 2 3 4"
P=${P:-$(nproc)}
RUNS="$R/runs_los"
LOGS="$R/logs/los"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
done_ok() {
  local d="$1" agent="$2" k="$3" seed="$4"
  python3 - "$d" "$agent" "$k" "$seed" << 'PY'
import json, os, sys
d, agent, k, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg_path, log_path, model = os.path.join(d, "config.json"), os.path.join(d, "train_log.csv"), os.path.join(d, "model.pt")
if not (os.path.isfile(cfg_path) and os.path.isfile(log_path) and os.path.isfile(model)):
    sys.exit(1)
cfg = json.load(open(cfg_path))
if cfg.get("obs_mode") != "los" or cfg.get("episodes") != 10000 or cfg.get("size") != 10:
    sys.exit(1)
if cfg.get("agent") != agent or cfg.get("obs") != k or cfg.get("seed") != seed or cfg.get("density") != 0.2:
    sys.exit(1)
n = sum(1 for _ in open(log_path)) - 1
sys.exit(0 if n == 10000 else 1)
PY
}
jobs=""
for s in $SEEDS; do
  jobs+="dqn 5 $s"$'\n'
  for k in 3 5 7; do jobs+="drqn $k $s"$'\n'; done
done
launch=""
while read -r agent k seed; do
  [ -n "${agent:-}" ] || continue
  out="$RUNS/${agent}_s${N}_k${k}_d0.2_los"
  seed_dir="$out/seed$seed"
  if done_ok "$seed_dir" "$agent" "$k" "$seed"; then
    echo "skip complete $agent k=$k seed=$seed"
    continue
  fi
  launch+="$agent $k $seed"$'\n'
done <<< "$jobs"
if [ -z "$launch" ]; then
  echo "TRAIN_DONE los N=10 all 20 already complete $(date)"
  exit 0
fi
printf '%s' "$launch" | xargs -P "$P" -L 1 sh -c 'python3 train.py --agent $0 --size '"$N"' --obs $1 --episodes '"$EP"' --seeds $2 --obs_mode los --out "'"$RUNS"'/$0_s'"$N"'_k$1_d0.2_los" > "'"$LOGS"'/train_$0_k$1_s$2.log" 2>&1'
echo TRAIN_DONE los N=$N EP=$EP "$(date)"
