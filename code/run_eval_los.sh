#!/usr/bin/env bash
# Formal evaluation of the N=10 obs_mode=los models in results/runs_los.
# Fixed test maps: seeds 10_000_000+i, n_maps=500. Not overridable.
# Greedy (eval_eps=0) and epsilon=0.01 at density 0.2, the primary protocol.
# Then CI / Welch / paired bootstrap / Holm, plots, failure labels, detour bins.
# All outputs stay out of the published window-mode files (summary.csv, plots/, failures/, detour/).
set -euo pipefail
cd "$(dirname "$0")"
R=${R:-$(cd .. && pwd)/results}
N=500
P=${P:-$(nproc)}
RUNS="$R/runs_los"
LOGS="$R/logs/los"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
need_eval() {
  python3 - "$1" "$2" << 'PY'
import json, os, sys
path, eps = sys.argv[1], float(sys.argv[2])
if not os.path.isfile(path):
    sys.exit(1)
e = json.load(open(path))
ok = (e.get("n_maps") == 500 and e.get("obs_mode") == "los" and abs(e.get("eval_eps", 0) - eps) < 1e-12
      and abs(e.get("eval_density", -1) - 0.2) < 1e-12 and not e.get("exploratory_cross_mode"))
sys.exit(0 if ok else 1)
PY
}
pending=""
for c in "$RUNS"/*/; do
  [ -d "$c" ] || continue
  name=$(basename "$c")
  case "$name" in
    dqn_s10_k5_d0.2_los|drqn_s10_k3_d0.2_los|drqn_s10_k5_d0.2_los|drqn_s10_k7_d0.2_los) ;;
    *) echo "unexpected directory under runs_los: $name" >&2; exit 1 ;;
  esac
  for s in "$c"seed*; do
    [ -d "$s" ] || continue
    if ! need_eval "$s/eval_d0.2.json" 0; then pending+="$s 0.2 0"$'\n'; fi
    if ! need_eval "$s/eval_d0.2_eps0.01.json" 0.01; then pending+="$s 0.2 0.01"$'\n'; fi
  done
done
if [ -n "$pending" ]; then
  printf '%s' "$pending" | xargs -P "$P" -L 1 sh -c 'python3 evaluate.py --run "$0" --n_maps '"$N"' --density $1 --eval_eps $2 >> "'"$LOGS"'/eval.log" 2>&1'
fi
python3 aggregate.py --runs "$RUNS" --out "$R/summary_los.csv"
python3 plot.py --runs "$RUNS" --obs_mode los --out "$R/plots_los"
python3 failure_analysis.py --runs "$RUNS" --out "$R/failures_los"
python3 failure_analysis.py --runs "$RUNS" --out "$R/failures_los" --eval_eps 0.01
python3 detour_analysis.py --runs "$RUNS" --out "$R/detour_los"
echo EVAL_DONE los n_maps=$N "$(date)"
