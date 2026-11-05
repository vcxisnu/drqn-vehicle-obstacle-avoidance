#!/usr/bin/env bash
# Evaluate every run on the fixed held-out test set (N=500 maps, seeds 10,000,000+i):
#   greedy (eval eps = 0, main protocol) at densities 0.1/0.2/0.3, and epsilon-greedy (EVAL_EPS, default 0.01 0.05)
#   at the training density 0.2; then aggregate (CI + Holm + significance tests), plot, analyse failures, detour analysis.
cd "$(dirname "$0")"
R=${R:-$(cd .. && pwd)/results}; N=${N:-500}; EVAL_EPS=${EVAL_EPS:-"0.01 0.05"}; P=${P:-8}
mkdir -p "$R/logs"
./make_random_runs.sh "$R"
{
for c in "$R"/runs/*/; do
  name=$(basename "$c")
  for s in "$c"seed*; do
    for d in 0.1 0.2 0.3; do echo "$s $d 0"; done
    case "$name" in random*) ;; *) for e in $EVAL_EPS; do echo "$s 0.2 $e"; done ;; esac
  done
done
} | xargs -P "$P" -L 1 sh -c 'python3 evaluate.py --run "$0" --n_maps '"$N"' --density $1 --eval_eps $2 >> "'"$R"'/logs/eval.log" 2>&1'
python3 aggregate.py --runs "$R/runs" --out "$R/summary.csv"
python3 plot.py --runs "$R/runs" --out "$R/plots"
python3 failure_analysis.py --runs "$R/runs" --out "$R/failures"
for e in $EVAL_EPS; do python3 failure_analysis.py --runs "$R/runs" --out "$R/failures" --eval_eps "$e"; done
python3 detour_analysis.py --runs "$R/runs" --out "$R/detour"
