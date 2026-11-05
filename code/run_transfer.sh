#!/usr/bin/env bash
# Zero-shot size transfer: evaluate the EXISTING 10x10-trained models (results/runs/<config>/seed*/model.pt, no
# retraining) on N x N maps, N in $SIZES (default "15 20"): 500 fixed test maps per N (seeds 10,000,000+i generated
# at that N), density 0.2, min start-goal Manhattan distance 7, max_steps = 8N; greedy and eps-greedy ($EVAL_EPS).
# Outputs: $R/runs/N<N>/<config>/seed<s>/  (10x10 files untouched), $R/summary_N<N>.{csv,md}, $R/significance_N<N>.csv,
#          $R/transfer/N<N>/{detour,failures}/
cd "$(dirname "$0")"
R=${R:-$(cd .. && pwd)/results}; SIZES=${SIZES:-"15 20"}; NM=${NM:-500}; EVAL_EPS=${EVAL_EPS:-"0 0.01"}; P=${P:-8}
mkdir -p "$R/logs"
./make_random_runs.sh "$R"
{
for N in $SIZES; do
  for c in "$R"/runs/*_s10_*/; do
    for s in "$c"seed*; do
      for e in $EVAL_EPS; do echo "$s $N $e"; done
    done
  done
done
} | xargs -P "$P" -L 1 sh -c 'python3 evaluate.py --run "$0" --size $1 --out "'"$R"'/runs/N$1" --n_maps '"$NM"' --density 0.2 --eval_eps $2 >> "'"$R"'/logs/transfer_eval.log" 2>&1'
for N in $SIZES; do
  python3 aggregate.py --runs "$R/runs/N$N" --out "$R/summary_N$N.csv"
  python3 detour_analysis.py --runs "$R/runs/N$N" --out "$R/transfer/N$N/detour"
  for e in $EVAL_EPS; do python3 failure_analysis.py --runs "$R/runs/N$N" --out "$R/transfer/N$N/failures" --eval_eps "$e" --render_per_config 2; done
done
echo TRANSFER_DONE "$(date)"
