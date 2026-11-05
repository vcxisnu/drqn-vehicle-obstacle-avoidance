#!/usr/bin/env bash
# Evaluate the models RETRAINED on N x N (run_full.sh with N=15/20 -> $R/runs_trained/N<N>/) on 500 fixed test maps
# generated at the same N (seeds 10,000,000+i), density 0.2: greedy and eps-greedy ($EVAL_EPS); then aggregate per N
# (mean +- 95% CI, within-N significance tests with Holm), detour and failure analysis.
# Outputs (zero-shot transfer results in $R/runs/N<N>/ and the training dirs are NOT touched):
#   $R/runs_trained_eval/N<N>/<config>/seed<s>/  (eval json/csv + source.json pointing at the trained model)
#   $R/summary_trained_N<N>.{csv,md}, $R/significance_trained_N<N>.csv, $R/trained/N<N>/{detour,failures}/
cd "$(dirname "$0")"
R=${R:-$(cd .. && pwd)/results}; SIZES=${SIZES:-"15 20"}; NM=${NM:-500}; EVAL_EPS=${EVAL_EPS:-"0 0.01"}; P=${P:-8}
mkdir -p "$R/logs"
{
for N in $SIZES; do
  for s in "$R"/runs_trained/N$N/*/seed*; do
    for e in $EVAL_EPS; do echo "$s $N $e"; done
  done
done
} | xargs -P "$P" -L 1 sh -c 'python3 evaluate.py --run "$0" --out "'"$R"'/runs_trained_eval/N$1" --n_maps '"$NM"' --density 0.2 --eval_eps $2 >> "'"$R"'/logs/trained_eval.log" 2>&1'
for N in $SIZES; do
  python3 aggregate.py --runs "$R/runs_trained_eval/N$N" --out "$R/summary_trained_N$N.csv"
  python3 detour_analysis.py --runs "$R/runs_trained_eval/N$N" --out "$R/trained/N$N/detour"
  for e in $EVAL_EPS; do python3 failure_analysis.py --runs "$R/runs_trained_eval/N$N" --out "$R/trained/N$N/failures" --eval_eps "$e" --render_per_config 2; done
done
python3 compare_trained_transfer.py --sizes $SIZES
echo EVAL_TRAINED_DONE "$(date)"
