"""Aggregate eval_*.json across seeds: mean and 95% CI (Student t) of success rate etc.
Writes results/summary.csv and prints a table."""
import argparse, csv, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def collect(runs="runs"):
    groups = defaultdict(list)
    for f in glob.glob(f"{runs}/*/seed*/eval_*.json"):
        e = json.load(open(f))
        groups[(e["agent"], e["obs"], e["train_density"], e["eval_density"])].append(e)
    rows = []
    for (agent, k, td, ed), es in sorted(groups.items()):
        row = {"agent": agent, "obs": k, "train_density": td, "eval_density": ed, "n_seeds": len(es)}
        for m in ("success_rate", "collision_rate", "timeout_rate", "avg_steps_success"):
            vals = [e[m] for e in es if e[m] is not None]
            row[m], row[m + "_ci95"] = ci95(vals) if vals else (None, None)
        for d in ("easy", "medium", "hard"):
            vals = [e["by_difficulty"][d]["success_rate"] for e in es if e["by_difficulty"][d]["n"] > 0]
            row[f"success_{d}"], row[f"success_{d}_ci95"] = ci95(vals) if vals else (None, None)
        rows.append(row)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--runs", default="runs"); p.add_argument("--out", default="results/summary.csv")
    a = p.parse_args()
    rows = collect(a.runs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fmt = lambda m, c: "  n/a " if m is None else f"{m:.3f}±{c:.3f}"
    for r in rows:
        print(f"{r['agent']:7s} k={r['obs']} train_d={r['train_density']} eval_d={r['eval_density']} seeds={r['n_seeds']} "
              f"succ={fmt(r['success_rate'], r['success_rate_ci95'])} coll={r['collision_rate']:.3f} "
              f"timeout={r['timeout_rate']:.3f} | easy={fmt(r['success_easy'], r['success_easy_ci95'])} "
              f"med={fmt(r['success_medium'], r['success_medium_ci95'])} hard={fmt(r['success_hard'], r['success_hard_ci95'])}")
    md = ["| agent | k | eval density | seeds | success (mean ± 95% CI) | easy | medium | hard | collision | timeout | avg steps (succ.) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda m, c: "n/a" if m is None else (f"{m:.3f} ± {c:.3f}" if c == c else f"{m:.3f}")
    for r in rows:
        md.append(f"| {r['agent']} | {r['obs']} | {r['eval_density']} | {r['n_seeds']} | {f2(r['success_rate'], r['success_rate_ci95'])} | "
                  f"{f2(r['success_easy'], r['success_easy_ci95'])} | {f2(r['success_medium'], r['success_medium_ci95'])} | "
                  f"{f2(r['success_hard'], r['success_hard_ci95'])} | {f2(r['collision_rate'], r['collision_rate_ci95'])} | "
                  f"{f2(r['timeout_rate'], r['timeout_rate_ci95'])} | {f2(r['avg_steps_success'], r['avg_steps_success_ci95'])} |")
    open(a.out.replace(".csv", ".md"), "w").write("\n".join(md) + "\n")
    print("saved", a.out, "and .md")
