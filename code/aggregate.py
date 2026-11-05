"""Aggregate eval_*.json across seeds: mean and 95% CI (Student t, n = #seeds) of success rate etc.,
plus significance tests between configurations.

Outputs (default results/):
  summary.csv               one row per (agent, k, train density, eval density, eval_eps)
  summary.md                (1) greedy table, (2) greedy vs epsilon-greedy table (density 0.2),
                            (3) significance tests
  significance.csv          all pairwise tests (overall and per difficulty group)

Significance tests (pairs: DQN k5 vs DRQN k5, DRQN k3 vs k5, k5 vs k7, k3 vs k7):
  * Welch t-test on the per-seed success rates (n = 5 vs 5, unequal variances).
  * Also within each config: eps-greedy (A) vs greedy (B) evaluation of the SAME models -> paired t-test over seeds,
    bootstrap with seeds AND maps paired.
  * Two-level paired bootstrap of the success-rate difference: every replicate resamples the 500 test maps
    (SAME map indices for both configs -> paired, since all configs share the identical test maps) and,
    independently for each config, resamples its seeds. 10,000 replicates; 95% percentile CI and a two-sided
    bootstrap p-value 2*min(P(diff<=0), P(diff>=0)).
"""
import argparse, csv, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats

DIFFS = ("easy", "medium", "hard")
PAIRS = [(("dqn", 5), ("drqn", 5)), (("drqn", 3), ("drqn", 5)), (("drqn", 5), ("drqn", 7)), (("drqn", 3), ("drqn", 7))]


def tcrit(n):
    return float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(tcrit(len(x)) * x.std(ddof=1) / np.sqrt(len(x)))


def collect(runs="runs", with_evals=False):
    groups = defaultdict(list)
    for f in glob.glob(f"{runs}/*/seed*/eval_*.json"):
        e = json.load(open(f))
        e["_file"] = f
        groups[(e["agent"], e["obs"], e["train_density"], e["eval_density"], e.get("eval_eps", 0.0))].append(e)
    rows = []
    for (agent, k, td, ed, ee), es in sorted(groups.items()):
        es.sort(key=lambda e: e["seed"])
        row = {"agent": agent, "obs": k, "train_density": td, "eval_density": ed, "eval_eps": ee, "n_seeds": len(es)}
        for m in ("success_rate", "collision_rate", "timeout_rate", "avg_steps_success"):
            vals = [e[m] for e in es if e[m] is not None]
            row[m], row[m + "_ci95"] = ci95(vals) if vals else (None, None)
        for d in DIFFS:
            vals = [e["by_difficulty"][d]["success_rate"] for e in es if e["by_difficulty"][d]["n"] > 0]
            row[f"success_{d}"], row[f"success_{d}_ci95"] = ci95(vals) if vals else (None, None)
        if with_evals:
            row["_evals"] = es
        rows.append(row)
    return rows


def episode_matrix(evals):
    """seeds x maps success matrix (0/1) and the per-map difficulty labels, from episodes_<tag>.csv."""
    S, diff, maps = [], None, None
    for e in evals:
        f = e["_file"].replace("eval_", "episodes_").replace(".json", ".csv")
        recs = sorted(csv.DictReader(open(f)), key=lambda r: int(r["map_seed"]))
        m = [int(r["map_seed"]) for r in recs]
        assert maps is None or m == maps, "configs must be evaluated on identical maps"
        maps, diff = m, np.array([r["difficulty"] for r in recs])
        S.append([r["outcome"] == "success" for r in recs])
    return np.array(S, float), diff, maps


def paired_bootstrap(SA, SB, B=10_000, seed=0, same_seeds=False):
    """Two-level bootstrap of mean(SA) - mean(SB); maps resampled jointly (paired), seeds per config
    (or jointly when same_seeds=True, i.e. A and B are the SAME trained models evaluated differently)."""
    rng = np.random.default_rng(seed)
    nA, nM = SA.shape; nB = SB.shape[0]
    wm = rng.multinomial(nM, np.full(nM, 1 / nM), size=B) / nM
    wa = rng.multinomial(nA, np.full(nA, 1 / nA), size=B) / nA
    wb = wa if same_seeds else rng.multinomial(nB, np.full(nB, 1 / nB), size=B) / nB
    d = ((wa @ SA) * wm).sum(1) - ((wb @ SB) * wm).sum(1)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), float(p)


def _test(ea, eb, ee_a, ee_b, A, B, same_seeds=False):
    out = []
    SA, dA, _ = episode_matrix(ea); SB, dB, _ = episode_matrix(eb)
    for grp in ("overall",) + DIFFS:
        sel = np.ones(SA.shape[1], bool) if grp == "overall" else (dA == grp)
        xa, xb = SA[:, sel].mean(1), SB[:, sel].mean(1)  # per-seed success rates
        t, p = stats.ttest_rel(xa, xb) if same_seeds else stats.ttest_ind(xa, xb, equal_var=False)
        lo, hi, pb = paired_bootstrap(SA[:, sel], SB[:, sel], same_seeds=same_seeds)
        out.append({"eval_eps": ee_a if ee_a == ee_b else f"{ee_a:g} vs {ee_b:g}",
                    "A": f"{A[0]}_k{A[1]}", "B": f"{B[0]}_k{B[1]}", "group": grp,
                    "n_maps": int(sel.sum()), "mean_A": xa.mean(), "mean_B": xb.mean(),
                    "diff_A_minus_B": xa.mean() - xb.mean(), "t_test": "paired t (same models)" if same_seeds else "Welch", "welch_t": float(t), "welch_p": float(p),
                    "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_p": pb,
                    "significant_0.05": bool(p < 0.05 and (lo > 0 or hi < 0))})
    return out


def significance(rows, density=0.2):
    """(a) between-config pairs at every eval eps; (b) within-config: eps-greedy (A) vs greedy (B)."""
    out = []
    idx = {(r["agent"], r["obs"], r["eval_eps"]): r for r in rows
           if abs(r["eval_density"] - density) < 1e-9 and abs(r["train_density"] - density) < 1e-9}
    epss = sorted({k[2] for k in idx})
    for ee in epss:
        for A, B in PAIRS:
            if (A + (ee,)) in idx and (B + (ee,)) in idx:
                out += _test(idx[A + (ee,)]["_evals"], idx[B + (ee,)]["_evals"], ee, ee, A, B)
    for C in sorted({k[:2] for k in idx if k[0] != "random"}):
        for ee in epss:
            if ee > 0 and (C + (ee,)) in idx and (C + (0.0,)) in idx:
                out += _test(idx[C + (ee,)]["_evals"], idx[C + (0.0,)]["_evals"], ee, 0.0, C, C, same_seeds=True)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs", default="runs"); p.add_argument("--out", default="results/summary.csv")
    p.add_argument("--density", type=float, default=0.2, help="density for the eps table and significance tests")
    a = p.parse_args()
    rows = collect(a.runs, with_evals=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sig = significance(rows, a.density)
    plain = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plain[0])); w.writeheader(); w.writerows(plain)
    sig_path = os.path.join(os.path.dirname(a.out) or ".", "significance.csv")
    if sig:
        with open(sig_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sig[0])); w.writeheader()
            w.writerows([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()} for s in sig])

    f2 = lambda m, c: "n/a" if m is None else (f"{m:.3f} ± {c:.3f}" if c == c else f"{m:.3f}")
    md = ["# 结果汇总 / Results summary", "",
          "所有数值为 5 个种子的均值 ± 95% 置信区间（Student t，n=种子数）；测试集为固定的 500 张地图（seed 10,000,000+i）。", "",
          "## 1. 贪心评估（ε=0，主协议） / Greedy evaluation", "",
          "| agent | k | eval density | seeds | success (mean ± 95% CI) | easy | medium | hard | collision | timeout | avg steps (succ.) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["eval_eps"] != 0:
            continue
        md.append(f"| {r['agent']} | {r['obs']} | {r['eval_density']} | {r['n_seeds']} | {f2(r['success_rate'], r['success_rate_ci95'])} | "
                  f"{f2(r['success_easy'], r['success_easy_ci95'])} | {f2(r['success_medium'], r['success_medium_ci95'])} | "
                  f"{f2(r['success_hard'], r['success_hard_ci95'])} | {f2(r['collision_rate'], r['collision_rate_ci95'])} | "
                  f"{f2(r['timeout_rate'], r['timeout_rate_ci95'])} | {f2(r['avg_steps_success'], r['avg_steps_success_ci95'])} |")
    md += ["", f"## 2. 贪心 vs ε-贪心评估（density {a.density}） / Greedy vs epsilon-greedy evaluation", "",
           "| agent | k | eval ε | success | easy | medium | hard | collision | timeout | avg steps (succ.) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["agent"], r["obs"], r["eval_eps"])):
        if abs(r["eval_density"] - a.density) > 1e-9 or r["agent"] == "random":
            continue
        md.append(f"| {r['agent']} | {r['obs']} | {r['eval_eps']:g} | {f2(r['success_rate'], r['success_rate_ci95'])} | "
                  f"{f2(r['success_easy'], r['success_easy_ci95'])} | {f2(r['success_medium'], r['success_medium_ci95'])} | "
                  f"{f2(r['success_hard'], r['success_hard_ci95'])} | {f2(r['collision_rate'], r['collision_rate_ci95'])} | "
                  f"{f2(r['timeout_rate'], r['timeout_rate_ci95'])} | {f2(r['avg_steps_success'], r['avg_steps_success_ci95'])} |")
    md += ["", f"## 3. 显著性检验（density {a.density}） / Significance tests", "",
           "Welch t 检验：以每个种子的成功率为样本（n=5 vs 5）。配对 bootstrap：两层重采样（500 张共享测试地图配对重采样 + 各配置种子独立重采样，10,000 次），"
           "给出成功率差 A−B 的 95% 百分位置信区间与双侧 bootstrap p 值。“显著”= Welch p<0.05 且 bootstrap CI 不含 0。"
           "“eval ε”列为 “e vs 0” 的行是同一配置 ε-贪心(A) 与贪心(B) 的比较：同一批模型，故改用配对 t 检验（按种子配对），bootstrap 中种子与地图都配对重采样（“Welch p”列此时为配对 t 检验 p 值）。", "",
           "| eval ε | A | B | group | n maps | A | B | A−B | Welch p | bootstrap 95% CI of A−B | bootstrap p | 结论 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bp = lambda p: "<1e-4" if p == 0 else f"{p:.2g}"  # 10,000 replicates -> resolution 1e-4
    for s in sig:
        verdict = "显著" if s["significant_0.05"] else "不显著"
        ee = s["eval_eps"] if isinstance(s["eval_eps"], str) else f"{s['eval_eps']:g}"
        md.append(f"| {ee} | {s['A']} | {s['B']} | {s['group']} | {s['n_maps']} | {s['mean_A']:.3f} | {s['mean_B']:.3f} | "
                  f"{s['diff_A_minus_B']:+.3f} | {s['welch_p']:.2g} | [{s['boot_ci_lo']:+.3f}, {s['boot_ci_hi']:+.3f}] | "
                  f"{bp(s['boot_p'])} | {verdict} |")
    open(a.out.replace(".csv", ".md"), "w").write("\n".join(md) + "\n")
    fmt = lambda m, c: "  n/a " if m is None else f"{m:.3f}±{c:.3f}"
    for r in rows:
        print(f"{r['agent']:7s} k={r['obs']} eval_d={r['eval_density']} eps={r['eval_eps']:g} seeds={r['n_seeds']} "
              f"succ={fmt(r['success_rate'], r['success_rate_ci95'])} | easy={fmt(r['success_easy'], r['success_easy_ci95'])} "
              f"med={fmt(r['success_medium'], r['success_medium_ci95'])} hard={fmt(r['success_hard'], r['success_hard_ci95'])}")
    for s in sig:
        if s["group"] == "overall":
            print(f"eps={s['eval_eps']} {s['A']} vs {s['B']}: diff={s['diff_A_minus_B']:+.3f} welch_p={s['welch_p']:.3g} "
                  f"boot CI=[{s['boot_ci_lo']:+.3f},{s['boot_ci_hi']:+.3f}] boot_p={s['boot_p']:.3g}")
    print("saved", a.out, ".md and", sig_path)


if __name__ == "__main__":
    main()
