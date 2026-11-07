"""Aggregate eval_*.json across seeds: mean and 95% CI (Student t, n = #seeds) of success rate etc.,
plus significance tests between configurations.

Outputs (default results/):
  summary.csv               one row per (agent, k, train density, eval density, eval_eps)
  summary.md                (1) greedy table, (2) greedy vs epsilon-greedy table (density 0.2),
                            (3) significance tests
  significance.csv          all pairwise tests (overall and per difficulty group)
Zero-shot size transfer (10x10-trained models evaluated on N x N maps, see evaluate.py --size):
  python3 aggregate.py --runs results/runs/N15 --out results/summary_N15.csv
Retrained on N x N (run_full.sh with N=15/20, results/runs_trained/N<N>):
  python3 aggregate.py --runs results/runs_trained/N15 --out results/summary_trained_N15.csv  (extra train_size column)
  -> summary_N15.csv / summary_N15.md / significance_N15.csv (the significance file name follows the --out name:
     "summary" -> "significance"), with an extra eval_size column; tests are within that N only, Holm over the 4
     primary pairs of that N (same procedure as 10x10). The 10x10 outputs (summary.csv etc.) are unchanged.

Significance tests (pairs: DQN k5 vs DRQN k5, DRQN k3 vs k5, k5 vs k7, k3 vs k7):
  * Welch t-test on the per-seed success rates (n = 5 vs 5, unequal variances).
  * Multiple comparisons: the PRIMARY family = between-config pairs, greedy evaluation, "overall" group;
    Holm-Bonferroni adjusted p-values (holm_p) are reported for it. Every other row (difficulty subgroups,
    eps-greedy evaluations, greedy vs eps-greedy) is EXPLORATORY and reported unadjusted.
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
from common import RESULTS_DIR, refuse_protected_thesis_output

DIFFS = ("easy", "medium", "hard")
PAIRS = [(("dqn", 5), ("drqn", 5)), (("drqn", 3), ("drqn", 5)), (("drqn", 5), ("drqn", 7)), (("drqn", 3), ("drqn", 7))]


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def tcrit(n):
    return float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(tcrit(len(x)) * x.std(ddof=1) / np.sqrt(len(x)))


def collect(runs="runs", with_evals=False, include_exploratory=False):
    groups = defaultdict(list)
    skipped = 0
    for f in glob.glob(f"{runs}/*/seed*/eval_*.json"):
        e = json.load(open(f))
        # Cross-mode evals are labelled exploratory and stay out of the summary unless asked.
        if e.get("exploratory_cross_mode") and not include_exploratory:
            skipped += 1
            continue
        e["_file"] = f
        groups[(e["agent"], e["obs"], e["train_density"], e["eval_density"], e.get("eval_eps", 0.0), e.get("eval_size", 0),
                e.get("size", 10), e.get("obs_mode", "window"))].append(e)
    if skipped:
        print(f"aggregate: skipped {skipped} exploratory cross-mode eval file(s); not mixed into the table")
    rows = []
    include_mode_col = any(key[-1] != "window" for key in groups)
    for (agent, k, td, ed, ee, es_, ts, mode), es in sorted(groups.items()):
        es.sort(key=lambda e: e["seed"])
        row = {"agent": agent, "obs": k, "train_density": td, "eval_density": ed, "eval_eps": ee}
        # Only when a non-window mode is present, so a window-only summary keeps its columns.
        if include_mode_col:
            row["obs_mode"] = mode
        # extra columns only for non-10x10 settings; the 10x10 summary.csv keeps its original columns
        if ts != 10:
            row["train_size"] = ts  # trained (and evaluated) on N x N maps (results/runs_trained/N<N>)
        if es_:  # size-transfer evaluation
            row["eval_size"] = es_
        row["n_seeds"] = len(es)
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
                    "diff_A_minus_B": xa.mean() - xb.mean(), "t_test": "paired t (same models)" if same_seeds else "Welch",
                    "t_stat": float(t), "t_p": float(p),
                    "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_p": pb,
                    "family": "primary" if (grp == "overall" and not same_seeds and ee_a == 0 and ee_b == 0) else "exploratory",
                    "holm_p": float("nan"),
                    "significant_0.05": bool(p < 0.05 and (lo > 0 or hi < 0))})
    return out


def significance(rows, density=0.2):
    """Within one obs_mode, identical to the historical tests. Mixed modes are split first.

    Cross-mode pairs are appended only when more than one mode is present, and they are
    exploratory (not part of the Holm family). `main` refuses a mixed folder unless asked.
    """
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get("obs_mode", "window")].append(r)
    if len(buckets) <= 1:
        return _significance_one(rows, density)
    out = []
    for mode in sorted(buckets):
        out.extend(_significance_one(buckets[mode], density))
    out.extend(_cross_mode_exploratory(buckets, density))
    return out


def _cross_mode_exploratory(buckets, density):
    modes = sorted(buckets)
    out = []
    for i, ma in enumerate(modes):
        idx_a = {(r["agent"], r["obs"], r["eval_eps"]): r for r in buckets[ma]
                 if abs(r["eval_density"] - density) < 1e-9 and abs(r["train_density"] - density) < 1e-9}
        for mb in modes[i + 1:]:
            idx_b = {(r["agent"], r["obs"], r["eval_eps"]): r for r in buckets[mb]
                     if abs(r["eval_density"] - density) < 1e-9 and abs(r["train_density"] - density) < 1e-9}
            for key in sorted(set(idx_a) & set(idx_b)):
                agent_k = (key[0], key[1])
                tests = _test(idx_a[key]["_evals"], idx_b[key]["_evals"], key[2], key[2], agent_k, agent_k)
                for s in tests:
                    s["family"] = "exploratory"
                    s["holm_p"] = float("nan")
                    s["A"] = f"{s['A']}:{ma}"
                    s["B"] = f"{s['B']}:{mb}"
                    s["significant_0.05"] = bool(s["t_p"] < 0.05 and (s["boot_ci_lo"] > 0 or s["boot_ci_hi"] < 0))
                out.extend(tests)
    return out


def _significance_one(rows, density=0.2):
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
    prim = [s for s in out if s["family"] == "primary"]
    if prim:
        for s_, hp in zip(prim, holm([s["t_p"] for s in prim])):
            s_["holm_p"] = float(hp)
            s_["significant_0.05"] = bool(hp < 0.05 and (s_["boot_ci_lo"] > 0 or s_["boot_ci_hi"] < 0))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs", default=os.path.join(RESULTS_DIR, "runs"))
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "summary.csv"))
    p.add_argument("--density", type=float, default=0.2, help="density for the eps table and significance tests")
    p.add_argument("--allow_mixed_obs_modes", action="store_true",
                   help="write one table that contains more than one obs_mode. Cross-mode tests are exploratory. "
                        "Will not overwrite results/summary.csv.")
    a = p.parse_args()
    rows = collect(a.runs, with_evals=True)
    if not rows:
        raise SystemExit("no eval files to aggregate (exploratory cross-mode files are left out of the table)")
    modes = sorted({r.get("obs_mode", "window") for r in rows})
    if len(modes) > 1 and not a.allow_mixed_obs_modes:
        raise SystemExit(
            "refusing to aggregate mixed obs_mode values (" + ", ".join(modes) + "). "
            "Window-mode thesis results and line-of-sight results must stay in separate summaries. "
            "Pass --allow_mixed_obs_modes and an --out path other than results/summary.csv."
        )
    if any(m != "window" for m in modes):
        refuse_protected_thesis_output(a.out, "summary.csv")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sig = significance(rows, a.density)
    plain = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plain[0])); w.writeheader(); w.writerows(plain)
    sig_path = os.path.join(os.path.dirname(a.out) or ".", os.path.basename(a.out).replace("summary", "significance", 1))
    if not os.path.basename(sig_path).startswith("significance"):
        sig_path = os.path.join(os.path.dirname(a.out) or ".", "significance.csv")
    sizes = sorted({r["eval_size"] for r in rows if "eval_size" in r})
    tsizes = sorted({r["train_size"] for r in rows if "train_size" in r and "eval_size" not in r})
    if sig:
        with open(sig_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sig[0])); w.writeheader()
            w.writerows([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()} for s in sig])

    nmaps = sorted({e["n_maps"] for r in rows for e in r["_evals"]})
    nmaps = str(nmaps[0]) if len(nmaps) == 1 else "/".join(map(str, nmaps))
    nseeds = sorted({r["n_seeds"] for r in rows if r["agent"] != "random"} or {5})
    nseeds = str(nseeds[0]) if len(nseeds) == 1 else "/".join(map(str, nseeds))
    f2 = lambda m, c: "n/a" if m is None else (f"{m:.3f} ± {c:.3f}" if c == c else f"{m:.3f}")
    show_mode = any("obs_mode" in r for r in rows)
    mode_note = (["**观测模式**：本表包含视线遮挡（`obs_mode=los`），或与 window 模式混排。"
                  "论文里已经发表的成功率（例如 10×10 上 DQN k5 贪心 0.689、DRQN k5 贪心 0.926）是 **window 模式、无遮挡** 的结果，"
                  "不是本表里的遮挡结果。跨观测模式的比较若出现，一律标为探索性分析，不进入主假设族的 Holm 校正。", ""]
                 if show_mode else [])
    md = ["# 结果汇总 / Results summary", "",
          f"所有数值为 {nseeds} 个种子的均值 ± 95% 置信区间（Student t，n=种子数）；测试集为固定的 {nmaps} 张地图（seed 10,000,000+i）。", "",
          *mode_note,
          *([f"**零样本尺寸迁移**：模型均在 10×10 上训练、未重新训练，测试地图尺寸 N={'/'.join(map(str, sizes))}"
             f"（密度 0.2，起终点曼哈顿距离 ≥7，max_steps=8N，目标偏移 /(N−1) 归一化）。显著性检验仅在同一 N 内进行。", ""] if sizes else []),
          *([f"**在大地图上重新训练**：模型在 N={'/'.join(map(str, tsizes))} 上训练并在同尺寸测试（超参数、网络、奖励、10000 回合与 10×10 相同；"
             f"密度 0.2，起终点曼哈顿距离 ≥7，max_steps=8N，目标偏移 /(N−1) 归一化，训练与测试一致）。显著性检验仅在同一 N 内进行。", ""] if tsizes else []),
          "## 1. 贪心评估（ε=0，主协议） / Greedy evaluation", "",
          ("| agent | k | obs_mode | eval density | seeds | success (mean ± 95% CI) | easy | medium | hard | collision | timeout | avg steps (succ.) |"
           if show_mode else
           "| agent | k | eval density | seeds | success (mean ± 95% CI) | easy | medium | hard | collision | timeout | avg steps (succ.) |"),
          ("|---|---|---|---|---|---|---|---|---|---|---|---|" if show_mode else
           "|---|---|---|---|---|---|---|---|---|---|---|")]
    for r in rows:
        if r["eval_eps"] != 0:
            continue
        mode_col = f" {r.get('obs_mode', 'window')} |" if show_mode else ""
        md.append(f"| {r['agent']} | {r['obs']} |{mode_col} {r['eval_density']} | {r['n_seeds']} | {f2(r['success_rate'], r['success_rate_ci95'])} | "
                  f"{f2(r['success_easy'], r['success_easy_ci95'])} | {f2(r['success_medium'], r['success_medium_ci95'])} | "
                  f"{f2(r['success_hard'], r['success_hard_ci95'])} | {f2(r['collision_rate'], r['collision_rate_ci95'])} | "
                  f"{f2(r['timeout_rate'], r['timeout_rate_ci95'])} | {f2(r['avg_steps_success'], r['avg_steps_success_ci95'])} |")
    md += ["", f"## 2. 贪心 vs ε-贪心评估（density {a.density}） / Greedy vs epsilon-greedy evaluation", "",
           ("| agent | k | obs_mode | eval ε | success | easy | medium | hard | collision | timeout | avg steps (succ.) |"
            if show_mode else
            "| agent | k | eval ε | success | easy | medium | hard | collision | timeout | avg steps (succ.) |"),
           ("|---|---|---|---|---|---|---|---|---|---|---|" if show_mode else
            "|---|---|---|---|---|---|---|---|---|---|")]
    for r in sorted(rows, key=lambda r: (r.get("obs_mode", "window"), r["agent"], r["obs"], r["eval_eps"])):
        if abs(r["eval_density"] - a.density) > 1e-9 or r["agent"] == "random":
            continue
        mode_col = f" {r.get('obs_mode', 'window')} |" if show_mode else ""
        md.append(f"| {r['agent']} | {r['obs']} |{mode_col} {r['eval_eps']:g} | {f2(r['success_rate'], r['success_rate_ci95'])} | "
                  f"{f2(r['success_easy'], r['success_easy_ci95'])} | {f2(r['success_medium'], r['success_medium_ci95'])} | "
                  f"{f2(r['success_hard'], r['success_hard_ci95'])} | {f2(r['collision_rate'], r['collision_rate_ci95'])} | "
                  f"{f2(r['timeout_rate'], r['timeout_rate_ci95'])} | {f2(r['avg_steps_success'], r['avg_steps_success_ci95'])} |")
    md += ["", f"## 3. 显著性检验（density {a.density}） / Significance tests", "",
           f"t 检验：以每个种子的成功率为样本（n={nseeds} vs {nseeds}；配置间用 Welch t，同一配置 ε-贪心 vs 贪心用配对 t）。配对 bootstrap：两层重采样"
           f"（{nmaps} 张共享测试地图配对重采样 + 各配置种子独立重采样，10,000 次），给出成功率差 A−B 的 95% 百分位置信区间与双侧 bootstrap p 值。", "",
           "**主假设族（primary）** = 配置间比较 × 贪心评估 × overall 组，对该族做 Holm 校正（列 Holm p）；“显著”= Holm p<0.05 且 bootstrap CI 不含 0。"
           "其余行（难度分组、ε-贪心评估、ε-贪心 vs 贪心）为**探索性分析**，报告未校正 p 值，“显著”= p<0.05 且 bootstrap CI 不含 0。"
           "“未检出差异”不等于“无差异”，请结合 CI 解读（例如 k5 vs k7 的 CI 给出了差异的可能范围）。", "",
           "| eval ε | A | B | group | family | n maps | A | B | A−B | t p | Holm p | bootstrap 95% CI of A−B | bootstrap p | 结论 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bp = lambda p: "<1e-4" if p == 0 else f"{p:.2g}"  # 10,000 replicates -> resolution 1e-4
    for s in sig:
        verdict = "显著" if s["significant_0.05"] else "未检出差异"
        ee = s["eval_eps"] if isinstance(s["eval_eps"], str) else f"{s['eval_eps']:g}"
        hp = f"{s['holm_p']:.2g}" if s["holm_p"] == s["holm_p"] else "–"
        md.append(f"| {ee} | {s['A']} | {s['B']} | {s['group']} | {s['family']} | {s['n_maps']} | {s['mean_A']:.3f} | {s['mean_B']:.3f} | "
                  f"{s['diff_A_minus_B']:+.3f} | {s['t_p']:.2g} | {hp} | [{s['boot_ci_lo']:+.3f}, {s['boot_ci_hi']:+.3f}] | "
                  f"{bp(s['boot_p'])} | {verdict} |")
    open(a.out.replace(".csv", ".md"), "w").write("\n".join(md) + "\n")
    fmt = lambda m, c: "  n/a " if m is None else f"{m:.3f}±{c:.3f}"
    for r in rows:
        print(f"{r['agent']:7s} k={r['obs']} eval_d={r['eval_density']} eps={r['eval_eps']:g} seeds={r['n_seeds']} "
              f"succ={fmt(r['success_rate'], r['success_rate_ci95'])} | easy={fmt(r['success_easy'], r['success_easy_ci95'])} "
              f"med={fmt(r['success_medium'], r['success_medium_ci95'])} hard={fmt(r['success_hard'], r['success_hard_ci95'])}")
    for s in sig:
        if s["group"] == "overall":
            print(f"eps={s['eval_eps']} {s['A']} vs {s['B']} [{s['family']}]: diff={s['diff_A_minus_B']:+.3f} t_p={s['t_p']:.3g} "
                  f"holm_p={s['holm_p']:.3g} boot CI=[{s['boot_ci_lo']:+.3f},{s['boot_ci_hi']:+.3f}] boot_p={s['boot_p']:.3g}")
    print("saved", a.out, ".md and", sig_path)


if __name__ == "__main__":
    main()
