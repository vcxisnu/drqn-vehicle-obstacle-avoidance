"""Compare models RETRAINED on N x N (results/runs_trained_eval/N<N>) with the zero-shot transfer of the 10x10 models
(results/runs/N<N>) on the SAME 500 N x N test maps, plus the 10x10 in-distribution reference (results/runs).

Per config / eval eps: mean success over 5 seeds +- 95% CI (Student t), difference retrained - zero-shot and an
exploratory Welch t-test on per-seed success (different models -> unpaired; not Holm-corrected, not in the primary family).
  python3 compare_trained_transfer.py  ->  results/trained_vs_transfer.{csv,md}
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CFGS = [("dqn", 5), ("drqn", 3), ("drqn", 5), ("drqn", 7)]


def per_seed(root, cfg, eps):
    out = []
    for f in sorted(glob.glob(os.path.join(root, cfg, "seed*", "eval_d0.2*.json"))):
        j = json.load(open(f))
        if abs(j.get("eval_eps", 0.0) - eps) < 1e-9:
            out.append(j["success_rate"])
    return np.array(out)


def ci(x):
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[15, 20])
    p.add_argument("--out", default=os.path.join(R, "trained_vs_transfer.csv"))
    a = p.parse_args()
    rows = []
    for N in a.sizes:
        for ag, k in CFGS:
            for e in (0.0, 0.01):
                t = per_seed(os.path.join(R, "runs_trained_eval", f"N{N}"), f"{ag}_s{N}_k{k}_d0.2", e)
                z = per_seed(os.path.join(R, "runs", f"N{N}"), f"{ag}_s10_k{k}_d0.2", e)
                b = per_seed(os.path.join(R, "runs"), f"{ag}_s10_k{k}_d0.2", e)
                if not len(t) or not len(z):
                    continue
                pv = stats.ttest_ind(t, z, equal_var=False).pvalue
                rows.append(dict(N=N, agent=ag, k=k, eval_eps=e, n_trained=len(t), n_transfer=len(z),
                                 retrained=round(t.mean(), 4), retrained_ci=round(ci(t), 4),
                                 zero_shot=round(z.mean(), 4), zero_shot_ci=round(ci(z), 4),
                                 diff_retrained_minus_zero_shot=round(t.mean() - z.mean(), 4), welch_p=round(float(pv), 4),
                                 ref_10x10=round(b.mean(), 4) if len(b) else float("nan"),
                                 ref_10x10_ci=round(ci(b), 4) if len(b) else float("nan")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    keys = list(rows[0])
    with open(a.out, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    md = ["# 大地图：重新训练 vs 零样本迁移 / Retrained vs zero-shot transfer", "",
          "同一批 N×N 测试地图（500 张，seed 10,000,000+i）；成功率为 5 个种子均值 ± 95% CI。差值 = 重新训练 − 零样本迁移；"
          "p 为按种子成功率的 Welch t 检验（探索性，未做多重校正）。10×10 列为原 10×10 模型在 10×10 测试集上的结果（参考）。", "",
          "| N | config | eval ε | 重新训练 | 零样本迁移 | 差值 | Welch p | 10×10 参考 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['N']} | {r['agent']}_k{r['k']} | {r['eval_eps']:g} | {r['retrained']:.3f} ± {r['retrained_ci']:.3f} | "
                  f"{r['zero_shot']:.3f} ± {r['zero_shot_ci']:.3f} | {r['diff_retrained_minus_zero_shot']:+.3f} | {r['welch_p']:.2g} | "
                  f"{r['ref_10x10']:.3f} ± {r['ref_10x10_ci']:.3f} |")
    open(os.path.splitext(a.out)[0] + ".md", "w").write("\n".join(md) + "\n")
    print("saved", a.out)


if __name__ == "__main__":
    main()
