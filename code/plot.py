"""Figures for the two experiments (mean ± 95% CI across seeds):
  plots/exp1_dqn_vs_drqn.png : training success curves (k=5) + held-out success by difficulty
  plots/exp2_obs_size.png    : DRQN (and DQN reference) held-out success vs window size 3/5/7
  plots/density.png          : held-out success vs evaluation obstacle density
  plots/eval_eps.png         : held-out success under greedy (eps=0) vs epsilon-greedy evaluation (eps=0.01/0.05)
All bands / error bars are Student-t 95% CIs over seeds (same as the summary tables).
"""
import argparse, csv, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from aggregate import collect, ci95, tcrit


def train_curves(runs, agent, k, metric="success", win=100):
    curves = []
    for log in glob.glob(f"{runs}/{agent}_*_k{k}_*/seed*/train_log.csv"):
        x = np.array([float(r[metric]) for r in csv.DictReader(open(log))])
        curves.append(np.convolve(x, np.ones(win) / win, mode="valid"))
    if not curves:
        return None
    n = min(map(len, curves)); c = np.array([x[:n] for x in curves])
    m = c.mean(0); s = c.std(0, ddof=1) / np.sqrt(len(c)) * tcrit(len(c)) if len(c) > 1 else np.zeros(n)  # t-CI, matches tables
    return m, s, len(c)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs", default="runs"); p.add_argument("--out", default="plots")
    p.add_argument("--k", type=int, default=5); p.add_argument("--density", type=float, default=0.2)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    all_rows = collect(a.runs)
    rows = [r for r in all_rows if r.get("eval_eps", 0) == 0]  # greedy evaluation = main protocol
    at = lambda r: abs(r["eval_density"] - a.density) < 1e-9 and abs(r["train_density"] - a.density) < 1e-9
    nz = lambda v: 0.0 if v is None or np.isnan(v) else v

    # ---- training curves (mean ± 95% CI band over seeds, moving average 100 episodes)
    for name, sets in (("exp1", [("dqn", a.k), ("drqn", a.k)]), ("exp2", [("drqn", 3), ("drqn", 5), ("drqn", 7)])):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for ag, k in sets:
            for ax_, metric in zip(axs, ("reward", "success")):
                c = train_curves(a.runs, ag, k, metric)
                if c:
                    m, s_, n = c; x = np.arange(len(m)) + 100
                    ax_.plot(x, m, label=f"{ag} k={k} (n={n})"); ax_.fill_between(x, m - s_, m + s_, alpha=0.2)
        axs[0].set(title=f"{name}: episode reward (MA100, mean ± 95% CI)", xlabel="episode", ylabel="reward")
        axs[1].set(title=f"{name}: success rate (MA100, mean ± 95% CI)", xlabel="episode", ylabel="success", ylim=(0, 1))
        for ax_ in axs: ax_.legend()
        plt.tight_layout(); plt.savefig(f"{a.out}/{name}_training_curves.png", dpi=120); plt.close()

    # ---- Experiment 1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for ag in ("dqn", "drqn", "tabular"):
        c = train_curves(a.runs, ag, a.k, "success")
        if c:
            m, s, n = c; x = np.arange(len(m))
            ax[0].plot(x, m, label=f"{ag} (n={n} seeds)"); ax[0].fill_between(x, m - s, m + s, alpha=0.2)
    ax[0].set(title=f"Exp1 training success (MA100), k={a.k}", xlabel="episode", ylabel="success rate"); ax[0].legend()
    cats = ["overall", "easy", "medium", "hard"]
    sel = [r for r in rows if at(r) and r["obs"] == a.k]
    for i, r in enumerate(sel):
        ms = [r["success_rate"]] + [r[f"success_{d}"] for d in cats[1:]]
        cs = [r["success_rate_ci95"]] + [r[f"success_{d}_ci95"] for d in cats[1:]]
        ax[1].bar(np.arange(4) + i * 0.8 / len(sel), [nz(v) for v in ms], 0.8 / len(sel),
                  yerr=[nz(v) for v in cs], capsize=3, label=f"{r['agent']} (n={r['n_seeds']})")
    ax[1].set_xticks(np.arange(4) + 0.4 - 0.4 / max(1, len(sel))); ax[1].set_xticklabels(cats)
    ax[1].set(title="Exp1 held-out success (mean ± 95% CI)", ylim=(0, 1)); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/exp1_dqn_vs_drqn.png", dpi=120); plt.close()

    # ---- Experiment 2
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ag in ("drqn", "dqn"):
        sel = sorted([r for r in rows if at(r) and r["agent"] == ag], key=lambda r: r["obs"])
        if sel:
            ax.errorbar([r["obs"] for r in sel], [r["success_rate"] for r in sel],
                        yerr=[nz(r["success_rate_ci95"]) for r in sel], marker="o", capsize=4,
                        label=ag + (" (reference)" if ag == "dqn" else ""))
    ax.set(title="Exp2: held-out success vs window size", xlabel="k (k x k window)", ylabel="success rate",
           xticks=[3, 5, 7], ylim=(0, 1)); ax.legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/exp2_obs_size.png", dpi=120); plt.close()

    # ---- density
    fig, ax = plt.subplots(figsize=(6, 4.5))
    g = defaultdict(list)
    for r in rows:
        if abs(r["train_density"] - a.density) < 1e-9:
            g[(r["agent"], r["obs"])].append(r)
    for (ag, k), sel in sorted(g.items()):
        sel.sort(key=lambda r: r["eval_density"])
        if len(sel) > 1:
            ax.errorbar([r["eval_density"] for r in sel], [r["success_rate"] for r in sel],
                        yerr=[nz(r["success_rate_ci95"]) for r in sel], marker="o", capsize=4, label=f"{ag} k={k}")
    ax.set(title=f"Held-out success vs density (trained at {a.density})", xlabel="obstacle density",
           ylabel="success rate", ylim=(0, 1)); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{a.out}/density.png", dpi=120); plt.close()
    # ---- greedy vs epsilon-greedy evaluation
    er = [r for r in all_rows if r["agent"] != "random" and abs(r["eval_density"] - a.density) < 1e-9
          and abs(r["train_density"] - a.density) < 1e-9]
    epss = sorted({r["eval_eps"] for r in er})
    if len(epss) > 1:
        cfgs = sorted({(r["agent"], r["obs"]) for r in er})
        fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
        for ax_, (cat, key) in zip(axs, (("overall", "success_rate"), ("easy", "success_easy"),
                                         ("medium", "success_medium"), ("hard", "success_hard"))):
            for j, e in enumerate(epss):
                sel = {(r["agent"], r["obs"]): r for r in er if r["eval_eps"] == e}
                ms = [nz(sel[c][key]) if c in sel else 0 for c in cfgs]
                cs = [nz(sel[c][key + "_ci95"]) if c in sel else 0 for c in cfgs]
                ax_.bar(np.arange(len(cfgs)) + j * 0.8 / len(epss), ms, 0.8 / len(epss), yerr=cs, capsize=3,
                        label=f"eval eps={e:g}" + (" (greedy)" if e == 0 else ""))
            ax_.set_xticks(np.arange(len(cfgs)) + 0.4 - 0.4 / len(epss))
            ax_.set_xticklabels([f"{g} k{k}" for g, k in cfgs], fontsize=8)
            ax_.set(title=f"{cat} (density {a.density})", ylim=(0, 1))
        axs[0].set_ylabel("held-out success (mean ± 95% CI)"); axs[0].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{a.out}/eval_eps.png", dpi=120); plt.close()
    print("saved plots to", a.out)


if __name__ == "__main__":
    main()
