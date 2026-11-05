"""Success rate stratified by DETOUR RATIO = d_BFS(start, goal) / Manhattan(start, goal) of the test map.

Rationale: memory matters on maps where the straight-line direction (given by the goal offset in the observation)
is blocked and the agent must go around; the shortest-path length alone mixes "long but straight" with "short but
winding". Ratio 1.0 = a Manhattan-shortest path exists; the larger the ratio, the longer the forced detour.

Works on EXISTING evaluation outputs: episodes_<tag>.csv only stores map_seed / outcome / shortest, so each test map
is regenerated from its seed (deterministic, numpy only) to recover start/goal. Newer episodes files already contain
`manhattan` and `detour_ratio` columns, which are used directly when present.

Outputs (default results/detour/):
  detour_by_config.csv   per (config, eval eps): success mean ± 95% t-CI over seeds for each detour bin, plus n maps
  detour_by_config.md    same as a markdown table
  detour.png             success vs detour bin (greedy evaluation, density 0.2)
"""
import argparse, csv, glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS_DIR, make_env
from envs.grid_env import TEST_SEED_BASE
from aggregate import ci95

BINS = [(1.0, 1.0, "=1.0 (straight)"), (1.0, 1.25, "(1.0, 1.25]"), (1.25, 1.5, "(1.25, 1.5]"), (1.5, 1e9, ">1.5")]


def bin_of(r):
    if r <= 1.0 + 1e-9:
        return BINS[0][2]
    for lo, hi, name in BINS[1:]:
        if lo < r <= hi:
            return name
    return BINS[-1][2]


def map_geometry(cfg, n_maps, cache):
    """map_seed -> (shortest, manhattan) by regenerating the test maps (density-independent for the key)."""
    key = (cfg["size"], cfg["density"], cfg.get("min_dist"))
    if key not in cache:
        env = make_env(cfg); d = {}
        for i in range(n_maps):
            _, info = env.reset(seed=TEST_SEED_BASE + i)
            s, g = info["start"], info["goal"]
            d[TEST_SEED_BASE + i] = (info["shortest"], abs(s[0] - g[0]) + abs(s[1] - g[1]))
        cache[key] = d
    return cache[key]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(RESULTS_DIR, "runs"))
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "detour"))
    p.add_argument("--density", type=float, default=0.2)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    cache, rows = {}, []
    names = [b[2] for b in BINS]
    for cdir in sorted(glob.glob(f"{a.runs}/*/")):
        name = os.path.basename(cdir.rstrip("/"))
        per_eps = defaultdict(list)  # eps -> list over seeds of {bin: success rate}
        counts = {}
        for ep_file in sorted(glob.glob(f"{cdir}/seed*/episodes_d{a.density:g}*.csv")):
            tag = os.path.basename(ep_file)[len("episodes_"):-4]
            eps = float(tag.split("_eps")[1]) if "_eps" in tag else 0.0
            recs = list(csv.DictReader(open(ep_file)))
            if not recs:
                continue
            if "detour_ratio" in recs[0]:  # always present for newer evaluations (incl. size-transfer dirs without config.json)
                ratios = {int(r["map_seed"]): float(r["detour_ratio"]) for r in recs}
            else:
                cfg = json.load(open(os.path.join(os.path.dirname(ep_file), "config.json")))
                ecfg = dict(cfg, density=a.density)
                geo = map_geometry(ecfg, len(recs), cache)
                ratios = {int(r["map_seed"]): geo[int(r["map_seed"])][0] / geo[int(r["map_seed"])][1] for r in recs}
            succ = defaultdict(list)
            for r in recs:
                succ[bin_of(ratios[int(r["map_seed"])])].append(r["outcome"] == "success")
            per_eps[eps].append({b: float(np.mean(v)) for b, v in succ.items()})
            counts = {b: len(v) for b, v in succ.items()}
        for eps, seeds in sorted(per_eps.items()):
            row = {"config": name, "eval_eps": eps, "n_seeds": len(seeds)}
            for b in names:
                vals = [s[b] for s in seeds if b in s]
                m, c = ci95(vals) if vals else (float("nan"), float("nan"))
                row[f"succ {b}"], row[f"ci95 {b}"], row[f"n_maps {b}"] = m, c, counts.get(b, 0)
            rows.append(row)
    if not rows:
        print("no episodes files found"); return
    with open(os.path.join(a.out, "detour_by_config.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    f2 = lambda m, c: "n/a" if m != m else (f"{m:.3f} ± {c:.3f}" if c == c else f"{m:.3f}")
    md = ["# 按绕行比分层的成功率 / Success by detour ratio", "",
          "绕行比 = BFS 最短路 / 起终点曼哈顿距离。=1.0 表示存在不绕行的最短路；越大表示目标方向被挡得越厉害，越需要记忆/绕行。"
          f"测试密度 {a.density}，均值 ± 95% CI（t 分布，跨种子）。", "",
          "| config | eval ε | " + " | ".join(f"{b} (n={rows[0][f'n_maps {b}']})" for b in names) + " |",
          "|---|---|" + "---|" * len(names)]
    for r in rows:
        md.append(f"| {r['config']} | {r['eval_eps']:g} | " + " | ".join(f2(r[f'succ {b}'], r[f'ci95 {b}']) for b in names) + " |")
    open(os.path.join(a.out, "detour_by_config.md"), "w").write("\n".join(md) + "\n")
    # plot: greedy only
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        if r["eval_eps"] != 0 or r["config"].startswith("random"):
            continue
        ax.errorbar(range(len(names)), [r[f"succ {b}"] for b in names], yerr=[r[f"ci95 {b}"] if r[f"ci95 {b}"] == r[f"ci95 {b}"] else 0 for b in names],
                    marker="o", capsize=3, label=r["config"].split("_s10")[0])
    ax.set_xticks(range(len(names))); ax.set_xticklabels([f"{b}\n(n={rows[0][f'n_maps {b}']})" for b in names], fontsize=8)
    ax.set(title=f"Held-out success vs detour ratio (greedy, density {a.density})", xlabel="detour ratio = d_BFS / Manhattan",
           ylabel="success rate", ylim=(0, 1)); ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "detour.png"), dpi=120); plt.close()
    for r in rows:
        if r["eval_eps"] == 0:
            print(r["config"], " ".join(f"{b}={r[f'succ {b}']:.3f}" for b in names))
    print("saved", a.out)


if __name__ == "__main__":
    main()
