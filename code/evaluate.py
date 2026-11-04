"""Greedy evaluation on the FIXED held-out test set (map seeds TEST_SEED_BASE + i, i < n_maps),
identical for every method/seed. Results are also split into easy/medium/hard by BFS shortest-path length.

--run may be a single seed dir (contains config.json) or a parent dir containing seed*/ subdirs.
"""
import argparse, csv, glob, json, os
import numpy as np
import torch
from common import load_run, make_env
from envs.grid_env import TEST_SEED_BASE, corridor_score


def difficulty(shortest, bins):
    return "easy" if shortest <= bins[0] else ("medium" if shortest <= bins[1] else "hard")


def summarize(eps):
    n = len(eps)
    if n == 0:
        return {"n": 0}
    succ = [e for e in eps if e["outcome"] == "success"]
    return {"n": n,
            "success_rate": len(succ) / n,
            "collision_rate": sum(e["outcome"] == "collision" for e in eps) / n,
            "timeout_rate": sum(e["outcome"] == "timeout" for e in eps) / n,
            "avg_steps_success": float(np.mean([e["steps"] for e in succ])) if succ else None,
            "avg_path_ratio_success": float(np.mean([e["steps"] / e["shortest"] for e in succ])) if succ else None}


def evaluate(run_dir, n_maps=300, density=None, bins=(9, 11), max_failures=10**9):
    cfg, agent = load_run(run_dir)
    ecfg = dict(cfg)
    if density is not None:
        ecfg["density"] = density
    env = make_env(ecfg)
    rng = np.random.default_rng(1000 + cfg["seed"])  # only used by stochastic (random) policy
    eps, failures = [], []
    for i in range(n_maps):
        o, info = env.reset(seed=TEST_SEED_BASE + i)
        agent.reset_state()
        done = False
        while not done:
            o, r, term, trunc, info = env.step(agent.act(o, 0.0, rng))
            done = term or trunc
        snap = env.snapshot()
        rec = {"map_seed": TEST_SEED_BASE + i, "outcome": info["outcome"], "steps": info["t"],
               "shortest": info["shortest"], "difficulty": difficulty(info["shortest"], bins),
               "corridor_score": corridor_score(snap["grid"], snap["trajectory"])}
        eps.append(rec)
        if rec["outcome"] != "success" and len(failures) < max_failures:
            failures.append({**rec, **snap})
    res = {"run": run_dir, "agent": cfg["agent"], "seed": cfg["seed"], "size": cfg["size"], "obs": cfg["obs"],
           "train_density": cfg["density"], "eval_density": ecfg["density"], "n_maps": n_maps,
           "difficulty_bins_shortest_path": list(bins), **summarize(eps),
           "by_difficulty": {d: summarize([e for e in eps if e["difficulty"] == d]) for d in ("easy", "medium", "hard")}}
    tag = f"d{ecfg['density']}"
    json.dump(res, open(os.path.join(run_dir, f"eval_{tag}.json"), "w"), indent=2)
    json.dump(failures, open(os.path.join(run_dir, f"failures_{tag}.json"), "w"))
    with open(os.path.join(run_dir, f"episodes_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0])); w.writeheader(); w.writerows(eps)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_maps", type=int, default=300)
    p.add_argument("--density", type=float, default=None, help="override density (difficulty-by-density / generalization)")
    p.add_argument("--bins", type=int, nargs=2, default=[9, 11], help="shortest-path thresholds: easy<=a<medium<=b<hard")
    a = p.parse_args()
    torch.set_num_threads(1)
    dirs = [a.run] if os.path.exists(os.path.join(a.run, "config.json")) else sorted(glob.glob(os.path.join(a.run, "seed*")))
    for d in dirs:
        r = evaluate(d, a.n_maps, a.density, tuple(a.bins))
        print(f"{d}: success={r['success_rate']:.3f} collision={r['collision_rate']:.3f} timeout={r['timeout_rate']:.3f} "
              + " ".join(f"{k}={v.get('success_rate', float('nan')):.2f}(n={v['n']})" for k, v in r["by_difficulty"].items()))
