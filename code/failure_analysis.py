"""Failure-case analysis on failures_d<density>.json files.

Tags per failed episode (multi-label; thresholds are heuristic, documented here):
  collision       : episode ended by moving into an obstacle
  timeout         : truncated at max_steps
  loop            : revisit ratio = steps / unique_cells >= 2.0
  dead_end        : trajectory entered a dead-end cell (free cell with >=3 blocked neighbours, border counts as blocked)
  u_trap          : at the point where the agent was Manhattan-closest to the goal, BFS distance >= 2x Manhattan distance + 2
                    (goal is "behind a wall": greedy direction-following gets stuck)
  narrow_corridor : >=50% of distinct visited cells are narrow (>=2 blocked neighbours)
Primary category priority: collision > u_trap > dead_end > narrow_corridor > loop > other_timeout.
"""
import argparse, csv, glob, json, os
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.grid_env import ACTIONS, bfs_dist, corridor_score


def n_blocked(g, r, c):
    n = len(g)
    return sum(1 for dr, dc in ACTIONS if not (0 <= r + dr < n and 0 <= c + dc < n) or g[r + dr][c + dc] == 1)


def tags_of(f):
    g = np.array(f["grid"]); traj = [tuple(p) for p in f["trajectory"]]; goal = tuple(f["goal"])
    tags = {f["outcome"]}
    free_traj = [p for p in traj if g[p] == 0]
    uniq = set(free_traj)
    if len(traj) / max(1, len(uniq)) >= 2.0:
        tags.add("loop")
    if any(n_blocked(g, *p) >= 3 and p != goal for p in uniq):
        tags.add("dead_end")
    d = bfs_dist(g, goal)
    man = [abs(p[0] - goal[0]) + abs(p[1] - goal[1]) for p in free_traj]
    p = free_traj[int(np.argmin(man))]
    if d[p] >= 2 * man[int(np.argmin(man))] + 2:
        tags.add("u_trap")
    if corridor_score(f["grid"], f["trajectory"]) >= 0.5:
        tags.add("narrow_corridor")
    for prim in ("collision", "u_trap", "dead_end", "narrow_corridor", "loop"):
        if prim in tags:
            return tags, prim
    return tags, "other_timeout"


def render(f, title, path):
    g = np.array(f["grid"]); n = len(g)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(g, cmap="Greys", vmin=0, vmax=1.5)
    t = np.array(f["trajectory"])
    ax.plot(t[:, 1] + np.random.uniform(-.12, .12, len(t)), t[:, 0] + np.random.uniform(-.12, .12, len(t)),
            "-", color="tab:blue", alpha=0.6, lw=1.2)
    ax.plot(f["start"][1], f["start"][0], "go", ms=10, label="start")
    ax.plot(f["goal"][1], f["goal"][0], "r*", ms=14, label="goal")
    ax.plot(t[-1, 1], t[-1, 0], "bx", ms=10, label="end")
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", lw=0.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8); ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="/workspace/thesis/results/runs")
    p.add_argument("--out", default="/workspace/thesis/results/failures")
    p.add_argument("--density", default="0.2")
    p.add_argument("--render_per_config", type=int, default=4)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows, rng = [], np.random.default_rng(0)
    for cdir in sorted(glob.glob(f"{a.runs}/*/")):
        name = os.path.basename(cdir.rstrip("/"))
        if name.startswith("random"):
            continue
        prim_c, tag_c, n_eval, n_fail = Counter(), Counter(), 0, 0
        examples = defaultdict(list)
        for fpath in sorted(glob.glob(f"{cdir}/seed*/failures_d{a.density}.json")):
            ev = json.load(open(fpath.replace("failures_", "eval_")))
            n_eval += ev["n_maps"]
            for f in json.load(open(fpath)):
                tags, prim = tags_of(f)
                n_fail += 1; prim_c[prim] += 1; tag_c.update(tags)
                examples[prim].append((os.path.basename(os.path.dirname(fpath)), f, sorted(tags)))
        if not n_eval:
            continue
        row = {"config": name, "test_episodes": n_eval, "failures": n_fail}
        for k in ("collision", "u_trap", "dead_end", "narrow_corridor", "loop", "other_timeout"):
            row[f"primary_{k}"] = prim_c[k]
        for k in ("timeout", "loop", "dead_end", "u_trap", "narrow_corridor"):
            row[f"tag_{k}_pct_of_failures"] = round(100 * tag_c[k] / max(1, n_fail), 1)
        rows.append(row)
        # render a few examples, spread over primary categories
        cats = [c for c in examples if examples[c]]
        for i in range(a.render_per_config):
            if not cats:
                break
            c = cats[i % len(cats)]
            seed, f, tags = examples[c][rng.integers(len(examples[c]))]
            render(f, f"{name} {seed} map{f['map_seed']}\n{f['outcome']} primary={c}\ntags={','.join(tags)}",
                   f"{a.out}/{name}_{seed}_map{f['map_seed']}_{c}.png")
    with open(f"{a.out}/failure_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
