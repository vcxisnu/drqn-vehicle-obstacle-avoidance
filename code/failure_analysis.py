"""Failure-case analysis on failures_d<density>[_eps<e>].json files.

(A) SINGLE, MUTUALLY EXCLUSIVE label per failed episode (first matching rule wins, priority order below).
    "tail" = the last min(40, T//2) steps of the trajectory (T = steps taken; for a timeout T = max_steps = 80 -> 40).
    "match_L" = fraction of tail steps t with pos[t] == pos[t-L] (L-periodicity of the tail).
    Dead-end pocket: every dead-end cell (free, >=3 of 4 neighbours blocked, border counts as blocked, goal excluded)
    plus the 1-wide corridor cells leading to it (free cells with exactly 2 free neighbours), up to the first junction.
    Narrow cell: free cell with >=2 blocked neighbours (border counts as blocked).

    1 collision            episode ended by moving into an obstacle
    2 border_bump_stuck    match_1 >= 0.8: agent keeps bumping the map border (position unchanged)
    3 dead_end_stuck       the last 20 positions all lie inside dead-end pockets (entered a dead end, never got out)
    4 two_cell_osc         match_2 >= 0.8: period-2 back-and-forth A-B-A-B dominates the tail
    5 long_cycle           match_L >= 0.8 for some L in 3..8 (repeating cycle of length 3-8), OR some cell is visited
                           >= 3 times within the tail (quasi-cycle that is not strictly periodic)
    6 narrow_corridor      >= 50% of tail positions are narrow cells (wandering in/around 1-wide corridors)
    7 u_trap               at the trajectory point Manhattan-closest to the goal, BFS distance >= 2 x Manhattan + 2
                           ("goal behind a wall"; threshold kept from v1, heuristic)
    8 dead_end_exited      the trajectory entered a dead-end pocket at some point but left it again
    9 other_timeout        any remaining timeout

    Rationale of the order: terminal cause first; then "physically trapped" (dead_end_stuck) before the generic
    loop shapes it would otherwise also match; then loop shapes (the behavioural signature of a deterministic
    memoryless policy under perceptual aliasing); then map-structure explanations for non-cyclic timeouts.

    Secondary context (also mutually exclusive, for timeouts only; WHERE the agent got stuck, see context_of()):
    ctx_dead_end (>=50% of tail in dead-end pockets) > ctx_u_trap > ctx_narrow_corridor (>=50% of tail narrow) > ctx_open.
    In practice labels 6-9 rarely/never fire (a 40-step tail of an 80-step timeout almost always revisits a cell),
    so the context columns carry the map-structure information.

    Info column info_timeouts_strictly_periodic_pct: share of timeouts whose tail is >=80% L-periodic, L in 1..8
    (an exact deterministic cycle; for a deterministic memoryless policy every timeout must end in such a cycle).

(B) The OLD broad multi-label tags of v1 are still reported for transparency (percent of failures carrying each tag;
    tags overlap so they do not sum to 100%):
    loop (steps/unique cells >= 2), dead_end (visited any dead-end cell), u_trap (as above),
    narrow_corridor (>= 50% of distinct visited cells narrow).
"""
import argparse, csv, glob, json, os
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.grid_env import ACTIONS, bfs_dist, corridor_score
from common import RESULTS_DIR

LABELS = ["collision", "border_bump_stuck", "dead_end_stuck", "two_cell_osc", "long_cycle", "narrow_corridor",
          "u_trap", "dead_end_exited", "other_timeout"]
OLD_TAGS = ["timeout", "loop", "dead_end", "u_trap", "narrow_corridor"]


def is_free(g, r, c):
    n = len(g)
    return 0 <= r < n and 0 <= c < n and g[r][c] == 0


def n_blocked(g, r, c):
    return sum(1 for dr, dc in ACTIONS if not is_free(g, r + dr, c + dc))


def dead_end_pockets(g, goal):
    pocket = set()
    for r, c in np.argwhere(g == 0):
        p = (int(r), int(c))
        if p == goal or n_blocked(g, *p) < 3:
            continue
        prev, cur = None, p
        while True:
            pocket.add(cur)
            nb = [(cur[0] + dr, cur[1] + dc) for dr, dc in ACTIONS if is_free(g, cur[0] + dr, cur[1] + dc)]
            nxt = [q for q in nb if q != prev]
            if len(nxt) != 1:
                break
            q = nxt[0]
            if q == goal or 4 - n_blocked(g, *q) != 2:  # q is a junction (>=3 free nbrs) -> pocket ends before it
                break
            prev, cur = cur, q
    return pocket


def u_trap(g, traj, goal):
    free_traj = [p for p in traj if is_free(g, *p)]
    d = bfs_dist(g, goal)
    man = [abs(p[0] - goal[0]) + abs(p[1] - goal[1]) for p in free_traj]
    i = int(np.argmin(man))
    return d[free_traj[i]] >= 2 * man[i] + 2


def label_of(f):
    """Single mutually exclusive label (see module docstring)."""
    if f["outcome"] == "collision":
        return "collision"
    g = np.array(f["grid"]); traj = [tuple(p) for p in f["trajectory"]]; goal = tuple(f["goal"])
    T = len(traj) - 1
    tl = min(40, T // 2)
    tail_idx = range(T - tl + 1, T + 1)
    match = lambda L: np.mean([traj[t] == traj[t - L] for t in tail_idx])
    if match(1) >= 0.8:
        return "border_bump_stuck"
    pockets = dead_end_pockets(g, goal)
    if pockets and all(p in pockets for p in traj[-20:]):
        return "dead_end_stuck"
    if match(2) >= 0.8:
        return "two_cell_osc"
    tail = traj[T - tl + 1:]
    if any(match(L) >= 0.8 for L in range(3, 9)) or max(Counter(tail).values()) >= 3:
        return "long_cycle"
    if np.mean([n_blocked(g, *p) >= 2 for p in tail]) >= 0.5:
        return "narrow_corridor"
    if u_trap(g, traj, goal):
        return "u_trap"
    if any(p in pockets for p in traj):
        return "dead_end_exited"
    return "other_timeout"


CONTEXTS = ["ctx_dead_end", "ctx_u_trap", "ctx_narrow_corridor", "ctx_open"]


def context_of(f):
    """Secondary, also mutually exclusive: WHERE a timeout got stuck (labels 2-5 almost always fire first because
    a 40-step tail in 80 steps nearly always revisits cells). Priority: >=50% of tail in dead-end pockets >
    u_trap criterion > >=50% of tail on narrow cells > open area."""
    g = np.array(f["grid"]); traj = [tuple(p) for p in f["trajectory"]]; goal = tuple(f["goal"])
    T = len(traj) - 1; tail = traj[T - min(40, T // 2) + 1:]
    pockets = dead_end_pockets(g, goal)
    if np.mean([p in pockets for p in tail]) >= 0.5:
        return "ctx_dead_end"
    if u_trap(g, traj, goal):
        return "ctx_u_trap"
    if np.mean([n_blocked(g, *p) >= 2 for p in tail]) >= 0.5:
        return "ctx_narrow_corridor"
    return "ctx_open"


def strictly_periodic(f):
    """Info only: tail is >=80% L-periodic for some L in 1..8 (a deterministic cycle)."""
    if f["outcome"] != "timeout":
        return False
    traj = [tuple(p) for p in f["trajectory"]]; T = len(traj) - 1; tl = min(40, T // 2)
    return any(np.mean([traj[t] == traj[t - L] for t in range(T - tl + 1, T + 1)]) >= 0.8 for L in range(1, 9))


def old_tags(f):
    """v1 broad multi-label tags (kept for transparency)."""
    g = np.array(f["grid"]); traj = [tuple(p) for p in f["trajectory"]]; goal = tuple(f["goal"])
    tags = {f["outcome"]}
    uniq = {p for p in traj if is_free(g, *p)}
    if len(traj) / max(1, len(uniq)) >= 2.0:
        tags.add("loop")
    if any(n_blocked(g, *p) >= 3 and p != goal for p in uniq):
        tags.add("dead_end")
    if u_trap(g, traj, goal):
        tags.add("u_trap")
    if corridor_score(f["grid"], f["trajectory"]) >= 0.5:
        tags.add("narrow_corridor")
    return tags


def render(f, title, path):
    g = np.array(f["grid"]); n = len(g)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(g, cmap="Greys", vmin=0, vmax=1.5)
    pk = dead_end_pockets(g, tuple(f["goal"]))
    for r, c in pk:
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="orange", alpha=0.25, lw=0))
    t = np.array(f["trajectory"])
    jit = np.random.default_rng(0).uniform(-.12, .12, (len(t), 2))
    ax.plot(t[:, 1] + jit[:, 1], t[:, 0] + jit[:, 0], "-", color="tab:blue", alpha=0.6, lw=1.2)
    ax.plot(f["start"][1], f["start"][0], "go", ms=10, label="start")
    ax.plot(f["goal"][1], f["goal"][0], "r*", ms=14, label="goal")
    ax.plot(t[-1, 1], t[-1, 0], "bx", ms=10, label="end")
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", lw=0.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8); ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(RESULTS_DIR, "runs"))
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "failures"))
    p.add_argument("--density", default="0.2")
    p.add_argument("--eval_eps", default="0", help="analyse failures of the eps-greedy evaluation with this eps (0 = greedy)")
    p.add_argument("--render_per_config", type=int, default=4)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    suffix = "" if float(a.eval_eps) == 0 else f"_eps{float(a.eval_eps):g}"
    rows, rng = [], np.random.default_rng(0)
    for cdir in sorted(glob.glob(f"{a.runs}/*/")):
        name = os.path.basename(cdir.rstrip("/"))
        if name.startswith("random"):
            continue
        lab_c, tag_c, n_eval, n_fail, n_per, n_to = Counter(), Counter(), 0, 0, 0, 0
        ctx_c = Counter()
        examples = defaultdict(list)
        for fpath in sorted(glob.glob(f"{cdir}/seed*/failures_d{a.density}{suffix}.json")):
            ev = json.load(open(fpath.replace("failures_", "eval_")))
            n_eval += ev["n_maps"]
            for f in json.load(open(fpath)):
                lab = label_of(f)
                n_fail += 1; lab_c[lab] += 1; tag_c.update(old_tags(f))
                n_to += f["outcome"] == "timeout"; n_per += strictly_periodic(f)
                if f["outcome"] == "timeout":
                    ctx_c[context_of(f)] += 1
                examples[lab].append((os.path.basename(os.path.dirname(fpath)), f))
        if not n_eval:
            continue
        row = {"config": name, "eval_eps": float(a.eval_eps), "test_episodes": n_eval, "failures": n_fail,
               "failure_rate": round(n_fail / n_eval, 4)}
        for k in LABELS:
            row[f"label_{k}"] = lab_c[k]
        for k in LABELS:
            row[f"label_{k}_pct"] = round(100 * lab_c[k] / max(1, n_fail), 1)
        for k in CONTEXTS:
            row[f"{k}_pct_of_timeouts"] = round(100 * ctx_c[k] / max(1, n_to), 1)
        row["info_timeouts_strictly_periodic_pct"] = round(100 * n_per / max(1, n_to), 1)
        for k in OLD_TAGS:
            row[f"oldtag_{k}_pct_of_failures"] = round(100 * tag_c[k] / max(1, n_fail), 1)
        rows.append(row)
        if suffix:  # only render examples for the main (greedy) analysis
            continue
        cats = sorted(examples, key=lambda c: -len(examples[c]))
        for i in range(a.render_per_config):
            if not cats:
                break
            c = cats[i % len(cats)]
            seed, f = examples[c][rng.integers(len(examples[c]))]
            render(f, f"{name} {seed} map{f['map_seed']}\n{f['outcome']} label={c}\nold tags={','.join(sorted(old_tags(f)))}",
                   f"{a.out}/{name}_{seed}_map{f['map_seed']}_{c}.png")
    out = f"{a.out}/failure_summary{suffix}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["config"], f"eps={r['eval_eps']:g}", f"fail={r['failures']}/{r['test_episodes']}",
              {k: r[f"label_{k}_pct"] for k in LABELS if r[f"label_{k}"]}, "periodic%", r["info_timeouts_strictly_periodic_pct"],
              {k: r[f"{k}_pct_of_timeouts"] for k in CONTEXTS})
    print("saved", out)


if __name__ == "__main__":
    main()
