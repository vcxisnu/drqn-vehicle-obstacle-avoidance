"""Evaluation (greedy by default; epsilon-greedy with --eval_eps > 0) on the FIXED held-out test set (map seeds TEST_SEED_BASE + i, i < n_maps),
identical for every method/seed.
--eval_eps e: with probability e a uniformly random action is taken (DRQN still feeds every observation through the
LSTM, so its memory is unaffected). e = 0 (default) reproduces the original greedy evaluation; output files for e > 0
get the suffix _eps<e> (e.g. eval_d0.2_eps0.01.json) so greedy and epsilon-greedy results coexist. Results are also split into easy/medium/hard by BFS shortest-path length.

--run may be a single seed dir (contains config.json) or a parent dir containing seed*/ subdirs.

Zero-shot size transfer (--size N): the model trained at its config size (10) is evaluated WITHOUT retraining on
N x N test maps (seeds TEST_SEED_BASE + i generated at size N; density, min start-goal distance 7 unchanged,
max_steps = 8N). Output then goes to --out (default <results>/runs/N<N>/<config>/seed<s>/) so the 10x10 files
under results/runs/<config>/seed<s>/ are never touched; a source.json there records which model was evaluated.

Observation mode comes from the checkpoint config (missing key = window, the published runs).
--obs_mode overrides it. A mismatch is exploratory cross-mode evaluation: results are written
outside the checkpoint directory and outside results/runs, and the json is marked
exploratory_cross_mode. Same-mode window evaluation keeps the historical output schema.
"""
import argparse, csv, glob, json, os
import numpy as np
import torch
from common import RESULTS_DIR, assert_separated_from_main_runs, load_run, make_env
from envs.grid_env import OBS_MODE_WINDOW, TEST_SEED_BASE, corridor_score, normalize_obs_mode


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


def evaluate(run_dir, n_maps=300, density=None, bins=(9, 11), max_failures=10**9, eval_eps=0.0, size=None, out_dir=None,
             obs_mode=None):
    cfg, agent = load_run(run_dir)
    train_mode = normalize_obs_mode(cfg.get("obs_mode"))
    eval_mode = train_mode if obs_mode is None else normalize_obs_mode(obs_mode)
    cross = eval_mode != train_mode
    if cross and (out_dir is None or os.path.abspath(out_dir) == os.path.abspath(run_dir)):
        raise RuntimeError(
            "cross-mode evaluation is exploratory and cannot write into the checkpoint directory "
            f"(train obs_mode={train_mode}, eval obs_mode={eval_mode}). Pass a separate out_dir."
        )
    dest = out_dir or run_dir
    if cross or eval_mode != OBS_MODE_WINDOW:
        assert_separated_from_main_runs(dest)
    ecfg = dict(cfg)
    ecfg["obs_mode"] = eval_mode
    if density is not None:
        ecfg["density"] = density
    if size is not None:
        ecfg["size"] = size  # max_steps (None in the configs) then defaults to 8 * size
    env = make_env(ecfg)
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    # used by the random policy and by epsilon-greedy evaluation; for eps > 0 the stream also depends on eps
    rng = np.random.default_rng(1000 + cfg["seed"]) if eval_eps == 0 else np.random.default_rng([1000 + cfg["seed"], int(round(eval_eps * 1e4))])
    eps, failures = [], []
    for i in range(n_maps):
        o, info = env.reset(seed=TEST_SEED_BASE + i)
        agent.reset_state()
        done = False
        while not done:
            o, r, term, trunc, info = env.step(agent.act(o, eval_eps, rng))
            done = term or trunc
        snap = env.snapshot()
        man = abs(info["start"][0] - info["goal"][0]) + abs(info["start"][1] - info["goal"][1])
        rec = {"map_seed": TEST_SEED_BASE + i, "outcome": info["outcome"], "steps": info["t"],
               "shortest": info["shortest"], "difficulty": difficulty(info["shortest"], bins),
               "manhattan": man, "detour_ratio": round(info["shortest"] / man, 4),  # >1: the goal requires a detour
               "corridor_score": corridor_score(snap["grid"], snap["trajectory"])}
        eps.append(rec)
        if rec["outcome"] != "success" and len(failures) < max_failures:
            failures.append({**rec, **snap})
    res = {"run": run_dir, "agent": cfg["agent"], "seed": cfg["seed"], "size": cfg["size"], "obs": cfg["obs"],
           "train_density": cfg["density"], "eval_density": ecfg["density"], "eval_eps": eval_eps, "n_maps": n_maps,
           **({"eval_size": ecfg["size"], "eval_max_steps": env.max_steps} if ecfg["size"] != cfg["size"] else {}),
           "difficulty_bins_shortest_path": list(bins), **summarize(eps),
           "by_difficulty": {d: summarize([e for e in eps if e["difficulty"] == d]) for d in ("easy", "medium", "hard")}}
    # Window-mode, same-mode evaluation keeps the historical json keys. Anything else records the mode
    # so later aggregation cannot treat an occlusion run as one of the published window runs.
    if cross or eval_mode != OBS_MODE_WINDOW or train_mode != OBS_MODE_WINDOW:
        res["obs_mode"] = eval_mode
        res["train_obs_mode"] = train_mode
    if cross:
        res["exploratory_cross_mode"] = True
    tag = f"d{ecfg['density']}" + (f"_eps{eval_eps:g}" if eval_eps > 0 else "")
    if cross:
        tag += f"_xmode_{eval_mode}"
    if out_dir != run_dir:
        res["model_run"] = os.path.relpath(run_dir, out_dir)
        source = {"model_run": res["model_run"], "train_size": cfg["size"], "eval_size": ecfg["size"],
                  "eval_max_steps": env.max_steps, "min_start_goal_dist": env.min_sg}
        if cross or eval_mode != OBS_MODE_WINDOW or train_mode != OBS_MODE_WINDOW:
            source.update({"train_obs_mode": train_mode, "eval_obs_mode": eval_mode,
                           "exploratory_cross_mode": cross})
        json.dump(source, open(os.path.join(out_dir, "source.json"), "w"), indent=2)
    json.dump(res, open(os.path.join(out_dir, f"eval_{tag}.json"), "w"), indent=2)
    json.dump(failures, open(os.path.join(out_dir, f"failures_{tag}.json"), "w"))
    with open(os.path.join(out_dir, f"episodes_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0])); w.writeheader(); w.writerows(eps)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_maps", type=int, default=300)
    p.add_argument("--density", type=float, default=None, help="override density (difficulty-by-density / generalization)")
    p.add_argument("--eval_eps", type=float, default=0.0,
                   help="epsilon for epsilon-greedy evaluation (default 0 = greedy, the original protocol)")
    p.add_argument("--bins", type=int, nargs=2, default=[9, 11], help="shortest-path thresholds: easy<=a<medium<=b<hard")
    p.add_argument("--size", type=int, default=None,
                   help="evaluate on N x N maps (zero-shot size transfer; default = training size from config.json)")
    p.add_argument("--out", default=None,
                   help="output root (default for --size runs: <results>/runs/N<size>; also usable without --size, e.g. results/runs_trained_eval/N15); files go to <out>/<config>/seed<s>/")
    p.add_argument("--obs_mode", choices=["window", "los"], default=None,
                   help="observation mode for this evaluation (default: the checkpoint's mode). "
                        "A value other than the checkpoint mode is exploratory cross-mode evaluation "
                        "and is written outside the checkpoint directory and outside results/runs.")
    a = p.parse_args()
    torch.set_num_threads(1)
    dirs = [a.run] if os.path.exists(os.path.join(a.run, "config.json")) else sorted(glob.glob(os.path.join(a.run, "seed*")))
    for d in dirs:
        cfg0 = json.load(open(os.path.join(d, "config.json")))
        train_mode = normalize_obs_mode(cfg0.get("obs_mode"))
        eval_mode = train_mode if a.obs_mode is None else normalize_obs_mode(a.obs_mode)
        cross = eval_mode != train_mode
        od = None
        size_transfer = a.size is not None and a.size != cfg0["size"]
        if size_transfer or a.out or cross:
            if cross and not a.out:
                root = os.path.join(RESULTS_DIR, "exploratory_cross_mode", f"{train_mode}_to_{eval_mode}")
                if size_transfer:
                    root = os.path.join(root, f"N{a.size}")
            else:
                root = a.out or os.path.join(RESULTS_DIR, "runs", f"N{a.size}")
            od = os.path.join(root, os.path.basename(os.path.dirname(os.path.abspath(d))), os.path.basename(os.path.abspath(d)))
        r = evaluate(d, a.n_maps, a.density, tuple(a.bins), eval_eps=a.eval_eps, size=a.size, out_dir=od,
                     obs_mode=a.obs_mode)
        print(f"{d} eps={a.eval_eps:g}: success={r['success_rate']:.3f} collision={r['collision_rate']:.3f} timeout={r['timeout_rate']:.3f} "
              + " ".join(f"{k}={v.get('success_rate', float('nan')):.2f}(n={v['n']})" for k, v in r["by_difficulty"].items()))
