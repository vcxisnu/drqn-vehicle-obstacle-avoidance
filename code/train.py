import argparse, csv, json, os, random, time
import numpy as np
import torch
from common import (AGENTS, DEFAULT_GAMMA, assert_separated_from_main_runs, default_run_dir,
                    make_env, make_agent, model_file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=AGENTS, default="drqn")
    p.add_argument("--size", "--N", dest="size", type=int, default=10, help="map side length N (default 10; max_steps defaults to 8N)")
    p.add_argument("--obs", type=int, default=5, choices=[3, 5, 7])
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="one run per seed (sequential)")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--min_dist", type=int, default=None, help="min Manhattan start-goal distance (default max(7, k) = 7 for k<=7, so test maps are identical for all k)")
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.01)
    p.add_argument("--eps_decay_frac", type=float, default=0.6, help="fraction of episodes for linear decay")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--burn_in", type=int, default=8)
    p.add_argument("--train_every", type=int, default=4)
    p.add_argument("--target_update", type=int, default=1000)
    p.add_argument("--learn_start", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--obs_mode", choices=["window", "los"], default="window",
                   help="window = historical local occupancy (default, published results); "
                        "los = line-of-sight occlusion. los never writes under results/runs "
                        "(default results/runs_los/<config>_los/seedN)")
    p.add_argument("--out", default=None, help="run directory (default results/runs/<agent>_s<size>_k<obs>_d<density>; "
                                               "los default is results/runs_los/..._los)")
    args = vars(p.parse_args())
    args["gamma"] = DEFAULT_GAMMA  # recorded in config.json; agent discount AND shaping gamma (not a CLI option)
    args["reward_version"] = "pbrs_v2"  # F = 0.1*(gamma*Phi(s') - Phi(s)), Phi = -d_BFS, Phi(terminal) = 0
    for seed in args["seeds"]:
        cfg = dict(args, seed=seed)
        cfg.pop("seeds")
        if args["out"]:
            cfg["out"] = os.path.join(args["out"], f"seed{seed}")
        run(cfg)


def run(cfg):
    mode = cfg.get("obs_mode") or "window"
    cfg["obs_mode"] = mode
    out = cfg["out"] or default_run_dir(cfg)
    if mode != "window":
        assert_separated_from_main_runs(out)
    cfg["out"] = out
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg["seed"])
    env = make_env(cfg)
    env.reset(seed=cfg["seed"])  # seeds env RNG; subsequent resets continue the training stream
    agent = make_agent(cfg, env.obs_dim)
    if hasattr(agent, "n_params"):
        print(f"[{cfg['agent']}] k={cfg['obs']} obs_mode={cfg['obs_mode']} trainable params: {agent.n_params():,}", flush=True)
    f = open(os.path.join(out, "train_log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "reward", "steps", "success", "collision", "timeout", "epsilon", "loss", "success_ma100"])
    hist, step, t0 = [], 0, time.time()
    decay_eps = max(1, int(cfg["episodes"] * cfg["eps_decay_frac"]))
    for ep in range(cfg["episodes"]):
        eps = max(cfg["eps_end"], cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * ep / decay_eps)
        obs, info = env.reset()
        agent.reset_state()
        total, losses, done = 0.0, [], False
        while not done:
            a = agent.act(obs, eps, rng)
            obs2, r, term, trunc, info = env.step(a)
            agent.observe(obs, a, r, obs2, term, trunc)  # truncation (timeout) is NOT terminal: bootstrap
            step += 1
            l = agent.maybe_update(step)
            if l is not None:
                losses.append(l)
            obs, total, done = obs2, total + r, term or trunc
        agent.end_episode()
        oc = info["outcome"]
        hist.append(oc == "success")
        ma = float(np.mean(hist[-100:]))
        w.writerow([ep, round(total, 3), info["t"], int(oc == "success"), int(oc == "collision"),
                    int(oc == "timeout"), round(eps, 4), round(np.mean(losses), 5) if losses else "", round(ma, 3)])
        if (ep + 1) % max(1, cfg["episodes"] // 20) == 0:
            f.flush()
            print(f"[{cfg['agent']}] ep {ep+1}/{cfg['episodes']} obs_mode={cfg['obs_mode']} eps={eps:.2f} succ@100={ma:.2f} "
                  f"steps={step} {time.time()-t0:.0f}s", flush=True)
    f.close()
    path = model_file(out, cfg["agent"])
    # Deep checkpoints carry obs_mode next to the weights. Tabular/random keep their own formats.
    if cfg["agent"] in ("dqn", "drqn"):
        agent.save(path, metadata={"obs_mode": cfg["obs_mode"], "obs_dim": int(env.obs_dim),
                                   "size": cfg["size"], "obs": cfg["obs"], "agent": cfg["agent"],
                                   "reward_version": cfg.get("reward_version")})
    else:
        agent.save(path)
    print("saved to", out)


if __name__ == "__main__":
    main()
