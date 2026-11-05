"""Diagnostic (NOT part of the two experiments, does not change training): why is the success rate logged during
the last training episodes (eps = 0.01) much higher for DQN than the held-out success of the FROZEN final model?

Starting from a trained checkpoint, play N episodes on fresh maps (seeds 20,000,000+i: neither training stream nor
test set) under three conditions:
  frozen_greedy   : weights frozen, eps = 0
  frozen_eps      : weights frozen, eps = 0.01
  online_eps      : eps = 0.01 AND the usual online updates continue (same agent code/hyper-parameters as train.py;
                    replay buffer starts empty, updates start after learn_start transitions) -> non-stationary policy
If online_eps >> frozen_eps, the training-log success is explained by within-episode policy non-stationarity
(weights change every 4 steps, which also breaks deterministic loops), not by the 1% random actions.
Usage: python3 diag_train_vs_eval.py --run RUN_SEED_DIR [--episodes 1500]
"""
import argparse, json, os
import numpy as np, torch
from common import load_run, make_env

BASE = 20_000_000


def play(cfg, agent, n, eps, learn, rng, warm=0):
    env = make_env(cfg); succ, step = [], 0
    for i in range(n + warm):
        o, _ = env.reset(seed=BASE + i); agent.reset_state(); done = False
        while not done:
            a = agent.act(o, eps, rng)
            o2, r, te, tr, info = env.step(a)
            if learn:
                agent.observe(o, a, r, o2, te, tr); step += 1; agent.maybe_update(step)
            o, done = o2, te or tr
        if learn:
            agent.end_episode()
        if i >= warm:
            succ.append(info["outcome"] == "success")
    return float(np.mean(succ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--episodes", type=int, default=1500)
    a = p.parse_args(); torch.set_num_threads(1)
    res = {}
    for name, eps, learn in (("frozen_greedy", 0.0, False), ("frozen_eps0.01", 0.01, False), ("online_eps0.01", 0.01, True)):
        torch.manual_seed(0); rng = np.random.default_rng(0)
        cfg, ag = load_run(a.run)
        # online: first 100 episodes only fill the buffer / start learning, report the following N episodes
        res[name] = play(cfg, ag, a.episodes, eps, learn, rng, warm=100 if learn else 0)
    print(a.run, res)
    json.dump(res, open(os.path.join(a.run, "diag_train_vs_eval.json"), "w"), indent=2)
