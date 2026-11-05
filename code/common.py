import json, os
from envs.grid_env import GridObstacleEnv
# agents are imported lazily inside make_agent so that the analysis scripts (aggregate/plot/failure/detour) run without torch

AGENTS = ["tabular", "dqn", "drqn", "random"]
DEFAULT_GAMMA = 0.99  # agent discount; the env's potential-based shaping uses the SAME value

# repository layout: <root>/code/*.py, <root>/results/...  (no absolute paths anywhere; override with --runs/--out)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))


class RandomAgent:
    """Uniform random policy (reference baseline; no training)."""
    recurrent = False
    def __init__(self, *a, **k): pass
    def reset_state(self): pass
    def act(self, obs, eps, rng): return int(rng.integers(4))
    def observe(self, *a, **k): pass
    def end_episode(self): pass
    def maybe_update(self, step): return None
    def save(self, path): open(path, "w").close()
    def load(self, path): pass


def make_env(cfg):
    return GridObstacleEnv(size=cfg["size"], obs_size=cfg["obs"], density=cfg["density"],
                           max_steps=cfg.get("max_steps"), min_start_goal_dist=cfg.get("min_dist"),
                           gamma=cfg.get("gamma", DEFAULT_GAMMA))


def make_agent(cfg, obs_dim):
    a, k = cfg["agent"], cfg["obs"]
    hp = dict(lr=cfg.get("lr", 5e-4), gamma=cfg.get("gamma", DEFAULT_GAMMA), batch_size=cfg.get("batch_size", 32),
              target_update=cfg.get("target_update", 1000), learn_start=cfg.get("learn_start", 1000),
              train_every=cfg.get("train_every", 4), hidden=cfg.get("hidden", 128))
    if a == "random":
        return RandomAgent()
    if a == "tabular":
        from agents.tabular import TabularQAgent
        return TabularQAgent(obs_dim, lr=cfg.get("tab_lr", 0.1), gamma=hp["gamma"])
    from agents.deep import DQNAgent, DRQNAgent
    if a == "dqn":
        return DQNAgent(obs_dim, k, **hp)
    return DRQNAgent(obs_dim, k, core="lstm", seq_len=cfg.get("seq_len", 16), burn_in=cfg.get("burn_in", 8), **hp)


def model_file(run_dir, agent):
    return os.path.join(run_dir, "model.pkl" if agent == "tabular" else "model.pt")


def load_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    env = make_env(cfg)
    ag = make_agent(cfg, env.obs_dim)
    ag.load(model_file(run_dir, cfg["agent"]))
    return cfg, ag
