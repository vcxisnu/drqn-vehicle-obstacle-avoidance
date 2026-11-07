import json, os
from envs.grid_env import OBS_MODE_WINDOW, GridObstacleEnv, normalize_obs_mode
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
    # Absent obs_mode is window mode: old configs and the default constructor agree.
    return GridObstacleEnv(size=cfg["size"], obs_size=cfg["obs"], density=cfg["density"],
                           max_steps=cfg.get("max_steps"), min_start_goal_dist=cfg.get("min_dist"),
                           gamma=cfg.get("gamma", DEFAULT_GAMMA),
                           obs_mode=normalize_obs_mode(cfg.get("obs_mode")))


def run_dirname(cfg):
    """Directory name for one configuration. Non-window modes get an explicit suffix."""
    mode = normalize_obs_mode(cfg.get("obs_mode"))
    name = f"{cfg['agent']}_s{cfg['size']}_k{cfg['obs']}_d{cfg['density']}"
    if mode != OBS_MODE_WINDOW:
        name += f"_{mode}"
    return name


def default_run_dir(cfg):
    """Window runs stay in results/runs. Occlusion runs go to results/runs_los."""
    mode = normalize_obs_mode(cfg.get("obs_mode"))
    root = "runs" if mode == OBS_MODE_WINDOW else f"runs_{mode}"
    return os.path.join(RESULTS_DIR, root, run_dirname(cfg), f"seed{cfg['seed']}")


def _under(path, parent):
    ap, parent = os.path.abspath(path), os.path.abspath(parent)
    return ap == parent or ap.startswith(parent + os.sep)


def assert_separated_from_main_runs(path):
    """Occlusion training, occlusion eval, and cross-mode eval must not land in results/runs."""
    runs = os.path.join(RESULTS_DIR, "runs")
    if _under(path, runs):
        raise RuntimeError(
            f"refusing to write {path} under {os.path.abspath(runs)}. "
            "Window-mode thesis runs live there. Use results/runs_los, "
            "results/occlusion_smoke, or another separate directory."
        )


def refuse_protected_thesis_output(path, protected_name):
    """Do not overwrite summary.csv / plots / failures / detour with non-window aggregation."""
    protected = os.path.join(RESULTS_DIR, protected_name)
    if _under(path, protected):
        raise SystemExit(
            f"refusing to write non-window or mixed-mode results onto {os.path.abspath(protected)}. "
            "Published thesis numbers are obs_mode=window, not line-of-sight occlusion. Pass --out."
        )


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
    """Load config + weights. Deep checkpoints must agree with config.json on obs_mode.

    A legacy model.pt (raw state_dict, no metadata) is a window-mode checkpoint.
    It loads for window configs, including configs that omit obs_mode. It does not
    load for a config that says los: that would mix an old policy with a new observation.
    """
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    mode = normalize_obs_mode(cfg.get("obs_mode"))
    env = make_env(dict(cfg, obs_mode=mode))
    ag = make_agent(cfg, env.obs_dim)
    ag.load(model_file(run_dir, cfg["agent"]))
    if cfg["agent"] in ("dqn", "drqn"):
        meta = getattr(ag, "checkpoint_metadata", None)
        if meta is None:
            if mode != OBS_MODE_WINDOW:
                raise RuntimeError(
                    f"{run_dir}: legacy checkpoint has no obs_mode metadata, but config.json "
                    f"requests obs_mode={mode!r}. Refusing to mix a window-era checkpoint with "
                    "a non-window observation."
                )
        else:
            ckpt_mode = normalize_obs_mode(meta.get("obs_mode"))
            if ckpt_mode != mode:
                raise RuntimeError(
                    f"{run_dir}: checkpoint metadata obs_mode={ckpt_mode!r} does not match "
                    f"config.json obs_mode={mode!r}."
                )
            if "obs_dim" in meta and int(meta["obs_dim"]) != int(env.obs_dim):
                raise RuntimeError(
                    f"{run_dir}: checkpoint obs_dim={meta['obs_dim']} != environment obs_dim={env.obs_dim}."
                )
    return cfg, ag
