"""Agent-side invariants (require torch): DRQN sub-sequence sampling, parameter matching,
and a short end-to-end train/save/load round trip."""
import json
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from agents.deep import DQNAgent, DRQNAgent
from common import make_agent, make_env, load_run, model_file
from envs.grid_env import TEST_SEED_BASE


def test_drqn_sample_batch_alignment():
    ag = DRQNAgent(obs_dim=4, k=3, batch_size=8, seq_len=4, burn_in=2, learn_start=0)
    # one episode of length 6 with obs[t] = t (broadcast), terminated at the end
    T = 6
    for t in range(T):
        ag.observe(np.full(4, t, np.float32), t % 4, float(t), np.full(4, t + 1, np.float32), t == T - 1, False)
    ag.end_episode()
    obs, act, rew, done, mask = ag.sample_batch()
    for i in range(8):
        valid = np.flatnonzero(mask[i])
        assert 1 <= len(valid) <= 4
        s = valid[0]                                         # first trained position in the window
        assert s <= 2                                        # at most burn_in steps before the first trained step
        for j in range(len(valid) + s):                     # burn-in + trained steps are real, aligned transitions
            t = int(obs[i, j, 0])
            assert act[i, j] == t % 4 and rew[i, j] == float(t) and obs[i, j + 1, 0] == t + 1
            assert done[i, j] == (1.0 if t == T - 1 else 0.0)
        assert np.all(done[i, len(valid) + s:] == 1.0) and np.all(mask[i, len(valid) + s:] == 0.0)  # padding


def test_parameter_matching():
    obs_dim = 2 * 25 + 2
    dqn, drqn = DQNAgent(obs_dim, 5), DRQNAgent(obs_dim, 5)
    assert (dqn.n_params(), drqn.n_params()) == (155_901, 155_908)


@pytest.mark.parametrize("agent", ["dqn", "drqn"])
def test_train_eval_round_trip(tmp_path, agent):
    cfg = dict(agent=agent, size=10, obs=3, density=0.2, seed=0, learn_start=10)
    env = make_env(cfg); env.reset(seed=0); ag = make_agent(cfg, env.obs_dim); rng = np.random.default_rng(0)
    step = 0
    for _ in range(3):
        o, _ = env.reset(); ag.reset_state(); done = False
        while not done:
            a = ag.act(o, 0.5, rng); o2, r, te, tr, _ = env.step(a); ag.observe(o, a, r, o2, te, tr); step += 1
            ag.maybe_update(step); o, done = o2, te or tr
        ag.end_episode()
    run = tmp_path / "seed0"; run.mkdir(); json.dump(cfg, open(run / "config.json", "w")); ag.save(model_file(str(run), agent))
    _, ag2 = load_run(str(run))
    o, _ = env.reset(seed=TEST_SEED_BASE); ag.reset_state(); ag2.reset_state()
    for _ in range(5):
        a1, a2 = ag.act(o, 0.0, rng), ag2.act(o, 0.0, rng)
        assert a1 == a2
        o, *_ = env.step(a1)
