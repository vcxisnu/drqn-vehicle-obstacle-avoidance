"""Environment invariants: window padding, potential-based shaping telescoping, fixed test set, no train/test leakage."""
import numpy as np
import pytest
from envs.grid_env import GridObstacleEnv, ACTIONS, TEST_SEED_BASE, RewardConfig

G = 0.99


@pytest.mark.parametrize("k", [3, 5, 7])
def test_window_padding_matches_grid(k):
    env = GridObstacleEnv(10, k, 0.2, gamma=G); env.reset(seed=TEST_SEED_BASE)
    h = k // 2
    for r in range(10):
        for c in range(10):
            env.pos = (r, c)
            o = env._obs(); w = o[:k * k].reshape(k, k); goal_ch = o[k * k:2 * k * k].reshape(k, k)
            for dr in range(-h, h + 1):
                for dc in range(-h, h + 1):
                    rr, cc = r + dr, c + dc
                    inside = 0 <= rr < 10 and 0 <= cc < 10
                    assert w[dr + h, dc + h] == (float(env.grid[rr, cc]) if inside else 1.0)
                    assert goal_ch[dr + h, dc + h] == (1.0 if inside and (rr, cc) == env.goal else 0.0)
            np.testing.assert_allclose(o[-2:], [(env.goal[0] - r) / 9, (env.goal[1] - c) / 9])
    assert env.obs_dim == 2 * k * k + 2 and o.shape == (env.obs_dim,) and o.dtype == np.float32


def base_reward(env, a):
    """Reward without the shaping term, recomputed independently of env.step()."""
    R = env.R; r, c = env.pos; nr, nc = r + ACTIONS[a][0], c + ACTIONS[a][1]
    if not (0 <= nr < env.size and 0 <= nc < env.size):
        return R.step + R.no_move
    if env.grid[nr, nc] == 1:
        return R.step + R.collision
    return R.step + (R.goal if (nr, nc) == env.goal else 0.0)


def test_shaping_telescopes_to_potential_difference():
    """sum_t gamma^t F_t == 0.1 * (gamma^T Phi(s_T) - Phi(s_0)), Phi(terminal) = 0, for random policies."""
    env = GridObstacleEnv(10, 5, 0.2, gamma=G); rng = np.random.default_rng(0)
    for i in range(200):
        env.reset(seed=TEST_SEED_BASE + i)
        phi0 = -float(env.goal_dist[env.start]); tot = base = 0.0; t = 0; done = False
        while not done:
            a = int(rng.integers(4)); b = base_reward(env, a)
            _, rew, term, trunc, _ = env.step(a)
            tot += G ** t * rew; base += G ** t * b; t += 1; done = term or trunc
        phiT = 0.0 if term else -float(env.goal_dist[env.pos])
        assert abs(tot - (base + env.R.shaping_scale * (G ** t * phiT - phi0))) < 1e-9


def test_terminal_and_truncation_semantics():
    env = GridObstacleEnv(10, 5, 0.2, gamma=G); env.reset(seed=TEST_SEED_BASE)
    # walk straight into an obstacle (if adjacent) or bump the border: never both term and trunc
    for a in range(4):
        env.reset(seed=TEST_SEED_BASE); _, r, term, trunc, info = env.step(a)
        assert not (term and trunc)
        assert info["outcome"] in (None, "collision", "success")
    # timeout is truncation, not termination, and carries no extra penalty
    env.reset(seed=TEST_SEED_BASE + 1); env.R = RewardConfig(); env.max_steps = 3
    env.pos = (0, 0); env.grid[:] = 0; env.goal = (9, 9); env.goal_dist = np.full((10, 10), 1, np.int32); env.goal_dist[9, 9] = 0
    for t in range(3):
        _, r, term, trunc, info = env.step(0)  # bump the top border three times
    assert trunc and not term and info["outcome"] == "timeout"
    assert abs(r - (env.R.step + env.R.no_move + env.R.shaping_scale * (G - 1) * (-1))) < 1e-9


def test_fixed_test_set_is_stable():
    """Guards against silently changing the held-out maps when the environment code is edited."""
    env = GridObstacleEnv(10, 5, 0.2, gamma=G)
    sig = []
    for i in range(5):
        env.reset(seed=TEST_SEED_BASE + i)
        sig.append((int(env.grid.sum()), env.start, env.goal, int(env.goal_dist[env.start])))
    assert sig == [(20, (9, 6), (2, 4), 13), (22, (5, 0), (0, 3), 8), (22, (0, 7), (6, 5), 8),
                   (16, (9, 5), (3, 8), 9), (20, (9, 8), (8, 1), 8)]


def test_min_start_goal_distance_and_reachability():
    env = GridObstacleEnv(10, 5, 0.2, gamma=G)
    for i in range(200):
        env.reset(seed=TEST_SEED_BASE + i)
        s, g = env.start, env.goal
        assert abs(s[0] - g[0]) + abs(s[1] - g[1]) >= 7 and env.goal_dist[s] >= 7 and env.grid[s] == 0 and env.grid[g] == 0


def test_training_stream_does_not_contain_test_maps():
    test = set(); env = GridObstacleEnv(10, 5, 0.2, gamma=G)
    for i in range(500):
        env.reset(seed=TEST_SEED_BASE + i); test.add((env.grid.tobytes(), env.start, env.goal))
    e = GridObstacleEnv(10, 5, 0.2, gamma=G); e.reset(seed=0)
    for _ in range(2000):
        e.reset(); assert (e.grid.tobytes(), e.start, e.goal) not in test
