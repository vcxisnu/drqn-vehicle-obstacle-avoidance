"""N x N maps for the zero-shot size-transfer evaluation (N = 15 / 20): solvable, min start-goal distance 7,
observation shape independent of N, goal offset normalised by (N-1), max_steps = 8N; N = 10 unchanged."""
import numpy as np
import pytest
from envs.grid_env import GridObstacleEnv, TEST_SEED_BASE

G = 0.99


@pytest.mark.parametrize("n", [15, 20])
@pytest.mark.parametrize("k", [3, 5, 7])
def test_large_maps_solvable_min_dist_obs(n, k):
    env = GridObstacleEnv(n, k, 0.2, gamma=G)
    assert env.max_steps == 8 * n and env.min_sg == 7 and env.obs_dim == 2 * k * k + 2
    dens = []
    for i in range(200):
        o, info = env.reset(seed=TEST_SEED_BASE + i)
        s, g = env.start, env.goal
        assert env.grid.shape == (n, n)
        assert abs(s[0] - g[0]) + abs(s[1] - g[1]) >= 7
        assert env.grid[s] == 0 and env.grid[g] == 0 and env.goal_dist[s] >= 7  # BFS path exists
        assert o.shape == (2 * k * k + 2,) and o.dtype == np.float32
        assert np.all(o >= -1) and np.all(o <= 1)
        np.testing.assert_allclose(o[-2:], [(g[0] - s[0]) / (n - 1), (g[1] - s[1]) / (n - 1)], rtol=1e-6)
        dens.append(env.grid.mean())
    assert abs(np.mean(dens) - 0.2) < 0.02  # density 0.2 (slightly below: unsolvable maps are rejected)


@pytest.mark.parametrize("n", [15, 20])
def test_offset_normalisation_extremes_and_window(n):
    env = GridObstacleEnv(n, 5, 0.2, gamma=G); env.reset(seed=TEST_SEED_BASE)
    env.goal = (n - 1, n - 1); env.pos = (0, 0)
    np.testing.assert_allclose(env._obs()[-2:], [1.0, 1.0])
    env.goal = (0, 0); env.pos = (n - 1, n - 1)
    o = env._obs(); np.testing.assert_allclose(o[-2:], [-1.0, -1.0])
    w = o[:25].reshape(5, 5)  # bottom-right corner: rows/cols beyond the border are padded as obstacles
    assert np.all(w[3:, :] == 1) and np.all(w[:, 3:] == 1)
    assert np.array_equal(w[:3, :3], env.grid[n - 3:, n - 3:].astype(np.float32))


@pytest.mark.parametrize("n", [15, 20])
def test_large_test_maps_are_fixed_and_differ_from_10(n):
    a, b, c = GridObstacleEnv(n, 5, 0.2, gamma=G), GridObstacleEnv(n, 3, 0.2, gamma=G), GridObstacleEnv(10, 5, 0.2, gamma=G)
    for i in range(20):
        a.reset(seed=TEST_SEED_BASE + i); b.reset(seed=TEST_SEED_BASE + i); c.reset(seed=TEST_SEED_BASE + i)
        assert np.array_equal(a.grid, b.grid) and a.start == b.start and a.goal == b.goal  # map independent of k
        assert a.grid.shape != c.grid.shape


def test_n10_defaults_unchanged():
    env = GridObstacleEnv(10, 5, 0.2, gamma=G)
    assert env.max_steps == 80 and env.min_sg == 7
    env.reset(seed=TEST_SEED_BASE)
    assert (int(env.grid.sum()), env.start, env.goal) == (20, (9, 6), (2, 4))
    np.testing.assert_allclose(env._obs()[-2:], [(2 - 9) / 9, (4 - 6) / 9])
