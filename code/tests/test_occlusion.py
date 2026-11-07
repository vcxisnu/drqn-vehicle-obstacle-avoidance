"""Optional line-of-sight occlusion. The default window observation stays historical."""
import json
import os

import numpy as np
import pytest

from envs.grid_env import (
    OCC_FREE, OCC_OBSTACLE, OCC_UNKNOWN, TEST_SEED_BASE, GridObstacleEnv,
    configure_occlusion_demo, los_visible, normalize_obs_mode,
)

G = 0.99


def _blocked_by(cells, n=10):
    blocked_cells = set(cells)

    def blocked(r, c):
        if not (0 <= r < n and 0 <= c < n):
            return True
        return (r, c) in blocked_cells

    return blocked


def _pair(mode, k=5):
    env = GridObstacleEnv(10, k, 0.0, gamma=G, obs_mode=mode)
    env.reset(seed=0)
    configure_occlusion_demo(env)
    return env


def _at(env, rr, cc):
    obs = env._obs()
    k = env.k
    occ = obs[:k * k].reshape(k, k)
    r, c = env.pos
    h = k // 2
    return float(occ[rr - r + h, cc - c + h])


def _goal_at(env, rr, cc):
    obs = env._obs()
    k = env.k
    goal = obs[k * k:2 * k * k].reshape(k, k)
    r, c = env.pos
    h = k // 2
    return float(goal[rr - r + h, cc - c + h])


def test_missing_mode_is_window_and_bad_mode_raises():
    assert normalize_obs_mode(None) == "window"
    env = GridObstacleEnv(10, 5, gamma=G)
    assert env.obs_mode == "window" and env.obs_dim == 2 * 25 + 2
    with pytest.raises(ValueError):
        GridObstacleEnv(10, 5, gamma=G, obs_mode="fog")


def test_window_occupancy_stays_binary_on_held_out_maps():
    env = GridObstacleEnv(10, 5, 0.2, gamma=G)
    for i in range(20):
        obs, _ = env.reset(seed=TEST_SEED_BASE + i)
        assert set(np.unique(obs[:25]).tolist()).issubset({OCC_FREE, OCC_OBSTACLE})
        assert obs.shape == (52,) and obs.dtype == np.float32


def test_corner_policy_blocks_diagonal_cracks_and_keeps_the_first_wall():
    crack = _blocked_by([(0, 1), (1, 2)])
    assert los_visible(crack, 2, 0, 0, 2) is False
    assert los_visible(crack, 2, 0, 0, 1) is True  # first obstacle on that side
    assert los_visible(crack, 2, 0, 1, 2) is True
    assert los_visible(_blocked_by([]), 2, 0, 0, 2) is True
    # Conservative: one blocked neighbor is enough. Grazing a single corner is not allowed.
    assert los_visible(_blocked_by([(0, 1)]), 2, 0, 0, 2) is False
    assert los_visible(_blocked_by([(1, 2)]), 2, 0, 0, 2) is False
    wall = _blocked_by([(4, 5)])
    assert los_visible(wall, 4, 3, 4, 5) is True
    assert los_visible(wall, 4, 3, 4, 6) is False
    assert los_visible(wall, 4, 3, 4, 7) is False
    cells = [(r, c) for r in range(-1, 11) for c in range(-1, 11)]
    once = [los_visible(crack, 2, 0, r, c) for r, c in cells]
    assert once == [los_visible(crack, 2, 0, r, c) for r, c in cells]


def test_demo_wall_hides_free_and_obstacle_cells_but_not_the_first_wall():
    los, win = _pair("los"), _pair("window")
    # Hidden free goal and hidden obstacle are unknown, regardless of true occupancy.
    assert los.grid[4, 4] == 0 and _at(los, 4, 4) == OCC_UNKNOWN
    assert los.grid[6, 4] == 1 and _at(los, 6, 4) == OCC_UNKNOWN
    assert _goal_at(los, 4, 4) == 0.0
    # The wall itself is the first obstacle on the sightline.
    assert _at(los, 4, 3) == OCC_OBSTACLE and _at(los, 6, 3) == OCC_OBSTACLE
    # Diagonal crack: both touching obstacles visible, the cell past the corner unknown.
    assert los.grid[3, 0] == 1 and _at(los, 3, 0) == OCC_UNKNOWN
    assert _at(los, 4, 0) == OCC_OBSTACLE and _at(los, 3, 1) == OCC_OBSTACLE
    # Historical window still reports the truth, including the goal channel.
    assert _at(win, 4, 4) == OCC_FREE and _at(win, 6, 4) == OCC_OBSTACLE
    assert _goal_at(win, 4, 4) == 1.0
    np.testing.assert_allclose(los._obs()[-2:], win._obs()[-2:])
    np.testing.assert_allclose(los._obs()[-2:], [(4 - 5) / 9, (4 - 2) / 9])
    assert los._obs().shape == win._obs().shape == (52,)


def test_reveal_after_stepping_around_and_no_map_memory():
    los = _pair("los")
    win = _pair("window")
    hidden = los._obs().copy()
    assert not any(hasattr(los, name) for name in ("known", "known_map", "memory", "seen"))
    for env in (los, win):
        o1, r1, t1, u1, i1 = env.step(3)
        o2, r2, t2, u2, i2 = env.step(3)
        assert env.pos == (5, 4) and not t1 and not u1 and not t2 and not u2
    # Same actions, same reward and termination. BFS shaping did not switch to the occluded view.
    # Re-run in lockstep for the numeric reward (the loops above already consumed both envs).
    los, win = _pair("los"), _pair("window")
    for _ in range(2):
        _, rl, tl, ul, il = los.step(3)
        _, rw, tw, uw, iw = win.step(3)
        assert rl == rw and tl == tw and ul == uw and il["outcome"] == iw["outcome"]
    assert _at(los, 4, 4) == OCC_FREE and _goal_at(los, 4, 4) == 1.0
    assert _at(los, 6, 4) == OCC_OBSTACLE
    # Putting the robot back restores the original observation: nothing was stored.
    los.pos = (5, 2)
    assert np.array_equal(los._obs(), hidden)
    assert np.array_equal(los._obs(), los._obs())


def test_maps_that_differ_only_behind_occluders_match():
    a, b = _pair("los"), _pair("los")
    b.grid = a.grid.copy()
    b.grid[6, 4] = 0          # hidden obstacle -> free
    b.grid[3, 0] = 0          # hidden crack cell -> free
    b.grid[7, 4] = 1          # another hidden cell (past the wall on the diagonal)
    assert np.array_equal(a._obs(), b._obs())
    b.grid[5, 3] = 1          # the visible gap cell
    assert not np.array_equal(a._obs(), b._obs())


def test_open_map_matches_window_including_borders():
    for k in (3, 5, 7):
        for pos in ((0, 0), (0, 5), (1, 1), (5, 5), (9, 9), (9, 2)):
            w = GridObstacleEnv(10, k, 0.0, gamma=G, obs_mode="window")
            los = GridObstacleEnv(10, k, 0.0, gamma=G, obs_mode="los")
            for env in (w, los):
                env.reset(seed=0)
                env.grid[:] = 0
                env.pos = pos
                env.goal = (0, 9) if pos == (5, 5) else (5, 5)
            np.testing.assert_array_equal(w._obs(), los._obs())


def test_boundary_cells_stay_obstacle_even_when_in_map_cells_are_hidden():
    w = GridObstacleEnv(10, 7, 0.2, gamma=G, obs_mode="window")
    los = GridObstacleEnv(10, 7, 0.2, gamma=G, obs_mode="los")
    w.reset(seed=TEST_SEED_BASE + 3)
    los.grid = w.grid.copy()
    los.goal = w.goal
    los.pos = w.pos
    h = 3
    for pos in ((0, 0), (0, 9), (9, 0), (9, 9), (0, 4), (4, 0)):
        w.pos = los.pos = pos
        ow = w._obs()[:49].reshape(7, 7)
        ol = los._obs()[:49].reshape(7, 7)
        r, c = pos
        for i in range(7):
            for j in range(7):
                rr, cc = r + i - h, c + j - h
                if not (0 <= rr < 10 and 0 <= cc < 10):
                    assert ow[i, j] == ol[i, j] == OCC_OBSTACLE


def test_window_limit_ignores_cells_outside_k_and_hidden_edits():
    small = _pair("los", k=5)
    large = _pair("los", k=7)
    narrow = _pair("los", k=3)
    before = small._obs().copy()
    # (5, 5) is chebyshev distance 3 from (5, 2): outside k=5, inside k=7, and visible.
    assert max(abs(5 - 5), abs(5 - 2)) == 3
    small.grid = small.grid.copy()
    small.grid[5, 5] = 1
    np.testing.assert_array_equal(small._obs(), before)
    assert _at(large, 5, 5) == OCC_FREE
    large.grid = large.grid.copy()
    large.grid[5, 5] = 1
    assert _at(large, 5, 5) == OCC_OBSTACLE
    # k=3 is the centre of the k=5 observation (same pose, same map).
    k5 = _pair("los", k=5)._obs()[:25].reshape(5, 5)
    k3 = narrow._obs()[:9].reshape(3, 3)
    np.testing.assert_array_equal(k5[1:4, 1:4], k3)
    # Editing a hidden in-window cell does not change k=3, which does not cover it.
    before3 = narrow._obs().copy()
    narrow.grid = narrow.grid.copy()
    narrow.grid[6, 4] = 0
    np.testing.assert_array_equal(narrow._obs(), before3)


def test_los_reward_matches_window_on_random_maps():
    rng = np.random.default_rng(0)
    differed = False
    for seed in range(12):
        w = GridObstacleEnv(10, 5, 0.2, gamma=G, obs_mode="window")
        los = GridObstacleEnv(10, 5, 0.2, gamma=G, obs_mode="los")
        w.reset(seed=TEST_SEED_BASE + seed)
        los.reset(seed=TEST_SEED_BASE + seed)
        assert np.array_equal(w.grid, los.grid) and w.start == los.start
        for _ in range(25):
            action = int(rng.integers(4))
            ow, rw, tw, uw, iw = w.step(action)
            ol, rl, tl, ul, il = los.step(action)
            assert rw == rl and tw == tl and uw == ul and iw["outcome"] == il["outcome"]
            np.testing.assert_allclose(ow[-2:], ol[-2:])
            assert ol.shape == (52,) and np.all(ol >= -1.0) and np.all(ol <= 1.0)
            if not np.array_equal(ow, ol):
                differed = True
            if tw or uw:
                break
    assert differed  # occlusion is not a no-op on these maps


def test_occupied_cell_does_not_hide_its_own_neighbour():
    """A collision leaves the robot on an obstacle. That cell does not block outgoing rays."""
    los = _pair("los")
    los.pos = (4, 3)
    assert _at(los, 4, 3) == OCC_OBSTACLE
    assert _at(los, 4, 4) == OCC_FREE and _goal_at(los, 4, 4) == 1.0


def test_output_paths_do_not_fall_into_results_runs():
    from common import assert_separated_from_main_runs, default_run_dir, run_dirname
    window = dict(agent="drqn", size=10, obs=5, density=0.2, seed=0)
    los = dict(window, obs_mode="los")
    assert run_dirname(window) == "drqn_s10_k5_d0.2"
    assert run_dirname(los) == "drqn_s10_k5_d0.2_los"
    assert default_run_dir(window).endswith(os.path.join("results", "runs", "drqn_s10_k5_d0.2", "seed0"))
    assert os.path.join("runs_los", "drqn_s10_k5_d0.2_los", "seed0") in default_run_dir(los)
    with pytest.raises(RuntimeError):
        assert_separated_from_main_runs(default_run_dir(window))
    assert_separated_from_main_runs(default_run_dir(los))


def _eval_json(root, config, seed, **extra):
    directory = root / config / f"seed{seed}"
    directory.mkdir(parents=True)
    body = {
        "agent": "dqn", "obs": 5, "train_density": 0.2, "eval_density": 0.2, "eval_eps": 0.0,
        "seed": seed, "size": 10, "n_maps": 1,
        "success_rate": 0.5, "collision_rate": 0.0, "timeout_rate": 0.5, "avg_steps_success": 12.0,
        "by_difficulty": {name: {"n": 1, "success_rate": 0.5} for name in ("easy", "medium", "hard")},
    }
    body.update(extra)
    json.dump(body, open(directory / "eval_d0.2.json", "w"))


def test_aggregate_keeps_window_schema_and_skips_cross_mode(tmp_path):
    from aggregate import collect
    _eval_json(tmp_path / "only", "dqn_s10_k5_d0.2", 0)
    rows = collect(tmp_path / "only")
    assert len(rows) == 1 and "obs_mode" not in rows[0]
    _eval_json(tmp_path / "mix", "dqn_s10_k5_d0.2", 0, success_rate=0.5)
    _eval_json(tmp_path / "mix", "dqn_s10_k5_d0.2_los", 0, obs_mode="los", success_rate=0.2)
    mixed = collect(tmp_path / "mix")
    assert {r["obs_mode"] for r in mixed} == {"window", "los"}
    assert {r["success_rate"] for r in mixed} == {0.5, 0.2}
    _eval_json(tmp_path / "x", "dqn_s10_k5_d0.2", 0, exploratory_cross_mode=True, obs_mode="los")
    assert collect(tmp_path / "x") == []


def test_aggregate_refuses_mixed_folder(tmp_path, monkeypatch):
    from aggregate import main
    _eval_json(tmp_path / "mix", "dqn_s10_k5_d0.2", 0)
    _eval_json(tmp_path / "mix", "drqn_s10_k5_d0.2_los", 0, obs_mode="los", agent="drqn")
    monkeypatch.setattr("sys.argv", ["aggregate.py", "--runs", str(tmp_path / "mix"),
                                     "--out", str(tmp_path / "summary.csv")])
    with pytest.raises(SystemExit, match="mixed obs_mode"):
        main()
    assert not (tmp_path / "summary.csv").exists()


def test_cross_mode_eval_and_legacy_checkpoint(tmp_path):
    pytest.importorskip("torch")
    import torch
    from agents.deep import DQNAgent
    from common import RESULTS_DIR, load_run
    from evaluate import evaluate
    from train import run

    published = os.path.join(RESULTS_DIR, "runs", "dqn_s10_k5_d0.2", "seed0")
    if os.path.exists(os.path.join(published, "model.pt")):
        cfg, agent = load_run(published)
        assert cfg.get("obs_mode", "window") == "window"
        assert agent.checkpoint_metadata is None
        assert agent.in_dim == 52

    rd = tmp_path / "random_s10_k5_d0.2" / "seed0"
    run(dict(agent="random", size=10, obs=5, density=0.2, episodes=1, max_steps=None, min_dist=None,
             eps_start=1.0, eps_end=0.01, eps_decay_frac=0.6, gamma=0.99, reward_version="pbrs_v2",
             seed=0, obs_mode="window", out=str(rd)))
    assert json.load(open(rd / "config.json"))["obs_mode"] == "window"
    before = sorted(os.listdir(rd))
    with pytest.raises(RuntimeError, match="cross-mode"):
        evaluate(str(rd), n_maps=2, obs_mode="los")
    assert sorted(os.listdir(rd)) == before
    out = tmp_path / "exploratory" / "seed0"
    result = evaluate(str(rd), n_maps=2, obs_mode="los", out_dir=str(out))
    assert result["exploratory_cross_mode"] is True
    assert result["train_obs_mode"] == "window" and result["obs_mode"] == "los"
    assert sorted(os.listdir(rd)) == before
    assert any("xmode_los" in name for name in os.listdir(out))
    same = evaluate(str(rd), n_maps=1)
    assert "obs_mode" not in same and "exploratory_cross_mode" not in same

    forbidden = os.path.join(RESULTS_DIR, "runs", "_los_guard_should_not_exist", "seed0")
    with pytest.raises(RuntimeError, match="refusing"):
        run(dict(agent="random", size=10, obs=5, density=0.2, episodes=1, max_steps=None, min_dist=None,
                 eps_start=1.0, eps_end=0.01, eps_decay_frac=0.6, gamma=0.99, seed=0,
                 obs_mode="los", out=forbidden))
    assert not os.path.exists(forbidden)

    los_dir = tmp_path / "los_seed"
    los_dir.mkdir()
    net = DQNAgent(52, 5)
    net.save(str(los_dir / "model.pt"), metadata={"obs_mode": "los", "obs_dim": 52})
    json.dump({"agent": "dqn", "size": 10, "obs": 5, "density": 0.2, "seed": 0, "obs_mode": "los"},
              open(los_dir / "config.json", "w"))
    _, loaded = load_run(str(los_dir))
    assert loaded.checkpoint_metadata["obs_mode"] == "los"
    json.dump({"agent": "dqn", "size": 10, "obs": 5, "density": 0.2, "seed": 0, "obs_mode": "window"},
              open(los_dir / "config.json", "w"))
    with pytest.raises(RuntimeError, match="does not match"):
        load_run(str(los_dir))
    torch.save(net.net.state_dict(), los_dir / "model.pt")
    json.dump({"agent": "dqn", "size": 10, "obs": 5, "density": 0.2, "seed": 0, "obs_mode": "los"},
              open(los_dir / "config.json", "w"))
    with pytest.raises(RuntimeError, match="legacy"):
        load_run(str(los_dir))
