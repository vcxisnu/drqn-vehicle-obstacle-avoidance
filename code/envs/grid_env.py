"""Partially observable grid obstacle-avoidance environment (Gymnasium API).

Observation (flat float32 vector, length 2*k*k + 2):
  [0 : k*k]      obstacle channel of the agent-centred k x k window (1=obstacle, out-of-bounds=1)
  [k*k : 2*k*k]  goal channel (1 at goal cell if inside the window)
  [-2:]          goal offset (row offset, col offset) / (size-1), in [-1, 1]
Use `obs_to_window(obs, k)` to reshape the first part to (2, k, k) for CNNs.
"""
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# action -> (drow, dcol): 0 up, 1 down, 2 left, 3 right
ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


@dataclass
class RewardConfig:
    shaping_scale: float = 0.1  # r_shape = scale * (d_BFS(s) - d_BFS(s')); full map used for REWARD ONLY
    goal: float = 10.0
    collision: float = -10.0   # moving into an obstacle (episode ends)
    timeout: float = 0.0       # timeout is truncation: no extra penalty (bootstrapped in TD target)
    no_move: float = -0.2      # bumping into the map border (position unchanged)
    step: float = -0.01        # per-step cost


def bfs_dist(grid, src):
    """BFS distance from src to every free cell (-1 = unreachable)."""
    n = grid.shape[0]
    d = -np.ones_like(grid, dtype=np.int32)
    d[src] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and grid[rr, cc] == 0 and d[rr, cc] < 0:
                d[rr, cc] = d[r, c] + 1
                q.append((rr, cc))
    return d


class GridObstacleEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=10, obs_size=5, density=0.2, max_steps=None,
                 dist_metric="bfs", min_start_goal_dist=None, rewards: RewardConfig = None):
        assert size >= 5 and obs_size in (3, 5, 7, 9)
        assert obs_size % 2 == 1
        self.size, self.k, self.density = size, obs_size, density
        self.max_steps = max_steps or 8 * size
        self.dist_metric = dist_metric
        # minimum Manhattan start-goal distance. Default 7 = largest window side (>= k for k in 3/5/7), so that the
        # held-out test maps are IDENTICAL for every method and every window size (map generation does not depend on k).
        self.min_sg = min_start_goal_dist if min_start_goal_dist is not None else max(7, obs_size)
        self.R = rewards or RewardConfig()
        self.obs_dim = 2 * obs_size * obs_size + 2
        self.observation_space = spaces.Box(-1.0, 1.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(4)
        self.grid = None

    # ---------- map generation ----------
    def _generate(self):
        n, rng = self.size, self.np_random
        for _ in range(1000):
            grid = (rng.random((n, n)) < self.density).astype(np.int8)
            free = np.argwhere(grid == 0)
            if len(free) < 2:
                continue
            s, g = free[rng.choice(len(free), 2, replace=False)]
            s, g = tuple(int(x) for x in s), tuple(int(x) for x in g)
            if abs(s[0] - g[0]) + abs(s[1] - g[1]) < self.min_sg:
                continue
            dist = bfs_dist(grid, g)
            if dist[s] > 0:  # BFS guarantees a path exists; otherwise regenerate
                return grid, s, g, dist
        raise RuntimeError("could not generate a solvable map; density too high?")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid, self.pos, self.goal, self.goal_dist = self._generate()
        self.start, self.t = self.pos, 0
        self.trajectory = [self.pos]
        return self._obs(), self._info()

    def _d(self, p):
        if self.dist_metric == "bfs":
            return int(self.goal_dist[p])
        return abs(p[0] - self.goal[0]) + abs(p[1] - self.goal[1])

    def _obs(self):
        k, h, n = self.k, self.k // 2, self.size
        padded = np.ones((n + 2 * h, n + 2 * h), np.float32)  # out-of-bounds = obstacle
        padded[h:h + n, h:h + n] = self.grid
        r, c = self.pos
        obst = padded[r:r + k, c:c + k]
        goal = np.zeros((k, k), np.float32)
        gr, gc = self.goal[0] - r + h, self.goal[1] - c + h
        if 0 <= gr < k and 0 <= gc < k:
            goal[gr, gc] = 1.0
        d = np.array([(self.goal[0] - r) / (n - 1), (self.goal[1] - c) / (n - 1)], np.float32)  # drow, dcol
        return np.concatenate([obst.ravel(), goal.ravel(), d]).astype(np.float32)

    def _info(self, outcome=None):
        return {"pos": self.pos, "goal": self.goal, "start": self.start, "t": self.t,
                "outcome": outcome, "shortest": int(self.goal_dist[self.start])}

    def step(self, action):
        self.t += 1
        R = self.R
        r, c = self.pos
        nr, nc = int(r + ACTIONS[action][0]), int(c + ACTIONS[action][1])
        reward, term, outcome = R.step, False, None
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            reward += R.no_move  # bumped the border, stays in place
        elif self.grid[nr, nc] == 1:
            reward += R.collision
            term, outcome = True, "collision"
            self.pos = (nr, nc)
        else:
            old = self._d(self.pos)
            self.pos = (nr, nc)
            reward += R.shaping_scale * (old - self._d(self.pos))  # potential-based: loops sum to 0
            if self.pos == self.goal:
                reward += R.goal
                term, outcome = True, "success"
        self.trajectory.append(self.pos)
        trunc = (not term) and self.t >= self.max_steps
        if trunc:
            reward += R.timeout
            outcome = "timeout"
        obs = self._obs() if outcome != "collision" else self._obs_safe()
        return obs, float(reward), term, trunc, self._info(outcome)

    def _obs_safe(self):
        return self._obs()  # position is the obstacle cell; window still well-defined

    def render(self):
        ch = {0: ".", 1: "#"}
        rows = []
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                row += "A" if (i, j) == self.pos else "G" if (i, j) == self.goal else ch[int(self.grid[i, j])]
            rows.append(row)
        return "\n".join(rows)

    def snapshot(self):
        """Serializable episode record (for failure analysis)."""
        return {"grid": self.grid.tolist(), "start": list(self.start), "goal": list(self.goal),
                "trajectory": [list(p) for p in self.trajectory],
                "shortest": int(self.goal_dist[self.start])}


def obs_to_window(obs, k):
    """(..., 2k^2+2) -> (..., 2, k, k) window and (..., 2) direction."""
    w = obs[..., : 2 * k * k].reshape(*obs.shape[:-1], 2, k, k)
    return w, obs[..., -2:]


TEST_SEED_BASE = 10_000_000  # held-out maps use seeds TEST_SEED_BASE + i; training never uses this range


def corridor_score(grid, traj):
    """Fraction of visited free cells that are 'narrow' (>=2 of 4 neighbours blocked, incl. border)."""
    n = len(grid)
    g = np.array(grid)
    def blocked(r, c):
        return not (0 <= r < n and 0 <= c < n) or g[r, c] == 1
    cells = {tuple(p) for p in traj if 0 <= p[0] < n and 0 <= p[1] < n and g[p[0], p[1]] == 0}
    if not cells:
        return 0.0
    narrow = sum(1 for r, c in cells if sum(blocked(r + dr, c + dc) for dr, dc in ACTIONS) >= 2)
    return narrow / len(cells)
