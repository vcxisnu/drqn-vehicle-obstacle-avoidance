"""Partially observable grid obstacle-avoidance environment (Gymnasium API).

Observation (flat float32 vector, length 2*k*k + 2; dimension does NOT depend on obs_mode):
  [0 : k*k]      occupancy channel of the agent-centred k x k window
  [k*k : 2*k*k]  goal channel (local sensing only)
  [-2:]          known goal offset (row offset, col offset) / (size-1), in [-1, 1]
Use `obs_to_window(obs, k)` to reshape the first part to (2, k, k) for CNNs.

obs_mode="window" (default, and the mode of every published thesis number):
  occupancy is 1 for an obstacle and for out-of-bounds, else 0. The goal channel is 1
  at the goal cell when it lies inside the window. This path is the historical `_obs`.

obs_mode="los" (optional line-of-sight occlusion; not the published results):
  same layout. In-map cells the agent cannot see are OCC_UNKNOWN (-1), whether the
  hidden cell is actually free or blocked. The first obstacle on each sightline stays
  visible (1). Out-of-bounds cells stay 1, the historical boundary encoding.
  The goal channel is 1 only when the goal cell is inside the window AND visible.
  The goal offset is still the true offset (known side information, not local sensing).
  Nothing is accumulated across steps: each observation is a function of the current
  pose and the current map. DRQN memory, if any, is only the agent's LSTM.
  BFS distances are used for the potential reward only and are not written into the
  observation. See docs/los-occlusion.md.

Map side length N (`size`, default 10) is configurable; N = 15 / 20 are used for the zero-shot transfer evaluation
of the 10x10-trained models (no retraining). Everything is a function of N and nothing changes at N = 10:
  * obstacle density 0.2 (i.i.d. per cell), start/goal drawn uniformly from free cells;
  * min Manhattan start-goal distance = max(7, k) = 7, NOT scaled with N (same rule as 10x10);
  * solvable-path guarantee: maps are regenerated until BFS finds a start->goal path;
  * max_steps = 8 * N (80 / 120 / 160 for N = 10 / 15 / 20): linear in N, i.e. the budget per unit of map side is
    the same as in training (8 x the side length >= 4 x the largest possible Manhattan distance 2(N-1));
  * goal offset normalised by (N - 1) so it stays in [-1, 1]  (=/9 at N = 10). NOTE: a model trained at N = 10
    learned the scale "1 cell = 1/9"; at N = 15 / 20 one cell is 1/14 / 1/19 -> input-distribution shift;
  * the k x k window, the out-of-bounds = obstacle padding and the shaping Phi = -d_BFS are N-independent.
"""
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# action -> (drow, dcol): 0 up, 1 down, 2 left, 3 right
ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

# "window" is the historical local occupancy. "los" is optional line-of-sight occlusion.
# Missing metadata (old checkpoints, old configs) means "window".
OBS_MODE_WINDOW = "window"
OBS_MODE_LOS = "los"
OBS_MODES = (OBS_MODE_WINDOW, OBS_MODE_LOS)

# Occupancy-channel scalars. FREE and OBSTACLE are exactly the historical window values.
# UNKNOWN is los-only, in-map only, and inside the existing Box(-1, 1). It is not between
# 0 and 1, so it is not a fractional obstacle. The goal offset already uses [-1, 1].
OCC_FREE = 0.0
OCC_OBSTACLE = 1.0
OCC_UNKNOWN = -1.0

# Fixed map shared by unit tests and docs/figures/los_window_vs_occlusion.png.
# Start (5, 2) looks east. Wall cells (4, 3) and (6, 3) hide the free goal (4, 4)
# and the obstacle (6, 4). Crack obstacles (4, 0) and (3, 1) hide obstacle (3, 0).
# With k=5 the window is entirely inside the map, so every -1 is occlusion, not boundary.
OCCLUSION_DEMO_START = (5, 2)
OCCLUSION_DEMO_GOAL = (4, 4)
OCCLUSION_DEMO_OBSTACLES = ((4, 3), (6, 3), (6, 4), (4, 0), (3, 1), (3, 0))


def normalize_obs_mode(value):
    """Old configs omit the key; that is window mode, not an error."""
    if value is None or value == OBS_MODE_WINDOW:
        return OBS_MODE_WINDOW
    if value == OBS_MODE_LOS:
        return OBS_MODE_LOS
    raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {value!r}")


def los_visible(blocked, r0, c0, r1, c1):
    """True if (r1, c1) lies on a clear center-to-center sightline from (r0, c0).

    `blocked(r, c)` is True for obstacles and for out-of-bounds. The origin never
    blocks. The endpoint is visible even when it is itself blocked: the first
    obstacle on the sightline is observed, and only cells strictly beyond a blocker
    are hidden.

    Traversal is the integer supercover of the segment that joins the two cell
    centers. The next boundary crossed is chosen by comparing
    `(1 + 2*ir) * nc` with `(1 + 2*ic) * nr` (all integers, no floats).

    Corner policy (conservative and deterministic): when that comparison ties, the
    segment passes through a grid corner. Both orthogonal neighbors are then on the
    ray (row-neighbor first, then col-neighbor; both are always inspected). The
    diagonal cell and everything beyond it are visible only when BOTH neighbors are
    free. One blocked neighbor is enough to stop the ray, so the agent cannot see
    through a diagonal crack or graze a single corner. If the queried cell is one
    of those two neighbors, it is visible: it sits on the near side of the corner.
    The rule is not symmetric in general; symmetry is not required.
    """
    if (r0, c0) == (r1, c1):
        return True
    dr, dc = r1 - r0, c1 - c0
    nr, nc = abs(dr), abs(dc)
    sr = (dr > 0) - (dr < 0)
    sc = (dc > 0) - (dc < 0)
    r, c = r0, c0
    ir = ic = 0
    while ir < nr or ic < nc:
        cmp_r = (1 + 2 * ir) * nc
        cmp_c = (1 + 2 * ic) * nr
        if cmp_r < cmp_c:
            r += sr
            ir += 1
            if (r, c) == (r1, c1):
                return True
            if blocked(r, c):
                return False
        elif cmp_c < cmp_r:
            c += sc
            ic += 1
            if (r, c) == (r1, c1):
                return True
            if blocked(r, c):
                return False
        else:
            side_r = (r + sr, c)
            side_c = (r, c + sc)
            diag = (r + sr, c + sc)
            if (r1, c1) in (side_r, side_c):
                return True
            if blocked(*side_r) or blocked(*side_c):
                return False
            if diag == (r1, c1):
                return True
            if blocked(*diag):
                return False
            r, c = diag
            ir += 1
            ic += 1
    return False


def configure_occlusion_demo(env):
    """Replace the current map with the fixed occlusion scene and recompute full-map BFS.

    BFS is refreshed only so a later `step` still applies the true potential. The
    observation does not read `goal_dist`.
    """
    grid = np.zeros((env.size, env.size), np.int8)
    for r, c in OCCLUSION_DEMO_OBSTACLES:
        grid[r, c] = 1
    env.grid = grid
    env.pos = OCCLUSION_DEMO_START
    env.start = OCCLUSION_DEMO_START
    env.goal = OCCLUSION_DEMO_GOAL
    env.goal_dist = bfs_dist(env.grid, env.goal)
    env.t = 0
    env.trajectory = [env.pos]
    return env


@dataclass
class RewardConfig:
    # Potential-based shaping (Ng, Harada & Russell 1999): F = scale * (gamma * Phi(s') - Phi(s)),
    # Phi(s) = -d_BFS(s, goal) (full map used for the REWARD ONLY, never observed), Phi(terminal) = 0 for BOTH
    # terminal kinds (goal and collision). gamma MUST equal the agent's discount (passed from the config).
    # Border bump: s' = s, so F = scale * (gamma - 1) * Phi(s). Timeout is truncation (not terminal): Phi(s') used normally.
    shaping_scale: float = 0.1
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
                 dist_metric="bfs", min_start_goal_dist=None, rewards: RewardConfig = None, gamma=0.99,
                 obs_mode=OBS_MODE_WINDOW):
        assert size >= 5 and obs_size in (3, 5, 7, 9)
        assert obs_size % 2 == 1
        self.size, self.k, self.density = size, obs_size, density
        # Default "window" keeps the historical observation. Callers that omit obs_mode
        # (every existing script and checkpoint) stay on that path.
        self.obs_mode = normalize_obs_mode(obs_mode)
        self.max_steps = max_steps or 8 * size  # 80 at N=10 (unchanged); 120 at N=15, 160 at N=20
        self.dist_metric = dist_metric
        # minimum Manhattan start-goal distance. Default 7 = largest window side (>= k for k in 3/5/7), so that the
        # held-out test maps are IDENTICAL for every method and every window size (map generation does not depend on k).
        self.min_sg = min_start_goal_dist if min_start_goal_dist is not None else max(7, obs_size)
        self.R = rewards or RewardConfig()
        self.gamma = gamma  # discount used by the potential-based shaping term; must equal the agent's gamma
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

    def _phi(self, p):
        """Shaping potential Phi(s) = -distance to goal (0 at the goal)."""
        return -float(self._d(p))

    def _obs(self):
        # Los mode is a separate function so the historical window body stays intact.
        if self.obs_mode == OBS_MODE_LOS:
            return self._obs_los()
        return self._obs_window()

    def _obs_window(self):
        k, h, n = self.k, self.k // 2, self.size
        padded = np.ones((n + 2 * h, n + 2 * h), np.float32)  # out-of-bounds = obstacle
        padded[h:h + n, h:h + n] = self.grid
        r, c = self.pos
        obst = padded[r:r + k, c:c + k]
        goal = np.zeros((k, k), np.float32)
        gr, gc = self.goal[0] - r + h, self.goal[1] - c + h
        if 0 <= gr < k and 0 <= gc < k:
            goal[gr, gc] = 1.0
        d = np.array([(self.goal[0] - r) / (n - 1), (self.goal[1] - c) / (n - 1)], np.float32)  # drow, dcol in [-1, 1]; /(N-1) adapts to N
        return np.concatenate([obst.ravel(), goal.ravel(), d]).astype(np.float32)

    def _blocked(self, r, c):
        """Obstacle or out-of-bounds. Used only by line-of-sight, not by rewards."""
        if not (0 <= r < self.size and 0 <= c < self.size):
            return True
        return int(self.grid[r, c]) == 1

    def _cell_visible(self, r, c):
        ar, ac = self.pos
        return los_visible(self._blocked, ar, ac, int(r), int(c))

    def _obs_los(self):
        """Line-of-sight observation. No map memory: only `grid`, `pos`, and `goal`.

        Out-of-bounds window cells are always OCC_OBSTACLE (1), the same encoding as
        window mode, including cells past the first layer of padding. Unknown (-1)
        is reserved for in-map cells the sightline does not reach. The goal channel
        is local sensing and is suppressed when the goal cell is hidden. The offset
        is the true goal offset either way.
        """
        k, h, n = self.k, self.k // 2, self.size
        r, c = self.pos
        obst = np.empty((k, k), np.float32)
        for i in range(k):
            rr = r + (i - h)
            for j in range(k):
                cc = c + (j - h)
                if not (0 <= rr < n and 0 <= cc < n):
                    obst[i, j] = OCC_OBSTACLE
                elif self._cell_visible(rr, cc):
                    obst[i, j] = OCC_OBSTACLE if int(self.grid[rr, cc]) == 1 else OCC_FREE
                else:
                    obst[i, j] = OCC_UNKNOWN
        goal = np.zeros((k, k), np.float32)
        gr, gc = self.goal[0] - r + h, self.goal[1] - c + h
        if 0 <= gr < k and 0 <= gc < k and self._cell_visible(self.goal[0], self.goal[1]):
            goal[gr, gc] = 1.0
        d = np.array([(self.goal[0] - r) / (n - 1), (self.goal[1] - c) / (n - 1)], np.float32)
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
        phi_s = self._phi(self.pos)
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            reward += R.no_move  # bumped the border, stays in place
            phi_next = phi_s
        elif self.grid[nr, nc] == 1:
            reward += R.collision
            term, outcome = True, "collision"
            self.pos = (nr, nc)
            phi_next = 0.0  # terminal state: Phi = 0
        else:
            self.pos = (nr, nc)
            phi_next = self._phi(self.pos)  # = 0 at the goal (d = 0), consistent with Phi(terminal) = 0
            if self.pos == self.goal:
                reward += R.goal
                term, outcome = True, "success"
        reward += R.shaping_scale * (self.gamma * phi_next - phi_s)  # strictly potential-based
        self.trajectory.append(self.pos)
        trunc = (not term) and self.t >= self.max_steps
        if trunc:
            reward += R.timeout
            outcome = "timeout"
        # after a collision the position is the obstacle cell; the window is still well-defined (terminal obs)
        return self._obs(), float(reward), term, trunc, self._info(outcome)

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
