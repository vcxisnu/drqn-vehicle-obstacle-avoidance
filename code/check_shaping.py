"""Numeric sanity check of the potential-based shaping (no training involved).

1) Value iteration on the fully-observable MDP of random 10x10 maps (density 0.2), gamma = 0.99:
   Q*_shaped(s,a) must equal Q*_orig(s,a) - Phi(s)  (Ng et al. 1999), so greedy optimal actions are identical.
2) Discounted return of degenerate behaviours (two-cell oscillation, wall bumping) vs. going straight to the goal,
   under the actual env.step() rewards: degenerate behaviours must never be preferable ("not gameable").
"""
import numpy as np
from envs.grid_env import GridObstacleEnv, ACTIONS, bfs_dist

G = 0.99


def rewards_table(env, shaped):
    n, R, goal = env.size, env.R, env.goal
    free = [tuple(p) for p in np.argwhere(env.grid == 0)]
    idx = {p: i for i, p in enumerate(free)}
    S = len(free)
    nxt = np.full((S, 4), -1); rew = np.zeros((S, 4)); term = np.zeros((S, 4), bool)
    for p in free:
        if p == goal:
            continue
        for a, (dr, dc) in enumerate(ACTIONS):
            env.pos, env.t = p, 0
            q = (int(p[0] + dr), int(p[1] + dc))
            r = env.step(a)[1] if shaped else None
            if not shaped:
                r = R.step
                if not (0 <= q[0] < n and 0 <= q[1] < n): r += R.no_move
                elif env.grid[q] == 1: r += R.collision
                elif q == goal: r += R.goal
            if not (0 <= q[0] < n and 0 <= q[1] < n): q = p
            rew[idx[p], a] = r
            if q == goal or (0 <= q[0] < n and 0 <= q[1] < n and env.grid[q] == 1):
                term[idx[p], a] = True
            else:
                nxt[idx[p], a] = idx[q]
    return free, idx, nxt, rew, term


def vi(nxt, rew, term, goal_i):
    V = np.zeros(len(nxt))
    for _ in range(3000):
        Q = rew + G * np.where(term, 0, V[np.maximum(nxt, 0)])
        Q[goal_i] = 0
        V2 = Q.max(1)
        if np.abs(V2 - V).max() < 1e-10: break
        V = V2
    return Q


def main(n_maps=200):
    max_err, mism, n_states = 0.0, 0, 0
    for i in range(n_maps):
        env = GridObstacleEnv(10, 5, 0.2, gamma=G); env.reset(seed=i)
        free, idx, nxt, rs, ts = rewards_table(env, True)
        _, _, _, ro, to = rewards_table(env, False)
        gi = idx[env.goal]
        Qs, Qo = vi(nxt, rs, ts, gi), vi(nxt, ro, to, gi)
        d = env.goal_dist
        reach = [j for j, p in enumerate(free) if d[p] > 0]
        phi = np.array([-0.1 * d[p] for p in free])  # scale * Phi
        err = np.abs(Qs[reach] - (Qo[reach] - phi[reach, None])).max()
        max_err = max(max_err, err)
        for j in reach:
            n_states += 1
            if set(np.flatnonzero(np.isclose(Qs[j], Qs[j].max(), atol=1e-9))) != \
               set(np.flatnonzero(np.isclose(Qo[j], Qo[j].max(), atol=1e-9))):
                mism += 1
    print(f"[VI] {n_maps} maps, {n_states} states: max |Q*_shaped - (Q*_orig - 0.1*Phi)| = {max_err:.2e}; "
          f"states with different optimal-action set: {mism}")

    # degenerate behaviours on a concrete map, per-step rewards from env.step()
    env = GridObstacleEnv(10, 5, 0.2, gamma=G); env.reset(seed=3)
    d = env.goal_dist; p0 = env.start
    def disc(rs): return sum(G ** t * r for t, r in enumerate(rs))
    # oscillation between p0 and a free neighbour, 1000 steps (~infinite horizon)
    for a, (dr, dc) in enumerate(ACTIONS):
        q = (p0[0] + dr, p0[1] + dc)
        if 0 <= q[0] < 10 and 0 <= q[1] < 10 and env.grid[q] == 0 and q != env.goal:
            back = [1, 0, 3, 2][a]; break
    env.pos, env.t = p0, 0; env.max_steps = 10**9
    osc = [env.step(a if t % 2 == 0 else back)[1] for t in range(1000)]
    # wall bumping: find a border cell? use the start cell and an out-of-bounds action if available, else stay via BFS free cell on border
    border = [tuple(p) for p in np.argwhere(env.grid == 0) if (p[0] in (0, 9) or p[1] in (0, 9)) and d[tuple(p)] > 0][0]
    ba = 0 if border[0] == 0 else 1 if border[0] == 9 else 2 if border[1] == 0 else 3
    env.pos = border; bump = [env.step(ba)[1] for _ in range(1000)]
    # optimal: follow BFS gradient from p0
    env.pos, path = p0, []
    while env.pos != env.goal:
        a = min(range(4), key=lambda a: d[(env.pos[0] + ACTIONS[a][0], env.pos[1] + ACTIONS[a][1])]
                if 0 <= env.pos[0] + ACTIONS[a][0] < 10 and 0 <= env.pos[1] + ACTIONS[a][1] < 10
                and d[(env.pos[0] + ACTIONS[a][0], env.pos[1] + ACTIONS[a][1])] >= 0 else 999)
        path.append(env.step(a)[1])
    print(f"[loops] start d_BFS={d[p0]}: two-cell oscillation: mean undiscounted reward/step={np.mean(osc):+.4f}, "
          f"discounted return={disc(osc):+.3f} (shaping part bounded by -0.1*Phi(s0)={0.1*d[p0]:.2f})")
    print(f"[loops] wall bump at {border} (d={d[border]}): reward/step={np.mean(bump):+.4f}, discounted={disc(bump):+.3f}")
    print(f"[loops] shortest path to goal ({len(path)} steps): discounted return={disc(path):+.3f}")
    # undiscounted per-step oscillation reward as function of d (for documentation)
    for dd in (2, 5, 10, 15, 20):
        f = 0.1 * ((dd - G * (dd + 1)) + ((dd + 1) - G * dd)) / 2
        print(f"   d={dd:2d}: two-cell oscillation mean per-step shaping {f:+.4f}, +step cost -0.01 => {f-0.01:+.4f}")


if __name__ == "__main__":
    main()
