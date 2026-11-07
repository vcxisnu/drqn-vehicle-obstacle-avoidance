"""Pygame visualization of a trained agent (reuses envs/agents; no training/eval logic changed).

Interactive:  python3 visualize.py [--run RUN_SEED_DIR]
   N / Space : new random map (env generation rules)    R : replay same map
   + / -     : speed up / slow down                      Q / Esc : quit
Headless GIF: python3 visualize.py --record out.gif --episodes 4   (uses SDL_VIDEODRIVER=dummy)
Default checkpoint: best DRQN k=5 seed (highest eval_d0.2 success) under results/runs.
"""
import argparse, glob, json, os, sys
import numpy as np
from common import RESULTS_DIR
from envs.grid_env import OBS_MODE_LOS, OCC_UNKNOWN

CELL, PANEL = 44, 70


def best_run(root=os.path.join(RESULTS_DIR, "runs", "drqn_s10_k5_d0.2")):
    best, score = None, -1
    for d in sorted(glob.glob(f"{root}/seed*")):
        if not os.path.exists(os.path.join(d, "model.pt")):
            continue
        e = os.path.join(d, "eval_d0.2.json")
        s = json.load(open(e))["success_rate"] if os.path.exists(e) else 0
        if s > score:
            best, score = d, s
    if best is None:
        sys.exit(f"no checkpoint found under {root}; pass --run")
    return best


def draw(pg, screen, font, env, status, speed):
    n, k, h = env.size, env.k, env.k // 2
    screen.fill((255, 255, 255))
    for r in range(n):
        for c in range(n):
            rect = pg.Rect(c * CELL, r * CELL, CELL, CELL)
            if env.grid[r, c]:
                pg.draw.rect(screen, (60, 60, 60), rect)
    # trajectory
    pts = [(c * CELL + CELL // 2, r * CELL + CELL // 2) for r, c in env.trajectory]
    if len(pts) > 1:
        pg.draw.lines(screen, (120, 160, 255), False, pts, 3)
    # observation window (clipped to map)
    r, c = env.pos
    win = pg.Surface((k * CELL, k * CELL), pg.SRCALPHA); win.fill((255, 200, 0, 60))
    screen.blit(win, ((c - h) * CELL, (r - h) * CELL))
    pg.draw.rect(screen, (230, 160, 0), pg.Rect((c - h) * CELL, (r - h) * CELL, k * CELL, k * CELL), 3)
    # Los only: paint in-map cells the agent cannot see. Window mode does not enter this block,
    # so the historical god-view frame is unchanged. Truth (the grey obstacles) stays visible;
    # the blue wash is the occlusion, not a second map memory.
    if getattr(env, "obs_mode", "window") == OBS_MODE_LOS:
        occ = env._obs()[:k * k].reshape(k, k)
        for i in range(k):
            for j in range(k):
                if float(occ[i, j]) != OCC_UNKNOWN:
                    continue
                rr, cc = r + (i - h), c + (j - h)
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                shade = pg.Surface((CELL - 2, CELL - 2), pg.SRCALPHA)
                shade.fill((50, 110, 210, 120))
                screen.blit(shade, (cc * CELL + 1, rr * CELL + 1))
    sr, sc = env.start; gr, gc = env.goal
    pg.draw.rect(screen, (60, 200, 90), pg.Rect(sc * CELL + 6, sr * CELL + 6, CELL - 12, CELL - 12), 3)
    pg.draw.circle(screen, (220, 40, 40), (gc * CELL + CELL // 2, gr * CELL + CELL // 2), CELL // 3)
    car = pg.Rect(c * CELL + 8, r * CELL + 12, CELL - 16, CELL - 24)
    pg.draw.rect(screen, (30, 90, 220) if status != "collision" else (200, 0, 0), car, border_radius=6)
    for x in (car.left + 4, car.right - 4):
        for y in (car.top, car.bottom):
            pg.draw.circle(screen, (20, 20, 20), (x, y), 4)
    for i in range(n + 1):
        pg.draw.line(screen, (200, 200, 200), (0, i * CELL), (n * CELL, i * CELL))
        pg.draw.line(screen, (200, 200, 200), (i * CELL, 0), (i * CELL, n * CELL))
    col = {"success": (0, 150, 0), "collision": (200, 0, 0), "timeout": (200, 120, 0)}.get(status, (0, 0, 0))
    screen.blit(font.render(f"step {env.t}/{env.max_steps}   shortest {int(env.goal_dist[env.start])}   "
                            f"status: {status}", True, col), (8, n * CELL + 8))
    mode = getattr(env, "obs_mode", "window")
    hint = f"win {k}x{k}  {speed}/s  N:new R:replay +/-:speed Q:quit"
    if mode == OBS_MODE_LOS:
        hint = f"LOS k={k}  blue=unknown  offset known  {speed}/s  N/R/+/-/Q"
    screen.blit(font.render(hint, True, (80, 80, 80)), (8, n * CELL + 36))


def run_episode(env, agent, rng, seed):
    """Generator: yields after reset and after every step (uses env/agent APIs only)."""
    obs, info = env.reset(seed=seed)
    agent.reset_state()
    yield "running"
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, 0.0, rng))
        done = term or trunc
        yield info["outcome"] or "running"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="seed dir with config.json + model.pt")
    p.add_argument("--speed", type=int, default=6)
    p.add_argument("--record", default=None, help="write a GIF headlessly instead of opening a window")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--map_seed", type=int, default=None, help="first map seed (default random)")
    p.add_argument("--need_success", action="store_true", help="(record) ensure >=1 successful episode in the GIF")
    p.add_argument("--obs_mode", choices=["window", "los"], default=None,
                   help="override the checkpoint observation mode for this viewing only. "
                        "A mismatch is exploratory: it does not rewrite the checkpoint.")
    a = p.parse_args()
    if a.record:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame as pg
    import torch
    from common import load_run, make_env
    torch.set_num_threads(1)
    run = a.run or best_run()
    cfg, agent = load_run(run)
    train_mode = cfg.get("obs_mode", "window")
    if a.obs_mode is not None and a.obs_mode != train_mode:
        print(f"EXPLORATORY cross-mode view: checkpoint obs_mode={train_mode}, viewing obs_mode={a.obs_mode}. "
              "The checkpoint files are not modified.")
        cfg = dict(cfg, obs_mode=a.obs_mode)
    env = make_env(cfg)
    rng = np.random.default_rng(0)
    pg.init()
    screen = pg.display.set_mode((env.size * CELL, env.size * CELL + PANEL))
    pg.display.set_caption(f"{cfg['agent']} k={cfg['obs']} — {run}")
    font = pg.font.SysFont(None, 22)
    seed_rng = np.random.default_rng(a.map_seed)
    new_seed = lambda: int(seed_rng.integers(2**31 - 1))
    seed = a.map_seed if a.map_seed is not None else new_seed()

    if a.record:
        from PIL import Image
        frames, outcomes, tries = [], [], 0
        while len(outcomes) < a.episodes and tries < 200:
            tries += 1
            ep_frames, status = [], "running"
            for status in run_episode(env, agent, rng, seed):
                draw(pg, screen, font, env, status, a.speed)
                ep_frames.append(Image.frombytes("RGB", screen.get_size(), pg.image.tostring(screen, "RGB")))
            seed = new_seed()
            last_slot = len(outcomes) == a.episodes - 1
            if a.need_success and last_slot and "success" not in outcomes and status != "success":
                continue  # keep searching for a success to fill the last slot
            outcomes.append(status)
            frames += ep_frames + [ep_frames[-1]] * 6  # pause on the final frame
        frames = [f.quantize(64) for f in frames]
        frames[0].save(a.record, save_all=True, append_images=frames[1:], duration=int(1000 / a.speed), loop=0)
        print(f"saved {a.record}: {len(frames)} frames, outcomes={outcomes}, checkpoint={run}")
        return

    clock, ep, status, paused_until = pg.time.Clock(), run_episode(env, agent, rng, seed), "running", 0
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_q, pg.K_ESCAPE)):
                pg.quit(); return
            if e.type == pg.KEYDOWN:
                if e.key in (pg.K_n, pg.K_SPACE):
                    seed = new_seed(); ep = run_episode(env, agent, rng, seed)
                elif e.key == pg.K_r:
                    ep = run_episode(env, agent, rng, seed)
                elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    a.speed = min(60, a.speed + 2)
                elif e.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    a.speed = max(1, a.speed - 2)
        if ep is not None:
            try:
                status = next(ep)
            except StopIteration:
                ep = None
        draw(pg, screen, font, env, status, a.speed)
        pg.display.flip()
        clock.tick(a.speed)


if __name__ == "__main__":
    main()
