"""Fixed-map figure: true grid vs historical kxk window vs line-of-sight observation.

The scene is `configure_occlusion_demo` (also used by tests). Two rows: before the robot
moves, and after it steps right twice through the gap, which reveals the cells the wall
was hiding. No policy is trained. Nothing under results/runs is read or written.

    python3 render_los_figure.py --out ../docs/figures/los_window_vs_occlusion.png
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from envs.grid_env import (
    OCC_FREE, OCC_OBSTACLE, OCC_UNKNOWN, GridObstacleEnv, configure_occlusion_demo,
)


def _font():
    """DejaVu draws the Latin labels. A CJK font, if present, is only a fallback."""
    from matplotlib import font_manager
    plt.rcParams["axes.unicode_minus"] = False
    cjk = None
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        if path and "dejavu" not in os.path.basename(path).lower():
            cjk = name
            break
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + ([cjk] if cjk else [])
    return cjk is not None


def make(mode):
    env = GridObstacleEnv(10, 5, density=0.0, gamma=0.99, obs_mode=mode)
    env.reset(seed=0)
    configure_occlusion_demo(env)
    return env


def channels(env):
    obs = env._obs()
    k2 = env.k * env.k
    return obs[:k2].reshape(env.k, env.k), obs[k2:2 * k2].reshape(env.k, env.k), obs[-2:].copy()


def draw_full(ax, env, title):
    ax.imshow(env.grid, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    n, k, h = env.size, env.k, env.k // 2
    r, c = env.pos
    ax.add_patch(Rectangle((c - h - 0.5, r - h - 0.5), k, k, fill=False, ec="#e6a000", lw=2.2))
    traj = np.array(env.trajectory)
    if len(traj) > 1:
        ax.plot(traj[:, 1], traj[:, 0], color="#7eb6ff", lw=2.2, zorder=3)
    gr, gc = env.goal
    ax.scatter([gc], [gr], marker="*", s=160, c="#d62728", edgecolors="white", linewidths=0.4, zorder=4)
    ax.scatter([c], [r], marker="o", s=90, c="#1f4e99", edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("column")
    ax.set_ylabel("row")


def draw_local(ax, env, occ, goal_ch, title):
    rgb = np.zeros(occ.shape + (3,))
    rgb[np.isclose(occ, OCC_FREE)] = (0.96, 0.94, 0.90)
    rgb[np.isclose(occ, OCC_OBSTACLE)] = (0.16, 0.16, 0.16)
    rgb[np.isclose(occ, OCC_UNKNOWN)] = (0.29, 0.45, 0.72)
    ax.imshow(rgb, interpolation="nearest")
    k, h = env.k, env.k // 2
    r, c = env.pos
    for i in range(k):
        for j in range(k):
            val = float(occ[i, j])
            label = {-1.0: "-1", 0.0: "0", 1.0: "1"}.get(val, f"{val:g}")
            ax.text(j, i, label, ha="center", va="center", fontsize=9, zorder=6,
                    color="white" if val != OCC_FREE else "#222222")
            if goal_ch[i, j] > 0.5:
                # Offset so the occupancy digit stays readable under the goal star.
                ax.scatter([j + 0.28], [i - 0.28], marker="*", s=90, c="#d62728", edgecolors="white",
                           linewidths=0.3, zorder=5)
    ax.scatter([h + 0.22], [h + 0.22], marker="o", s=55, c="#1f4e99", edgecolors="white",
               linewidths=0.7, zorder=5)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([str(c + (j - h)) for j in range(k)], fontsize=8)
    ax.set_yticklabels([str(r + (i - h)) for i in range(k)], fontsize=8)
    ax.set_xlabel("map column")
    ax.set_ylabel("map row")
    ax.set_title(title, fontsize=11)


def render(path):
    _font()
    before_w, before_l = make("window"), make("los")
    after_w, after_l = make("window"), make("los")
    for env in (after_w, after_l):
        env.step(3)  # right, into the gap
        env.step(3)  # right, beside the cells the wall was hiding
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.8))
    panels = (
        (0, before_w, before_l, "Before movement  (robot at row 5, col 2)"),
        (1, after_w, after_l, "After two steps right  (robot at row 5, col 4)"),
    )
    for row, env_w, env_l, row_title in panels:
        occ_w, goal_w, off_w = channels(env_w)
        occ_l, goal_l, off_l = channels(env_l)
        draw_full(axes[row, 0], env_l, "Full map (true occupancy)")
        draw_local(axes[row, 1], env_w, occ_w, goal_w,
                   f"Historical window   goal channel sum={int(goal_w.sum())}")
        draw_local(axes[row, 2], env_l, occ_l, goal_l,
                   f"Occluded observation   goal channel sum={int(goal_l.sum())}")
        axes[row, 0].annotate(row_title, xy=(0, 1.18), xycoords="axes fraction", fontsize=10, fontweight="bold")
        assert np.allclose(off_w, off_l)
    off = channels(before_l)[2]
    legend = [
        Patch(facecolor=(0.96, 0.94, 0.90), edgecolor="#888", label="0  free (visible)"),
        Patch(facecolor=(0.16, 0.16, 0.16), edgecolor="#888", label="1  obstacle, or out-of-bounds boundary"),
        Patch(facecolor=(0.29, 0.45, 0.72), edgecolor="#888", label="-1  unknown (in-map, behind a blocker)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f4e99", markeredgecolor="white",
                   markersize=8, label="robot"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728", markersize=12,
                   label="goal (star in a local panel only if that mode's goal channel is 1)"),
        Patch(facecolor="none", edgecolor="#e6a000", label="k=5 window on the full map"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.005))
    caption = (
        f"Known goal offset is the same in both modes and is not part of local sensing: "
        f"(Δrow, Δcol)/(N−1) = ({off[0]:+.3f}, {off[1]:+.3f}) before the move. "
        "Before the move the goal cell and the obstacle behind the wall are -1 under line-of-sight, "
        "while the first wall cell stays 1; the historical window still shows the true 0 and 1, and its goal channel is 1. "
        "Corner rule: a diagonal step is blocked if either orthogonal neighbor is blocked "
        "(the crack at (3,0) is -1; its two touching obstacles stay 1). "
        "Out-of-bounds cells stay 1, matching the historical boundary encoding; this window is inside the map, so every -1 is occlusion. "
        "No accumulated map: the bottom row is a fresh sightline from the new cell. Reward BFS is not drawn."
    )
    fig.suptitle("Optional line-of-sight occlusion  vs  historical local window   (N=10, k=5, fixed map)",
                 fontsize=13, y=0.98)
    fig.text(0.012, 0.175, textwrap.fill(caption, 168), fontsize=7.4, va="top", ha="left")
    fig.tight_layout(rect=(0, 0.20, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "figures",
                                                 "los_window_vs_occlusion.png"))
    render(p.parse_args().out)
