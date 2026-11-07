# 可选视线遮挡（line-of-sight occlusion）

本文档描述 **可选** 的 `obs_mode=los`。默认 `obs_mode=window` 与仓库里已经训练好的模型和已经发表的成功率完全是同一套观测。

## 已发表的数字不是遮挡结果

10×10、密度 0.2、500 张固定测试图、5 个种子、贪心评估（均值 ± 95% CI）仍然是 **无遮挡局部窗口**：

| 配置 | 贪心成功率 |
|---|---|
| DQN k5 | 0.689 ± 0.078 |
| DRQN k3 | 0.875 ± 0.019 |
| DRQN k5 | 0.926 ± 0.014 |
| DRQN k7 | 0.916 ± 0.014 |

这些数字、`results/runs/` 下的检查点，以及 `results/summary.csv` 都 **不是** 视线遮挡实验。遮挡模式没有重训那 20 个（或更大地图上的 40 个）模型。下文的短训练只检查管线能否跑通，**不能**用来声称遮挡下 DQN 或 DRQN 更好或更差。

## 假设

- 地图仍是静态 N×N 栅格。动作、地图生成、最小曼哈顿距离 7、`max_steps = 8N`、终止/截断、奖励与严格势函数塑形都不因观测模式改变。
- 智能体没有外部环境里的累积地图。每一步观测只由 **当前** 格子、**当前** 位置和 **当前** 目标决定。DRQN 的历史只在 LSTM 隐状态里。
- 奖励用的 BFS 距离仍来自完整地图，只进入势函数，不进入观测。
- 目标偏移 `(Δrow, Δcol) / (N−1)` 是 **已知的侧信息**，和局部视线分开。目标被墙挡住时，偏移照样给出；局部目标通道则不再标出那个格子。
- 输入维度不变：仍是 `2k²+2`。网络（DQN 的 FC128-FC128-FC997-4，DRQN 的 FC128-FC128-LSTM128-4）、回放（DQN：`deque` + `random.sample`；DRQN：整回合、burn-in 8 + 序列 16，梯度穿过 burn-in）都不改。
- 比较仍是原来的两组：DQN vs DRQN（k=5），以及 DRQN 的 k=3/5/7。本改动不增加基线，也不做遗忘门实验。

## 编码

占用通道（向量的前 `k²` 项）只用三个标量，都在原有的 `Box(-1, 1)` 里：

| 值 | 常数 | 含义 |
|---|---|---|
| `0` | `OCC_FREE` | 看见的空地。与旧窗口模式相同。 |
| `1` | `OCC_OBSTACLE` | 看见的障碍，**以及一切越界格**。与旧窗口模式相同。 |
| `-1` | `OCC_UNKNOWN` | 仅 `los`：地图内、视线到不了的格子。真实是空地还是障碍都不写进来。 |

`-1` 不在 0 和 1 之间，避免被读成“半个障碍”。旧模式的占用通道里不会出现 `-1`。

目标通道（接下来的 `k²` 项）：

- `window`：目标格落在窗口内则为 1（旧行为）。
- `los`：目标格在窗口内 **并且** 该格可见，才为 1。被挡住时这一通道是 0，避免从局部通道泄露墙后的目标格。

最后两项永远是真实的归一化偏移，两种模式数值相同。

## 视线与拐角

从机器人所在格的 **格心** 到目标格的 **格心** 做整数超覆盖（supercover）。比较 `(1+2·ir)·nc` 与 `(1+2·ic)·nr`，全程整数。

- 起点格子自己不挡射线。撞上障碍之后，机器人站在障碍格上，这一格记为 1，但不会把整张窗口都变成未知。
- 射线上 **第一面** 障碍可见，记为 1。它后面的格子，不论真实占据，一律 `-1`。
- **拐角（保守、确定）**：比较打成平手时，线段正好穿过格点角。这一步的两个正交邻居都算在射线上（先横行邻居，再纵列邻居，两个都看）。只有 **两个邻居都是空地**，视线才能进到对角格以及更远处。任一邻居是障碍或越界，对角格及更远处都不可见。因此两块障碍只在角上相接时不能透视，贴着单个障碍的对角也不能“擦角”看过去。被问到的那个邻居本身可见（它在拐角的近侧）。此规则一般不对称，实现不要求对称。

## 边界

窗口落在地图外的格子 **恒为 1**，和旧模式逐格相同，包括贴边第二层和斜向外的越界格。越界是已知边界，不是“没看见的地图格”，所以不用 `-1`。

空地图上（没有任何障碍），任意位置、任意 k，`los` 与 `window` 的整条观测向量相同：地图内全可见，越界仍是 1，目标通道与偏移也一致。有障碍时，两种模式只可能在“地图内被挡住的格子”和“因此被关掉的目标通道”上不同。

## 检查点、评估与汇总

- `config.json` 写入 `obs_mode`。缺这个字段的旧配置视为 `window`。
- 新的 DQN/DRQN `model.pt` 是 `{format: drqn_ckpt_v1, state_dict, metadata}`，`metadata.obs_mode` 必须和 `config.json` 一致，否则 `load_run` 拒绝加载。
- 旧的 `model.pt` 仍是裸 `state_dict`，没有 metadata，只能当作 `window`。配置若写成 `los`，加载会失败，避免把旧权重 silently 配上新观测。
- 训练默认路径：`window` 仍是 `results/runs/<agent>_sN_kK_dD/seed*`；`los` 是 `results/runs_los/<agent>_sN_kK_dD_los/seed*`。`los` 若指向 `results/runs/` 会被拒绝。
- 评估默认跟检查点的模式走。`--obs_mode` 与检查点不一致时，是 **探索性跨模式评估**：不能写进检查点目录，也不能写进 `results/runs/`；默认目录是 `results/exploratory_cross_mode/<train>_to_<eval>/`；文件名带 `_xmode_<mode>`，json 里 `exploratory_cross_mode: true`。`aggregate.py` 默认跳过这些文件。
- 同一张汇总表里如果同时有 `window` 和 `los`，`aggregate.py` 默认直接退出。`--allow_mixed_obs_modes` 才会分模式做检验，跨模式比较标成探索性，并且仍然拒绝覆盖 `results/summary.csv`。`plot.py` / `failure_analysis.py` / `detour_analysis.py` 同样拒绝把非 window 结果写进 `results/plots`、`results/failures`、`results/detour`。

## 复现

在 `code/` 下：

```bash
python3 -m pytest tests -q

# 示意图（全图 / 旧窗口 / 实际遮挡观测，移动前与向右两步之后）
python3 render_los_figure.py --out ../docs/figures/los_window_vs_occlusion.png

# 正式遮挡训练不要写入 results/runs。默认目录是 results/runs_los。
# 下面是与主实验相同的算法设置；回合数按论文应为 10000、种子 0–4。
# 本仓库没有用这个命令重训，也不要把输出当成已发表结果。
python3 train.py --agent dqn  --size 10 --obs 5 --density 0.2 --obs_mode los \
    --episodes 10000 --seeds 0 1 2 3 4
python3 train.py --agent drqn --size 10 --obs 5 --density 0.2 --obs_mode los \
    --episodes 10000 --seeds 0 1 2 3 4
python3 evaluate.py --run ../results/runs_los/drqn_s10_k5_d0.2_los --n_maps 500
python3 aggregate.py --runs ../results/runs_los --out ../results/summary_los.csv

# 把 window 检查点拿来看 los 观测：探索性，单独目录
python3 evaluate.py --run ../results/runs/drqn_s10_k5_d0.2 --obs_mode los --n_maps 500
```

不带 `--obs_mode`（或显式 `--obs_mode window`）时，训练、评估、作图的数值路径与改动前的 window 模式一致。

## 这次实际跑过的短冒烟（不是有效性）

只跑了管线，没有重训 20 个正式模型，也没有跑大地图。命令（在 `code/` 下，输出在 `results/occlusion_smoke/`，已被 gitignore，不进入论文结果）：

```bash
python3 train.py --agent dqn  --size 10 --obs 5 --density 0.2 --obs_mode los --episodes 50 --seeds 0 \
    --out ../results/occlusion_smoke/dqn_s10_k5_d0.2_los
python3 train.py --agent drqn --size 10 --obs 5 --density 0.2 --obs_mode los --episodes 50 --seeds 0 \
    --out ../results/occlusion_smoke/drqn_s10_k5_d0.2_los
python3 evaluate.py --run ../results/occlusion_smoke/dqn_s10_k5_d0.2_los --n_maps 5
python3 evaluate.py --run ../results/occlusion_smoke/drqn_s10_k5_d0.2_los --n_maps 5
```

超参数保持原样（`learn_start=1000`，ε 在前 60% 回合线性降到 0.01，所以 50 回合里大约第 30 回合起 ε 已经是 0.01）。种子 0：

| | 交互步数 | 日志里出现过 loss 的回合 | 最后一条 loss | 训练末 100 回合成功率 | 贪心评估 5 张图 |
|---|---|---|---|---|---|
| DQN k5 los | 1604 | 12 | 0.164 | 0.02 | 成功 0.000（碰撞 0.200，超时 0.800） |
| DRQN k5 los | 1107 | 4 | 1.096 | 0.00 | 成功 0.000（碰撞 0.400，超时 0.600） |

检查点 metadata 为 `obs_mode=los`、`obs_dim=52`。评估 json 带 `obs_mode=los`，没有 `exploratory_cross_mode`。成功率是 0 只说明 50 回合不够学，**不能**解释成遮挡模式无效或有效。

## 示意图

`docs/figures/los_window_vs_occlusion.png`（由 `code/render_los_figure.py` 从固定地图画出）：

- 机器人在 (5, 2)，k=5。墙 (4, 3)、(6, 3) 挡住空地目标 (4, 4) 和障碍 (6, 4)。拐角障碍 (4, 0)、(3, 1) 挡住 (3, 0)。
- 左：真实全图。中：旧窗口（墙后仍是 0 或 1，目标通道为 1）。右：遮挡观测（墙后为 −1，第一面墙为 1，目标通道为 0）。偏移两种模式相同。
- 下一行：向右走两步穿过缺口之后，原先被挡住的空地和障碍都变成可见。这是新位置上重新算的视线，不是一张记下来的地图。
