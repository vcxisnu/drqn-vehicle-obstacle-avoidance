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

# 正式 N=10 遮挡实验（已完成，见下一节）。不写 results/runs，不改 window 结果。
./run_full_los.sh    # 4 配置 × 5 种子 × 10000 回合 → results/runs_los/
./run_eval_los.sh    # 500 张固定图，贪心 + ε=0.01 → summary_los / plots_los / failures_los / detour_los

# 把 window 检查点拿来看 los 观测：探索性，单独目录
python3 evaluate.py --run ../results/runs/drqn_s10_k5_d0.2 --obs_mode los --n_maps 500
```

不带 `--obs_mode`（或显式 `--obs_mode window`）时，训练、评估、作图的数值路径与改动前的 window 模式一致。

## 正式 N=10 遮挡训练（已完成，20/20）

这是补充实验，写在本分支，**没有合并进 main，也没有改写** `results/runs/`、`results/summary.csv` 和上表里的 window 数字。模型是在 `obs_mode=los` 下从头训练的，不是 window 检查点，也不是 50 回合冒烟模型。

协议与 `run_full.sh` 的 N=10 相同，并逐项核对过 20 个 `config.json`、`train_log.csv`（各 10000 行，最后一回合编号 9999）和 `model.pt` metadata：

- 配置：DQN k=5，DRQN k=3/5/7；种子 0–4。共 20 个模型。
- 每个模型 10000 回合；密度 0.2；`min_dist` 空（环境默认曼哈顿距离 ≥7）；`max_steps` 空（默认 8N=80）。
- γ=0.99，奖励 `pbrs_v2`，ε 从 1.0 线性降到 0.01（前 60% 回合），lr 5e-4，batch 32，DRQN `seq_len=16`、`burn_in=8`，`train_every=4`，`target_update=1000`，`learn_start=1000`，hidden 128。
- 网络尺寸与 window 相同（k=5：DQN 155,901，DRQN 155,908）。输入维度仍是 `2k²+2`。
- 检查点格式 `drqn_ckpt_v1`，`metadata.obs_mode=los`。路径：`results/runs_los/<agent>_s10_k<k>_d0.2_los/seed<s>/model.pt`。

评估：每个模型在同一批 500 张图上评估，种子 `10_000_000+i`。已与仓库里 window 模式 DQN k5 seed0 的 `episodes_d0.2.csv` 逐图核对 `map_seed` 与最短路，500 张完全一致。难度仍是最短路 ≤9 / 10–11 / ≥12，本测试集为 226 / 142 / 132。贪心（ε=0）和 ε=0.01 各一份。汇总只用 `results/runs_los`，主假设族仍是 4 个配置间比较 × 贪心 × overall，Holm 只在这个族内做。**没有做 window 对 los 的显著性检验。**

贪心与 ε=0.01，5 个种子均值 ± 95% CI（Student t）。完整表：`results/summary_los.md`。显著性：`results/significance_los.csv`。

| 配置 | los 贪心 | los ε=0.01 | window 贪心（已发表，仅对照） | window ε=0.01（仅对照） |
|---|---|---|---|---|
| DQN k5 | 0.629 ± 0.018 | 0.645 ± 0.016 | 0.689 ± 0.078 | 0.698 ± 0.068 |
| DRQN k3 | 0.873 ± 0.021 | 0.862 ± 0.021 | 0.875 ± 0.019 | 0.863 ± 0.021 |
| DRQN k5 | 0.889 ± 0.013 | 0.878 ± 0.017 | 0.926 ± 0.014 | 0.914 ± 0.021 |
| DRQN k7 | 0.857 ± 0.021 | 0.850 ± 0.016 | 0.916 ± 0.014 | 0.907 ± 0.016 |

los 贪心的简单 / 中等 / 困难：DQN k5 为 0.765 ± 0.043 / 0.639 ± 0.044 / 0.386 ± 0.046；DRQN k3 为 0.960 ± 0.016 / 0.892 ± 0.031 / 0.705 ± 0.047；DRQN k5 为 0.965 ± 0.020 / 0.920 ± 0.008 / 0.726 ± 0.054；DRQN k7 为 0.942 ± 0.022 / 0.876 ± 0.010 / 0.691 ± 0.059。贪心碰撞率四个配置都是 0.000；失败全是超时（超时率 DQN 0.371 ± 0.018，DRQN k3/k5/k7 为 0.127 / 0.111 / 0.143）。

**los 内部主假设族（贪心，overall，Holm）**，差值是 A−B：

| 比较 | A−B | Holm p | bootstrap 95% CI | 结论 |
|---|---|---|---|---|
| DQN k5 − DRQN k5 | −0.260 | 1.1e-08 | [−0.292, −0.226] | 显著（DRQN 更高） |
| DRQN k3 − k5 | −0.016 | 0.25 | [−0.041, +0.010] | 未检出差异 |
| DRQN k5 − k7 | +0.032 | 0.029 | [+0.007, +0.057] | 显著（k5 更高） |
| DRQN k3 − k7 | +0.016 | 0.25 | [−0.011, +0.044] | 未检出差异 |

ε=0.01 下 DQN k5 − DRQN k5 = −0.233，bootstrap CI [−0.265, −0.199]，这是探索性结果，不在 Holm 族里。同一批模型上，ε=0.01 相对贪心：DQN +0.016（探索性，CI 不含 0），DRQN k3 和 k5 各约 −0.011（探索性，CI 不含 0），k7 为 −0.007（未检出差异）。随机动作带来的碰撞率约 0.036–0.055。

失败标签（贪心，2500 次评估 = 5 种子 × 500 图；`results/failures_los/failure_summary.csv`）：DQN 失败 927 次，其中 91.6% 是两格来回，超时尾段 100% 严格周期。DRQN k3/k5/k7 失败 317 / 278 / 357 次，两格来回占 31.9% / 36.7% / 37.3%，长周期占 59.3% / 55.0% / 56.0%，超时尾段严格周期 75.1% / 79.9% / 81.5%。贪心下没有碰撞标签。

绕行比（贪心，`results/detour_los/detour_by_config.md`）：DQN 在直走图（n=363）0.739 ± 0.019，在 >1.5（n=22）降到 0.036 ± 0.047；DRQN k5 对应为 0.954 ± 0.011 和 0.309 ± 0.084。>1.5 只有 22 张图，区间宽。

图：`results/plots_los/exp1_dqn_vs_drqn.png`、`exp2_obs_size.png`、`eval_eps.png`、`exp1_training_curves.png`。`density.png` 只有训练密度 0.2，没有密度扫描。动画：`results/plots_los/drqn_k5_seed0_los.gif`，DRQN k5 种子 0，第一段就是测试图 `10000000`（评估记录为超时、80 步、最短路 13），其后两段来自同一随机数发生器的后续地图，没有为了成功去挑。三段结果是超时、成功、成功。

**局限（照实写）**：只做了 N=10、这一种遮挡编码和拐角规则、5 个种子。没有 N=15/20，没有新基线，没有调参。window 数字只放在旁边作对照，**不能**把“los 低于 window 的点估计”说成已经检验过的跨模式差异；两者的置信区间对 DQN 重叠很多，对 DRQN k5 点估计间隔开，但这不是一次跨模式检验。los 里 k5 高于 k7 约 3 个百分点，主族里算显著，差别不大。训练末 100 回合成功率（种子 0：DQN 0.78，DRQN k5 0.93）高于冻结模型的贪心测试，和 window 实验里“在线更新更高”的现象同类，不能把训练曲线当成测试成功率。

## 这次实际跑过的短冒烟（不是上面的正式结果）

正式 20 个模型完成之前，只跑过下面这条管线检查。它**不是**上一节的结果。输出在 `results/occlusion_smoke/`（gitignore）。

命令（在 `code/` 下）：

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
