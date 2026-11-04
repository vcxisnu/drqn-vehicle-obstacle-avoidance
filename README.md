# 记忆增强深度强化学习：部分可观测栅格环境中的车辆避障（第一版代码）

两个实验：
- **实验一**：DQN（无记忆）vs DRQN（LSTM 记忆），参数量对齐。
- **实验二**：固定 DRQN，观测窗口 3×3 / 5×5 / 7×7。

## 目录结构
```
code/
├── envs/grid_env.py     # Gymnasium 环境：随机地图+BFS 可达性、最小起终点距离、局部观测、BFS 势函数塑形奖励、terminated/truncated、失败轨迹快照
├── agents/deep.py       # DQN / DRQN（Double DQN 目标、目标网络、经验回放；DRQN 为整回合存储+子序列采样+burn-in）
├── agents/cores.py      # 循环核心接口（目前只有 LSTM；以后改进 LSTM 可在此注册）
├── agents/tabular.py    # 表格 Q-learning 基线（低优先级，保留未调优）
├── common.py            # 构造环境/智能体、加载模型
├── train.py             # 训练（支持多种子 --seeds 0 1 2 3 4）
├── evaluate.py          # 固定测试集评估 + 简单/中等/困难分组 + 失败轨迹
├── aggregate.py         # 跨种子汇总：均值 ± 95% 置信区间（t 分布）→ results/summary.csv
├── plot.py              # 实验一、实验二、密度曲线三张图 → plots/
└── run_smoke.sh         # 冒烟测试（3 种子、短训练）
```

## 环境
- N×N 栅格（默认 10，可 10–20）；每回合随机障碍（`--density`），随机起终点；BFS 检查不可达则重新生成。
- **最小起终点曼哈顿距离** `--min_dist`：默认 7（= 最大窗口边长，≥ 所有 k）。故意不随 k 变化，使 3/5/7 三种窗口**使用完全相同的地图分布和测试集**。
- 动作：0 上 / 1 下 / 2 左 / 3 右。
- **状态** = k×k 局部窗口展平（障碍通道，越界视为障碍）+ k×k 目标通道（目标在视野内时为 1）+ 目标偏移 (行偏移, 列偏移)/(N−1)。向量长度 2k²+2。不含全局地图。
  - 注：除障碍外额外加了一个“目标是否在窗口内”的通道；若论文要求严格只用障碍窗口，可在 `_obs()` 中去掉（偏移量已能推出目标位置）。
- **奖励**（`RewardConfig`）：
  - 塑形：`0.1 × (d_BFS(s) − d_BFS(s'))`，势函数形式，绕圈总和为 0。BFS 用全局地图，**只用于计算奖励，不进入状态**。
  - 到达 +10；撞障碍 −10（terminated）；撞边界原地不动 −0.2；每步 −0.01。
  - 超时（默认 8N 步）：**truncated，无额外惩罚**，TD 目标照常自举。环境返回分开的 terminated/truncated，经验回放中也分开存储，只有 terminated 截断自举。
- **固定测试集**：第 i 张测试图由种子 `10_000_000 + i` 生成，所有方法/种子/窗口共用；训练不使用该范围。

## 网络
- DQN：输入 → FC(128) → FC(128) → **FC(w)**（w≈997，用来与 LSTM 参数量对齐）→ 4 个 Q 值
- DRQN：输入 → FC(128) → FC(128) → **LSTM(128)** → 4 个 Q 值
- 训练开始时打印参数量（k=5：DQN 155,901，DRQN 155,908）。
- ε 线性从 1.0 衰减到 0.01（前 60% 回合）；Adam lr 5e-4；batch 32；目标网络每 1000 步同步；Huber loss；Double DQN。
- DRQN 回放：按回合存储，采样长度 16 的子序列，前接最多 8 步 burn-in；burn-in 起点隐藏状态置零，burn-in 部分不计损失（梯度仍会流经 burn-in，和 R2D2 略有差异，可在论文中说明）。

## 运行
```bash
pip install -r requirements.txt
# 实验一（建议正式实验 ≥5 种子、更多回合）
python3 train.py --agent dqn  --obs 5 --episodes 20000 --seeds 0 1 2 3 4
python3 train.py --agent drqn --obs 5 --episodes 20000 --seeds 0 1 2 3 4
# 实验二
for k in 3 5 7; do python3 train.py --agent drqn --obs $k --episodes 20000 --seeds 0 1 2 3 4; done
# 评估（--run 可以是某个配置目录，自动遍历 seed*/）；--density 用于按密度分难度
python3 evaluate.py --run runs/drqn_s10_k5_d0.2 --n_maps 300
python3 evaluate.py --run runs/drqn_s10_k5_d0.2 --n_maps 300 --density 0.3
# 汇总 + 作图
python3 aggregate.py --runs runs --out results/summary.csv
python3 plot.py --runs runs --out plots
# 一键冒烟测试
./run_smoke.sh
```
输出：`runs/<配置>/seed<n>/` 下有 `config.json`、`train_log.csv`、模型、`eval_d<密度>.json`、`episodes_d<密度>.csv`（每张测试图的结果与难度）、`failures_d<密度>.json`（最多 50 个失败回合：地图、起终点、轨迹、corridor_score）。

## 完整实验
```bash
EP=10000 nohup ./run_full.sh > ../results_full_nohup.log 2>&1 &   # 训练
N=500 ./run_eval_all.sh                                          # 评估 + 汇总 + 作图 + 失败分析
```
输出在 `/workspace/thesis/results/`：`summary.csv / summary.md`（均值 ± 95% CI）、`plots/`（exp1/exp2 训练曲线、难度柱状图、窗口大小曲线、密度曲线）、`failures/`（分类统计 `failure_summary.csv` 和轨迹图）。

## 可视化（Pygame）
```bash
python3 visualize.py                                   # 默认加载 eval 成功率最高的 DRQN 5×5 种子
python3 visualize.py --run ../results/runs/dqn_s10_k5_d0.2/seed0
# 按键：N / 空格 新地图；R 重放同一地图；+ / - 调速；Q / Esc 退出
# 无显示器：录制 GIF（自动使用 SDL_VIDEODRIVER=dummy）
python3 visualize.py --record ../results/demo_drqn_5x5.gif --episodes 4 --need_success
```
画面：栅格线、障碍（深灰）、起点（绿框）、目标（红点）、小车（蓝）、观测窗口（黄色高亮，大小随模型的 k）、轨迹、步数与状态（success / collision / timeout）。

## 难度分组
- 按最短路径长度：简单 ≤9，中等 10–11，困难 ≥12（`--bins 9 11`，约为 10×10、密度 0.2 测试集的三分位数；换地图尺寸要重新设定）。
- 按障碍密度：用 `--density 0.1/0.2/0.3` 分别评估（`plots/density.png`）。

## 下一步
1. 正式实验：≥5 种子、更长训练；必要时调奖励系数、lr、序列长度。
2. 扩展到 15×15、20×20（需要重新设定 `--bins` 和最大步数）。
3. 失败案例分析：可视化 `failures_*.json`，统计狭窄通道（`corridor_score`）、死胡同、来回振荡。
4. 可选消融：去掉目标偏移输入、DQN+帧堆叠。
