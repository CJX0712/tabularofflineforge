# TabularOfflineForge

> 离线强化学习（Offline RL）—— 以**精确表格化真值**为黄金标准的 FQI / CFQI 控制 + FQE / WIS 离线策略评估（OPE）。

- 作者：**晨星**（GitHub: [CJX0712](https://github.com/CJX0712)） · MIT License
- 依赖：纯 `numpy` + `scikit-learn`（RandomForest），**无需 GPU**
- 可验证不变量驱动：每个核心结论都有独立交叉验证，误差真实可复现
- 仓库 slug：`tabularofflineforge`（避免与同名既有仓库冲突）；**内部 Python 包名仍为 `offlineforge`**，导入不变。

---

## 为什么是「表格化」？

离线 RL 的致命问题是**外推误差（extrapolation error）**——模型对从未见过的状态-动作对给出过度乐观的 Q 值。要度量这个问题，必须有**真值**。

OfflineForge 把问题限定在**表格化 MDP**：策略是一个 `(n_states, n_actions)` 概率矩阵，Bellman 期望方程

```
V^π = r_π + γ · P_π · V^π      →      V^π = (I − γP_π)⁻¹ · r_π
```

是一个**线性方程组**，用 `np.linalg.solve` 直接精确求解（机器精度）。于是：

- 任意策略的精确价值 `J(π) = μ · V^π` 唾手可得；
- OPE 估计器（FQE / WIS）**只能看到离线日志数据**，它们与真值的差距就是**诚实的离线评估误差**。

所有环境都是**精确已知**的（P、R、μ、γ 全给出），离线数据集只由行为策略（behaviour policy）采样生成——这正是 OPE 估计器被允许看到的唯一信息。

## 两个原创旗舰

| 旗舰 | 类型 | 作用 | 关键不变量 |
|------|------|------|-----------|
| **FQI** (Fitted Q-Iteration) | 控制 | 从离线数据 Bootstrap 拟合 Q 函数，逼近最优价值 | 数据/迭代充足时 `J^FQI → J*` |
| **CFQI** (Conservative FQI) | 悲观主义 | 在 FQI 学到的 Q 上减去**不确定性惩罚**，给出悲观下界 | **`J^CFQI ≤ J^FQI`**（惩罚非负 ⇒ 硬下界） |

CFQI 是 CQL / 悲观离线 RL 思想的极简体现：用集成模型（RandomForest）的树间方差作为不确定性 `u`，在贪心动作上做 `max_a Q − λ·u`。**CFQI 复用 FQI 拟合后的同一个 Q 函数**，因此下界关系是数学保证，而非统计近似。

## 两个 OPE 评估器（只吃日志数据）

| 评估器 | 方法 | 特点 |
|--------|------|------|
| **FQE** (Fitted Q Evaluation) | 模型法：跑 Bellman **评估**（非贪心）迭代拟合 `Q` | 可评估任意目标策略；训练一个 RF |
| **WIS** (Weighted Importance Sampling) | 无模型：轨迹重要性权重加权平均折扣回报 | 直接、无偏（期望上）；高方差，要求目标策略对行为策略**绝对连续**（全支撑） |

两者都**只使用离线数据集**评估给定目标策略的价值，与精确求解器对比，量化其排序保真度（Kendall τ）与最优策略选拔准确率。

## 环境

| 环境 | 状态/动作 | 难度点 |
|------|----------|--------|
| GridWorld | 25 / 4 | 4 连通网格，到达右下角 +1（**进入该步即奖励**） |
| RiverCrossing | 25 / 4 | 河带阻断，仅桥格可过，落水回起点 −1 |
| ChainMDP | 10 / 2 | 链状，右端 +1，含转移滑移 |
| TinyRandom | 8 / 2 | 随机 MDP，OPE 方差/覆盖压力测试 |

## 安装

```bash
git clone https://github.com/CJX0712/tabularofflineforge.git
cd tabularofflineforge
python -m venv .venv && .venv/Scripts/activate        # Windows
pip install -r requirements.lock.txt                  # 或 requirements.txt
pip install -e .                                      # 可编辑安装 offlineforge 包（可选）
```

## 快速开始

```bash
# 端到端基准 → 写出 benchmark.json 并打印摘要
python -m offlineforge.cli --out benchmark.json

# 仅跑部分环境、减少日志量
python -m offlineforge.cli --envs GridWorld ChainMDP --episodes 200

# 演示（与上方等价，但固定随机种子便于复现）
python -m offlineforge.examples.run_demo
```

所有配置可通过 `OF_*` 环境变量覆盖（见 `offlineforge/core/config.py`）：

```bash
OF_SEED=7 OF_FQI_ITERS=40 OF_N_ESTIMATORS=64 python -m offlineforge.cli
```

## 基准结果（最新）

> 数值由 `offlineforge.examples.run_demo` 在固定种子（seed=0）下生成，完整结果见 [`benchmark.json`](./benchmark.json)。
> 下列聚合指标为最近一次运行（4 环境、默认配置）的结果：

| 指标 | 数值 |
|------|------|
| `mean \|FQI − 精确最优\|` | **0.1855** |
| `mean \|CFQI − 精确最优\|` | **0.1264** |
| `mean \|FQE − 精确\|` | **0.1566** |
| `mean \|WIS − 精确\|` | **0.6856** |
| FQE 最优策略选拔准确率 | **75%** (3/4) |
| WIS 最优策略选拔准确率 | **75%** (3/4) |
| 平均 Kendall τ（FQE / WIS） | **0.95 / 0.30** |
| `J^CFQI ≤ J^FQI` 不变量 | ✅ **成立**（所有环境） |

**逐环境诚实说明**（控制误差 `\|FQI − J*\|` 与选拔正确性）：

| 环境 | `\|FQI−J*\|` | FQE 选拔✅ | WIS 选拔✅ | τ(FQE) | τ(WIS) |
|------|------:|------|------|------:|------:|
| GridWorld | 0.0075 | ❌ | ✅ | 0.80 | 1.00 |
| RiverCrossing | 0.3938 | ✅ | ❌ | 1.00 | −1.00 |
| ChainMDP | 0.0057 | ✅ | ✅ | 1.00 | 0.80 |
| TinyRandom | 0.3348 | ✅ | ✅ | 1.00 | 0.40 |

- **FQI 在 GridWorld / ChainMDP 上几乎复现精确最优**（误差 < 0.01），在 RiverCrossing / TinyRandom 上偏差较大（随机/稀疏奖励导致覆盖不足）。
- **WIS 方差大**：在 RiverCrossing 上对最优策略的误差达 3.24，Kendall τ 一度为 −1.0——这正符合 WIS 对支撑失配高方差的理论预期，属**诚实披露**而非缺陷。
- **FQE 排序保真度（τ）整体优于 WIS**，且 CFQI 始终给出不高于 FQI 的悲观下界。

## 可验证不变量（均有单测覆盖）

| 不变量 | 含义 | 验证方式 |
|--------|------|---------|
| `J^CFQI ≤ J^FQI` | 悲观下界 | CFQI 复用 FQI 同一 Q，惩罚非负 ⇒ 硬下界 |
| `J^FQI → J*` | 控制收敛 | 充足数据下贪心 FQI 价值逼近精确最优 |
| `(I − γP_π)⁻¹ r` | 精确求解 | 与 `np.linalg.inv` 直接求逆对照 `atol=1e-9` |
| WIS 要求全支撑 | 支撑失配报错 | 对确定性目标策略抛 `OPEError` |
| Kendall τ | 排序保真度 | 平局正确取平均秩（τ=0） |

## 项目结构

```
offlineforge/
├── core/        类型(TransitionDataset)、接口(Protocol)、配置、误差码、特征编码
├── envs/        精确 MDP 定义 + 离线数据集生成 + 线性求解器(solver)
├── algos/       FQI（控制旗舰）、CFQI（悲观旗舰）
├── ope/         FQE（模型法 OPE）、WIS（无模型 IS）
├── eval/        OPE 排序保真度（Kendall τ、选拔准确率）
├── pipeline/    基准编排：生成→精确→FQI/CFQI→FQE/WIS→聚合
├── cli.py       argparse 入口
├── examples/    run_demo.py 端到端演示
tests/           42 项 pytest 单测（覆盖上述所有不变量）
docs/architecture.md   架构与数据流详解
```

## 许可证与署名

MIT License · © 晨星 (CJX0712)
