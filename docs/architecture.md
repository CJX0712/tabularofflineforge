# OfflineForge 架构文档

## 1. 设计哲学：用「精确可解」换「可验证」

离线 RL 的最大痛点是**没有真值**。OfflineForge 选择全部使用**表格型 MDP**，
使得策略价值成为线性方程组的精确解，从而：

- 任意策略 `π` 的精确价值 `J(π) = μᵀ(I − γP_π)⁻¹ r_π` 可机器精度求解；
- OPE 估计器（FQE / WIS）与**控制**估计器（FQI / CFQI）的误差可被独立交叉验证；
- 整个框架零 GPU、纯 numpy + scikit-learn、CPU 可跑、确定性可复现。

```
            已知模型 (envs/specs.py)
                    │  exact_value()
                    ▼
            ┌───────────────┐
            │  solver.py    │  金标准真值 V^π
            │ (I−γPπ)⁻¹ rπ  │
            └───────────────┘
                    ▲ 对比
                    │
   日志数据集 (TabularMDP.generate_dataset)
                    │  仅此数据可见
        ┌───────────┼───────────────┐
        ▼           ▼               ▼
    FQI / CFQI    FQE             WIS
   (控制/改进)   (评估给定π)     (重要性采样)
        │           │               │
        └───────────┴───────────────┘
                    ▼
            eval/comparison.py
        Kendall τ + @1 选择正确率
```

## 2. 依赖方向（单向，无环）

```
cli → pipeline → { algos, ope, envs, eval } → core
                                  algos/ope ──┘
```

- `core/`：纯数据 + numpy，不依赖任何上层；
- `envs/solver.py`：精确求解，只依赖 `core.types`；
- `algos/` 与 `ope/`：依赖 `core.features` 做 one-hot 编码 + `core.types`；
- `pipeline/`：编排一切，产 `EnvReport`；
- `cli.py` / `examples/`：入口。

## 3. 模块细节

### 3.1 `envs/specs.py` — TabularMDP

- `P: (S,A,S)` 转移概率，`R: (S,A)` 期望奖励，`mu` 起始分布，`gamma` 折扣；
- `generate_dataset(policy, n_episodes, max_steps, rng)` → `TransitionDataset`，
  记录每一步的 `behavior_probs`（重要性采样必需）与 `episode_ids`（WIS 分组）；
- 行为策略 `make_behaviour_policy` 用 softmax 保证**全支撑**（每动作概率 > 0），
  这是 WIS 良定义的先决条件。

### 3.2 `envs/solver.py` — 精确求解器（金标准）

```python
P_pi, r_pi = Σ_a π[a] P[:,a,:], Σ_a π[a] R[:,a]
V = solve(I − γ P_pi, r_pi)          # np.linalg.solve
J(π) = mu · V
```

最优策略由值迭代得到（greedy on exact Q）。

### 3.3 `algos/fqi.py` — Fitted Q-Iteration（控制旗舰）

每次迭代用监督回归拟合 Bellman **greedy** 目标：

```
y_i = r_i + γ · max_a Q_k(s'_i, a) · (1 − done_i)
```

回归器默认 `RandomForestRegressor(n_estimators=64)`。收敛后：

```
J^FQI = mean_{s0 ∈ D_init} max_a Q(s0, a)
```

应随数据量 / 迭代次数逼近精确最优值 `J*`。

### 3.4 `algos/cfqi.py` — Conservative FQI（悲观旗舰）

复用 FQI 训练好的同一棵随机森林（保证不变量），在贪婪动作处减去集成不确定性
（树间预测标准差）：

```
J^CFQI = mean_{s0} [ max_a Q(s0,a) − λ · u(s0, a*) ] ,  u ≥ 0
```

⇒ `J^CFQI ≤ J^FQI`（惩罚非负），这就是悲观离线 RL（CQL 一族）的核心原则，
也是测试套件里硬编码验证的硬不变量。

### 3.5 `ope/fqe.py` — Fitted Q Evaluation（OPE #1）

对**给定目标策略** `π_e` 跑 Bellman **评估**迭代：

```
y_i = r_i + γ · Σ_a π_e(a|s'_i) Q_k(s'_i, a) · (1 − done_i)
```

与 FQI 的区别仅在目标是「策略条件期望」而非「max」。最终：

```
J^FQE = mean_{s0} Σ_a π_e(a|s0) Q_K(s0, a)
```

**仅使用日志数据**，永不直接访问模型。

### 3.6 `ope/wis.py` — Weighted Importance Sampling（OPE #2）

轨迹级重要性加权：

```
w_i = Π_t π_e(a_t|s_t) / π_b(a_t|s_t) ,  G_i = Σ_t γ^t r_t
J^WIS = Σ_i w_i G_i / Σ_i w_i
```

无偏但高方差 —— 基准同时报告其估计与精确误差，让权衡可见。

### 3.7 `eval/comparison.py` — 排序保真度

- `kendall_tau(估计排序, 真值排序)`：估计器是否复现精确排序；
- `selection_correct`：按估计挑出的「最优策略」是否就是精确最优；
- 这是 OfflineForge 真正的评判标准——**不是单点误差，而是排序是否可信**。

## 4. 候选策略设计（为何排除纯 greedy 最优）

WIS 要求目标策略相对行为策略**绝对连续**（支撑不脱节）。纯 greedy 最优策略在其
行为策略探索到的动作上概率为 0，会使整条轨迹的重要性权重归零。因此基准只评估
**保留全支撑**的近最优策略（ε-greedy / 与行为策略混合），纯 greedy 最优仅作为
`optimal_value` 的精确参考值出现。

## 5. 复现

```bash
python -m offlineforge.cli --seed 0 --episodes 400 --out benchmark.json
```

结果含每个环境的 `fqi_abs_err / cfqi_abs_err`，每候选策略的 `fqe_abs_err / wis_abs_err`，
以及交叉环境的 `kendall_tau`、`selection_correct_*_rate`、`cfqi_le_fqi` 不变量。

## 6. 已修复的关键 Bug（踩坑记录）

| 坑 | 根因 | 修法 |
|----|------|------|
| WIS 对贪心候选策略分母归零（`OPEError`） | 确定性目标策略对行为策略无全支撑，整条轨迹权重 0 | 候选"最优"策略改为 ε-贪婪（全支撑）；确定性目标仍抛 `OPEError` |
| `J^CFQI ≤ J^FQI` 不变量被破坏 | CFQI 独立重训随机森林，两模型发散 | CFQI 通过 `base=` 复用 FQI 拟合模型，惩罚非负 ⇒ 硬下界 |
| `joblib` 崩溃 `concurrent send_bytes` | 受限 Windows 下 `n_jobs=-1` 触发 | `n_jobs` 固定为 1 |
| 目标态奖励从未兑现 | `+1` 挂在 `R[goal, :]`（从终止态出发行走），永不触发，且终止态奖励被强制归零 | 改为在进入目标的转移 `R[s,a]` 上结算 |
| Kendall τ 常数估计给出 −1 | 平局被稳定排序赋成不同秩，制造伪 discordance | 平局取平均秩 ⇒ τ=0（τ-b 正确行为） |
| FQE 需 ~40–60 次迭代才收敛 | 自举回归欠迭代则偏差大（err 0.5→0.03 随迭代下降） | 默认 `fqe_iterations=40`，代价与精度权衡后压到 ~5 分钟可跑完 |
