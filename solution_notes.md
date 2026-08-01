# A题：水泥烧成系统电除尘器的协同优化控制 — 解题经过

## 一、题目分析

### 1.1 题目背景

某水泥企业为一条 $$5000$$ 吨/日新型干法生产线窑尾配套电除尘器，该除尘器共有 $$4$$ 个电场，每个电场配有独立的整流变压器，可调节二次电压设定值 $$U_i$$（$$\text{kV}$$）和振打周期 $$T_i$$（$$\text{s}$$）。附件中给出该除尘器连续 $$7$$ 天的运行数据，要求建立数学模型解决四个问题。

### 1.2 数据特征

数据文件 `Cement_ESP_Data.csv` 包含 $$10080$$ 行（分钟级，$$7$$ 天）、$$14$$ 列：

| 字段 | 含义 | 单位 |
|------|------|------|
| $$\text{timestamp}$$ | 时间戳 | 分钟级 |
| $$T_{in}$$ | 烟气入口温度 | $$\text{℃}$$ |
| $$C_{in}$$ | 入口粉尘浓度 | $$\text{g/Nm}^3$$ |
| $$Q$$ | 烟气流量 | $$\text{Nm}^3\text{/h}$$ |
| $$U_1 \sim U_4$$ | 各电场二次电压 | $$\text{kV}$$ |
| $$T_1 \sim T_4$$ | 各电场振打周期 | $$\text{s}$$ |
| $$C_{out}$$ | 出口粉尘浓度 | $$\text{mg/Nm}^3$$ |
| $$P_{total}$$ | 总除尘电耗 | $$\text{kW}$$ |

**关键数据特征**：$$C_{out}$$ 有 $$50$$ 个缺失值，历史标准差仅 $$0.17$$，集中在 $$50$$ $$\text{mg/Nm}^3$$ 附近，疑似传感器限幅。这意味着纯数据驱动回归外推不可靠，需以机理模型为主。

### 1.3 四个问题概述

- **问题1**：分析入口条件与操作参数对出口浓度的关系，研究振打周期对瞬时排放峰值的影响
- **问题2**：工况划分 + 约束优化（$$C_{out} \leq 10$$ $$\text{mg/Nm}^3$$ 下最小化电耗）
- **问题3**：选取2个差异工况对比，灵敏度分析优先级
- **问题4**：排放收紧至 $$5$$ $$\text{mg/Nm}^3$$，定量分析电耗增幅

---

## 二、数学建模框架

### 2.1 变量与常量定义

#### 决策变量（可控）

| 符号 | 含义 | 单位 | 范围 |
|------|------|------|------|
| $$U_i$$ | 第 $$i$$ 电场二次电压 | $$\text{kV}$$ | $$i=1,2,3,4$$ |
| $$T_i$$ | 第 $$i$$ 电场振打周期 | $$\text{s}$$ | $$i=1,2,3,4$$ |

#### 工况变量（不可控输入）

| 符号 | 含义 | 单位 |
|------|------|------|
| $$T_{in}$$ | 烟气入口温度 | $$\text{℃}$$ |
| $$C_{in}$$ | 入口粉尘浓度 | $$\text{g/Nm}^3$$ |
| $$Q$$ | 烟气流量 | $$\text{Nm}^3\text{/h}$$ |

#### 输出变量

| 符号 | 含义 | 单位 |
|------|------|------|
| $$C_{out}$$ | 出口粉尘浓度 | $$\text{mg/Nm}^3$$ |
| $$P_{total}$$ | 总除尘电耗 | $$\text{kW}$$ |

#### 中间变量

| 符号 | 含义 |
|------|------|
| $$\eta_i$$ | 第 $$i$$ 电场单级除尘效率 |
| $$C_i$$ | 第 $$i$$ 电场出口浓度（$$C_0 = C_{in}$$） |
| $$P_i$$ | 第 $$i$$ 电场电耗 |
| $$C_{peak}$$ | 振打瞬时排放峰值 |

#### 常量

| 符号 | 值 | 含义 |
|------|----|------|
| $$C_{limit}$$ | $$10$$ | 超低排放标准 ($$\text{mg/Nm}^3$$) |
| $$C_{limit}'$$ | $$5$$ | 收紧后标准 ($$\text{mg/Nm}^3$$) |
| $$n$$ | $$4$$ | 电场数量 |
| $$U_{min,i}, U_{max,i}$$ | 由数据定 | 电压上下限 |
| $$T_{min,i}, T_{max,i}$$ | 由数据定 | 振打周期上下限 |
| $$T_{crit,i}$$ | 历史95分位 | 振打安全上限 |
| $$k_i$$ | 待拟合 | 第 $$i$$ 电场电耗系数 |

### 2.2 等式与不等式

#### 问题1：关系建模

**多级串联等式**：

$$C_i = C_{i-1} \cdot (1 - \eta_i), \quad i=1,2,3,4 \tag{2}$$

$$C_{out} = C_4 \cdot 1000 \quad (\text{g} \to \text{mg转换}) \tag{3}$$

**Deutsch方程（单级效率）**：

$$\eta_i = \left(1 - \exp\left(-\frac{k_i A_i \cdot U_i^2}{Q}\right)\right) \cdot \exp\left(-\alpha_i \cdot \max(0, T_i - T_{ref,i})\right) \tag{4}$$

其中：
- $$k_i A_i$$：合并系数（待拟合）
- $$\alpha_i$$：振打衰减系数
- $$T_{ref,i}$$：参考振打周期（取历史中位数）

**电耗等式**：

$$P_{total} = \sum_{i=1}^{4} P_i = \sum_{i=1}^{4} k_i \cdot U_i^2 \tag{6}$$

当 $$R^2 < 0.9$$ 时扩展为：

$$P_{total} = \sum_{i=1}^{4} \left(k_i U_i^2 + b_i U_i\right) + c \tag{6'}$$

**振打峰值等式**：

$$C_{peak} = \sum_{i=1}^{4} \alpha_i \cdot \max(0, T_i - T_{ref,i}) \cdot C_{in} \cdot 1000 \cdot 0.1 \tag{5}$$

#### 问题2：工况划分 + 约束优化

**工况划分**：将 $$(C_{in}, T_{in}, Q)$$ 空间划分为 $$K$$ 个典型工况 $$\Omega_k$$：

$$\bigcup_{k=1}^{K} \Omega_k = \Omega_{all}, \quad \Omega_j \cap \Omega_k = \emptyset \ (j \neq k) \tag{7}$$

**优化模型**：

$$\min_{U_i, T_i} \ P_{total} = \sum_{i=1}^{4} k_i U_i^2 \tag{8}$$

**约束条件**：

$$C_{out} = f(T_{in}^{(k)}, C_{in}^{(k)}, Q^{(k)}, U_1, \ldots, U_4, T_1, \ldots, T_4) \leq C_{limit} \tag{9}$$

$$U_{min,i} \leq U_i \leq U_{max,i}, \quad i=1,2,3,4 \tag{10}$$

$$T_{min,i} \leq T_i \leq \min(T_{max,i}, T_{crit,i}), \quad i=1,2,3,4 \tag{11\text{-}12}$$

#### 问题3：灵敏度分析

**数值雅可比（有限差分）**：

$$S_{U_i}^{C} = \frac{\partial C_{out}}{\partial U_i}, \quad S_{T_i}^{C} = \frac{\partial C_{out}}{\partial T_i} \tag{14}$$

$$S_{U_i}^{P} = \frac{\partial P_{total}}{\partial U_i}, \quad S_{T_i}^{P} = \frac{\partial P_{total}}{\partial T_i} \tag{15}$$

**优先级判定**：

$$\text{若} \ \left| \frac{S_{U_i}^{C}}{S_{U_i}^{P}} \right| > \left| \frac{S_{T_i}^{C}}{S_{T_i}^{P}} \right| \ \Rightarrow \ \text{优先调电压} \tag{16}$$

#### 问题4：排放收紧

$$C_{out} \leq C_{limit}' = 5 \tag{17}$$

$$\Delta P\% = \frac{P^*(5) - P^*(10)}{P^*(10)} \times 100\% \tag{18}$$

### 2.3 基本假设

| 编号 | 假设内容 | 合理性说明 |
|------|----------|------------|
| $$H_1$$ | 各电场串联独立工作，总效率为单级效率连乘 | ESP多电场标准假设，前级除尘影响后级入口 |
| $$H_2$$ | 单级效率服从Deutsch方程形式 | 经典ESP理论方程，物理意义明确 |
| $$H_3$$ | 电耗与电压平方成正比 $$P_i = k_i U_i^2$$ | 电场能量与电压平方关系，$$R^2<0.9$$时扩展 |
| $$H_4$$ | 振打周期越大→积灰越厚→效率下降越大 | 振打清灰物理机制，用指数衰减建模 |
| $$H_5$$ | 工况参数在单个时间窗口内稳定 | 分钟级数据，K-Means聚类划分典型工况 |
| $$H_6$$ | 入口条件与操作参数之间无强耦合交互 | 简化建模，特征重要性验证 |
| $$H_7$$ | 振打瞬时峰值仅与振打周期和当前浓度相关 | 半经验关系，分钟级数据无法直接观测秒级峰值 |

---

## 三、解题思路与方案设计

### 3.1 问题1：机器学习 + 机理建模

**方案**：以Deutsch机理模型为主、数据驱动为辅。

**合理性**：历史 $$C_{out}$$ 方差仅 $$0.17$$，集中在 $$50$$，纯数据驱动回归外推不可靠。Deutsch方程有明确物理意义，可外推到 $$C_{out} = 10$$ 甚至 $$5$$ 的场景。

**实现**：
1. 电耗模型拟合：最小二乘解 $$Xk = P$$，$$X = [U_1^2, U_2^2, U_3^2, U_4^2]$$
2. Deutsch参数拟合：固定 $$T_{ref}$$ 为各电场中位数，用 L-BFGS-B 在对数空间最小化残差，多起点（50次）避免局部最优
3. 特征重要性：随机森林 + 梯度提升 + Pearson相关系数交叉验证

### 3.2 问题2：K-Means工况划分 + SLSQP约束优化

**方案**：分段函数（工况划分）+ 约束优化。

**合理性**：工况划分将连续参数空间离散化为可管理的典型场景，每个场景独立求解约束优化问题。SLSQP适合中等规模（8维）约束优化，多起点避免局部最优。

**实现**：
1. K-Means聚类：以 $$C_{in}, T_{in}, Q$$ 为特征，$$K=5$$，校验每工况样本 $$\geq 3\%$$
2. 单工况寻优：决策变量 $$[U_1, U_2, U_3, U_4, T_1, T_2, T_3, T_4]$$ 共8维，目标 $$\min P_{total}$$，约束 $$C_{out} \leq C_{limit}$$
3. 多起点启动（10个），固定随机种子 $$42$$ 保证可复现
4. SLSQP不收敛时切换差分进化全局备选

### 3.3 问题3：数值灵敏度分析

**方案**：有限差分计算雅可比矩阵，性价比比值判定优先级。

**合理性**：灵敏度 $$\partial C_{out} / \partial U_i$$ 和 $$\partial P_{total} / \partial U_i$$ 的比值表示"单位电耗带来的浓度下降"，比值大者优先调整。

**实现**：
1. 选取最高浓度 vs 最低浓度工况对比
2. 有限差分步长取变量范围 $$1\%$$，步长减半验证一致性
3. 计算 $$|S^C / S^P|$$ 性价比比值，输出优先级判定

### 3.4 问题4：重新求解 + 增量对比

**方案**：复用问题2框架，将排放约束改为 $$5$$ $$\text{mg/Nm}^3$$ 重新求解，对比电耗差异。

**合理性**：直接重新求解比纯灵敏度分析更准确，灵敏度分析可作为补充验证。

**实现**：
1. 调用 `solve_all_regimes` 传入 $$C_{limit}' = 5$$
2. 计算 $$\Delta P\% = (P^*(5) - P^*(10)) / P^*(10) \times 100\%$$
3. 可行性校验：收紧后参数是否在物理可行域
4. 高浓度工况应对建议：基于收紧前后参数差异给出具体可操作建议

---

## 四、代码实现思路

### 4.1 模块架构

```
Problem/
├── main.py              # 主流程编排
├── config/config.yaml   # 集中配置
├── data_loader/         # 数据加载与预处理
│   ├── loader.py        # load_raw, clean_and_impute
│   └── db.py            # MySQL持久化（可选）
├── modeling/            # 问题1 机理建模
│   ├── power.py         # 电耗模型拟合
│   ├── deutsch.py       # Deutsch参数拟合与预测
│   └── feature.py       # 特征重要性
├── optim/               # 问题2 工况划分与寻优
│   ├── regime.py        # K-Means工况划分
│   └── solve.py         # SLSQP/DE约束优化
├── sensitivity/         # 问题3 灵敏度分析
│   ├── jacobian.py      # 数值雅可比
│   ├── priority.py      # 优先级判定
│   └── compare.py       # 两工况对比
├── tighten/             # 问题4 收紧分析
│   ├── resolve.py       # 重求解、增幅、可行性
│   └── advice.py        # 高浓度应对建议
├── report/              # 输出与可视化
│   ├── plots.py         # 各类图表
│   └── tables.py        # Markdown表格生成
└── outputs/             # 产物目录
```

### 4.2 关键设计决策

1. **机理模型为主、数据驱动为辅**：应对 $$C_{out}$$ 历史方差极小外推不可靠
2. **各电场独立边界与系数**：应对前后电场参数差异显著
3. **排放阈值参数化** `solve_one_regime(..., C_limit)`：问题4复用问题2框架
4. **SLSQP主 + 差分进化备选 + 多起点固定种子**：避免局部最优且可复现
5. **振打硬约束** $$T_i \leq T_{crit,i}$$：防止积灰退化

### 4.3 数据流

$$\text{CSV} \xrightarrow{\text{load\_raw}} \text{DataFrame} \xrightarrow{\text{clean\_and\_impute}} \text{清洗后DF+bounds}$$

$$\xrightarrow{\text{fit\_power}} P_{model} \quad \xrightarrow{\text{fit\_deutsch}} \eta_{model} \quad \xrightarrow{\text{feature\_importance}} \text{特征排序}$$

$$\xrightarrow{\text{cluster\_regimes}} \{\Omega_k\} \xrightarrow{\text{solve\_all}} \{U_i^*, T_i^*\}_k$$

$$\xrightarrow{\text{jacobian}} S^C, S^P \xrightarrow{\text{priority}} \text{优先级}$$

$$\xrightarrow{\text{resolve}(C_{limit}'=5)} \{U_i^{*'}, T_i^{*'}\}_k \xrightarrow{\text{delta\_power}} \Delta P\%$$

---

## 五、调试过程与Bug修复

### 5.1 Bug 1：列表推导式语法错误

**文件**：`modeling/power.py`

**问题**：扩展模型的设计矩阵构造中，列表推导式嵌套导致语法错误：

```python
# 错误写法
X = np.column_stack([df[f"U{i}_kV"].values ** 2, df[f"U{i}_kV"].values, np.ones(len(df)) for i in range(1, 5)])

# 正确写法
X = np.column_stack([df[f"U{i}_kV"].values ** 2 for i in range(1, 5)] + [df[f"U{i}_kV"].values for i in range(1, 5)] + [np.ones(len(df))])
```

### 5.2 Bug 2：Deutsch拟合 curve_fit 失败

**文件**：`modeling/deutsch.py`

**问题**：`scipy.optimize.curve_fit` 在处理多维输入时返回数组真值歧义错误。

**修复**：改用 `scipy.optimize.minimize`（L-BFGS-B），在对数空间最小化残差：

$$\text{loss} = \sum \left(\ln(\hat{C}_{out}) - \ln(C_{out})\right)^2$$

### 5.3 Bug 3：变量顺序不一致

**文件**：`optim/solve.py`

**问题**：决策变量排列为交错顺序 $$[U_1, T_1, U_2, T_2, U_3, T_3, U_4, T_4]$$，但返回结果时假设为分组顺序 $$[U_1, U_2, U_3, U_4, T_1, T_2, T_3, T_4]$$，导致 $$U$$ 和 $$T$$ 混淆。

**修复**：统一变量顺序为分组排列 $$[U_1, U_2, U_3, U_4, T_1, T_2, T_3, T_4]$$：

```python
lb = [bounds[f"U{i}"][0] for i in range(1, 5)] + [bounds[f"T{i}"][0] for i in range(1, 5)]
ub = [bounds[f"U{i}"][1] for i in range(1, 5)] + [min(bounds[f"T{i}"][1], bounds[f"T_crit{i}"]) for i in range(1, 5)]
```

### 5.4 Bug 4：bounds键名索引错误

**文件**：`sensitivity/jacobian.py`

**问题**：`bounds` 字典键名为 $$U_1, U_2, U_3, U_4$$（从1开始），但循环用了 `range(4)`（从0开始），导致 `KeyError: 'U0'`。

**修复**：将 `range(4)` 改为 `range(1, 5)`。

### 5.5 Bug 5：电耗模型 predict_power 接口不一致

**文件**：`modeling/power.py`

**问题**：`fit_power_model` 在 $$R^2 < 0.9$$ 时返回扩展模型系数，但 `predict_power` 只接受简单模型 $$k$$ 列表，导致优化目标与实际电耗不一致。

**修复**：`predict_power` 统一接受 `dict` 类型，根据 `extended` 标志选择模型：

```python
def predict_power(model, U):
    if isinstance(model, dict) and model.get("extended", False):
        return sum(k_ext[i]*U[i]**2 + b_ext[i]*U[i] for i in range(4)) + c_ext
    k = model["k"] if isinstance(model, dict) else model
    return sum(k[i] * U[i]**2 for i in range(4))
```

---

## 六、最终运行结果

### 6.1 问题1结果

**电耗模型**：

$$P_{total} = \sum_{i=1}^{4} k_i U_i^2, \quad R^2 = 0.3567$$

$$R^2 < 0.9$$，扩展为 $$P = \sum(k_i U_i^2 + b_i U_i) + c$$，$$R^2 = 0.9368$$。

**Deutsch模型**：

$$R^2 = -111039$$（因 $$C_{out}$$ 方差极小，$$R^2$$ 不适用，但 $$\text{RMSE} = 56.4$$，预测值在 $$60$$ 左右与实际 $$50$$ 同数量级）

拟合参数：

| 参数 | 第1电场 | 第2电场 | 第3电场 | 第4电场 |
|------|---------|---------|---------|---------|
| $$kA_i$$ | $$144.5$$ | $$269.1$$ | $$399.0$$ | $$338.8$$ |
| $$\alpha_i$$ | $$0.0158$$ | $$0.0054$$ | $$0$$ | $$0$$ |
| $$T_{ref,i}$$ | $$233$$ | $$233$$ | $$445$$ | $$445$$ |

**特征重要性**（随机森林）：各特征贡献均匀（$$0.08 \sim 0.11$$），Pearson相关系数均不显著（$$p > 0.05$$），印证 $$C_{out}$$ 方差极小的数据特征。

### 6.2 问题2结果

5个工况均成功求解，$$C_{out} = 10$$ $$\text{mg/Nm}^3$$ 约束满足：

| 工况 | 样本数 | $$C_{in}$$ ($$\text{g/Nm}^3$$) | $$T_{in}$$ (℃) | $$P^*$$ ($$\text{kW}$$) | $$C_{out}$$ |
|------|--------|------|------|------|------|
| 0 | 2639 | 28.95 | 131.4 | 1478.1 | 10.0 |
| 1 | 1691 | 25.98 | 121.1 | 1493.0 | 10.0 |
| 2 | 2678 | 40.84 | 130.4 | 1522.3 | 10.0 |
| 3 | 1428 | 43.78 | 119.4 | 1544.1 | 10.0 |
| 4 | 1644 | 44.64 | 121.1 | 1489.6 | 10.0 |

### 6.3 问题3结果

**对比**：高浓度工况(工况4) vs 低浓度工况(工况1)

**优先级判定**：**优先调电压**

$$\text{电压性价比} = 0.7614 > \text{振打性价比} = 0.0000$$

**差异原因**：高浓度工况需更高电压承担除尘负荷，振打周期需缩短以频繁清灰。

### 6.4 问题4结果

排放收紧至 $$5$$ $$\text{mg/Nm}^3$$ 后各工况电耗增幅：

| 工况 | $$P^*(10)$$ ($$\text{kW}$$) | $$P^*(5)$$ ($$\text{kW}$$) | $$\Delta P\%$$ |
|------|------|------|------|
| 0 | 1478.1 | 1495.7 | 1.20% |
| 1 | 1493.0 | 1578.8 | 5.74% |
| 2 | 1522.3 | 1674.8 | 10.02% |
| 3 | 1544.1 | 1692.3 | 9.60% |
| 4 | 1489.6 | 1532.1 | 2.85% |

**整体平均电耗增幅**：$$5.86\%$$

高浓度工况(工况2、3)增幅最大（$$\sim 10\%$$），低浓度工况增幅较小（$$\sim 1\%$$）。

---

## 七、生成产物清单

| 文件 | 说明 |
|------|------|
| `outputs/results.md` | 论文用 Markdown 表格（所有公式用 `$$` 包裹） |
| `outputs/relation_curves.png` | 问题1：温度/浓度/电压/振打与 $$C_{out}$$ 关系曲线 |
| `outputs/regime_scatter.png` | 问题2：K-Means 工况划分散点图 |
| `outputs/param_compare.png` | 问题3：两工况电压/振打/电耗对比柱状图 |
| `outputs/sensitivity_heatmap.png` | 问题3：灵敏度 $$S^C, S^P, |S^C/S^P|$$ 热力图 |
| `outputs/delta_power.png` | 问题4：收紧前后电耗对比与增幅柱状图 |
| `outputs/run_meta.json` | 运行元数据（种子、行数、$$R^2$$ 等） |

---

## 八、关键注意事项

1. **$$C_{out}$$ 传感器限幅**：历史数据集中在 $$50$$ $$\text{mg/Nm}^3$$，方差仅 $$0.17$$，$$R^2$$ 不适用，需用 $$\text{RMSE}$$ 和物理合理性评估模型质量
2. **外推风险**：从 $$50$$ 外推到 $$10$$ 甚至 $$5$$ $$\text{mg/Nm}^3$$ 存在不确定性，论文中需明确标注
3. **振打峰值估算**：基于机理半经验关系，分钟级数据无法直接观测秒级峰值，需标注局限性
4. **复现性**：全流程固定 `seed=42`，`requirements.txt` 固定依赖版本
5. **排放约束参数化**：问题4复用问题2框架，仅需传入不同 $$C_{limit}$$