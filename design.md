# 一、需求与存量功能关系分析

> 说明：本项目为 2026 年 XJTU 校赛 A 题新建工程，仓库初始仅有 `README.md`，无任何存量业务代码。因此本章节将"存量功能"理解为"数据文件 `Cement_ESP_Data.csv` 所蕴含的历史运行信息以及电除尘领域经典机理公式（Deutsch 方程、$P\propto U^2$）"这两类可复用资产，并据此分析四个问题需求与这些资产的关系，作为增量设计的依据。

## 1.1 需求功能与存量功能对比

### 1.1.1 已实现（可直接复用）功能

| 需求功能 | 存量功能 | 代码位置 / 数据来源 | 匹配度 |
|---------|---------|---------|--------|
| 分钟级历史运行数据加载与基础统计 | `Cement_ESP_Data.csv` 10080 条 × 14 列完整记录 | `question/.../A/Cement_ESP_Data.csv` | 100% |
| 电耗—电压平方关系 $P_{total}=\sum k_i U_i^2$ 的系数拟合 | 历史含 $U_1\sim U_4$ 与 $P_{total}$ 字段，可直接最小二乘拟合 $k_i$ | `Cement_ESP_Data.csv` 第 5–8、14 列 | 100% |
| 单级除尘效率 Deutsch 结构 $\eta_i=1-\exp(-k_i U_i^2 A_i/Q)$ | 电除尘经典理论公式（领域知识） | 领域机理（非代码） | 75%（结构已知，系数需拟合） |
| 各电场电压/振打边界确定 | 历史各电场 min/max 可直接作为 $U_{min},U_{max},T_{min},T_{max}$ | `Cement_ESP_Data.csv` 第 5–12 列 | 100% |

### 1.1.2 需要扩展（在存量基础上改造）的功能

| 需求功能 | 存量功能 | 差异说明 | 扩展方向 |
|---------|---------|---------|---------|
| 出口浓度预测 $C_{out}=f(T_{in},C_{in},Q,U_{1\sim4},T_{1\sim4})$ | 历史 $C_{out}$ 仅集中在 $49.8\sim50.0$，标准差 $0.17$，疑似传感器量程限幅 | 因变量方差极小，纯数据驱动回归学不到"参数如何降低 $C_{out}$"的信息；需从 $50$ 外推到 $\leq10$ 甚至 $\leq5$ | 以 Deutsch 机理模型为主框架，从历史数据拟合系数 $k_i A_i$ 与振打衰减系数 $\alpha_i$，靠物理单调性保证外推合理；随机森林仅作特征重要性交叉验证 |
| 振打瞬时排放峰值 $C_{peak}=g(T_{1\sim4},C_{in})$ | 数据为分钟级，无法直接观测秒级振打峰值 | 时间分辨率不足，峰值不可直接测量 | 用机理估算（脱落粉尘量 $\propto$ 积灰厚度 $\propto$ 振打周期）建立半经验关系，并在论文中明确标注为估算 |
| 工况划分 | 历史数据具备 $C_{in}\in[18,72]$、$T_{in}\in[111.7,158.2]$ 宽波动 | 存量只是原始点云，未做分群 | 以 $C_{in},T_{in}$（辅以 $Q$）为特征做 K-Means 聚类划分为 $4\sim6$ 个典型工况 |
| 带约束非线性寻优（8 维） | 无存量优化器 | 需新建优化求解层 | 采用 `scipy.optimize.minimize(SLSQP)` 为主、PSO/GA 全局备选验证，多起点启动 |

### 1.1.3 需要新增的功能或接口

按业务模块分组如下：

**A. 数据层（`data_loader`）**
- `load_raw(path) -> DataFrame`：读取 CSV、解析时间戳、报告缺失值（$C_{out}$ 有 50 个缺失）。
- `clean_and_impute(df) -> DataFrame`：缺失值插补（时间线性插值）、异常值标记、各电场边界统计。
- `to_mysql(df, table)` / `from_mysql(table) -> DataFrame`：持久化到 MySQL（用户偏好），便于复算。

**B. 问题 1 机理建模层（`modeling`）**
- `fit_power_model(df) -> dict`：拟合 $k_i$，返回系数与 $R^2$。
- `fit_deutsch_params(df) -> dict`：拟合各电场 $k_i A_i$ 与振打衰减 $\alpha_i$。
- `predict_cout(params, Tin, Cin, Q, U, T) -> float`：机理模型预测 $C_{out}$。
- `predict_peak(params, T, Cin) -> float`：振打峰值半经验估算。
- `feature_importance(df) -> dict`：随机森林/梯度提升特征重要性交叉验证。

**C. 问题 2 工况划分与寻优层（`optim`）**
- `cluster_regimes(df, k) -> dict`：K-Means 工况划分，返回各工况边界、样本数、特征均值方差。
- `solve_one_regime(regime, model, C_limit) -> dict`：单工况 SLSQP 寻优，返回最优 $U,T$、$P_{total}$、收敛状态。
- `solve_all_regimes(regimes, model, C_limit) -> dict`：批量寻优 + PSO/GA 备选验证。

**D. 问题 3 灵敏度分析层（`sensitivity`）**
- `numeric_jacobian(model, x0) -> dict`：有限差分计算 $S^C,S^P$。
- `priority_rule(sens) -> dict`：性价比比值 $|S^C/S^P|$ 排序与优先级判定。
- `compare_two_regimes(rA, rB) -> dict`：两差异工况参数对比与差异原因。

**E. 问题 4 收紧分析层（`tighten`）**
- `resolve_under_limit(regimes, model, C_limit_new) -> dict`：以 $C_{limit}'=5$ 重求解。
- `delta_power(P10, P5) -> dict`：电耗增加百分比 $\Delta P\%$。
- `feasibility_check(sol) -> dict`：物理可行域校验。
- `high_conc_advice(regime, sol10, sol5) -> str`：高浓度工况应对建议。

**F. 输出与可视化层（`report`）**
- `plot_*` 系列：关系曲线、工况散点、参数对比柱状、灵敏度热力、电耗增幅柱状。
- `to_markdown_tables(results) -> str`：生成论文用 Markdown 表格。

模块依赖关系：`data_loader` ← `modeling` ← `optim` ← `sensitivity` / `tighten` ← `report`，呈单向流水线。

## 1.2 存量功能详细分析

> 本节深入解读 1.1.1 中"已实现/可直接复用"资产的关键属性，作为增量设计的约束输入。

### 1.2.1 历史数据接口契约

- **入参**：CSV 文件路径，UTF-8 编码，14 列固定顺序。
- **出参**：`pandas.DataFrame`，10080 行，dtype 由 `pandas` 自动推断后强制转换（时间戳 `datetime64[ns]`，其余 `float64`）。
- **异常**：缺失值仅出现在 $C_{out}$（50 个），其余字段无缺失；时间戳连续无断点。
- **副作用**：可选写入 MySQL 表 `esp_raw`（用户偏好数据库持久化）。
- **业务规则**：7 天 × 1440 分钟 = 10080 条；时间戳单调递增、间隔 1 分钟。
- **约束**：内存占用约 $10080\times14\times8\approx1.1$ MB，可全量载入；无需分块。

### 1.2.2 电耗模型 $P_{total}=\sum_{i=1}^{4} k_i U_i^2$ 的可验证性

- **接口契约**：输入历史 $U_{1\sim4}$ 矩阵（$10080\times4$）与 $P_{total}$ 向量，输出 $k_i$（4 个）与拟合 $R^2$。
- **业务规则**：构造设计矩阵 $X=[U_1^2,U_2^2,U_3^2,U_4^2]$，解线性最小二乘 $X k = P_{total}$。
- **扩展点**：若 $R^2$ 偏低，可扩展为 $P_i=k_i U_i^2 + b_i U_i + c_i$（加入线性项与常数项）。
- **约束**：$k_i>0$（物理要求电耗正比于电压平方），拟合后需强制非负；历史 $P_{total}\in[1479.9,2087.3]$、$U$ 范围见 spec 3.1，拟合条件数需检查。

### 1.2.3 Deutsch 效率结构的约束

- **接口契约**：输入历史 $U_i,Q,C_{in},C_{out}$，输出各电场合并系数 $k_i A_i$ 与振打衰减 $\alpha_i$。
- **业务规则**：$\eta_i=1-\exp(-k_i U_i^2 A_i/Q)$；振打衰减 $\eta_i(T_i)=\eta_{i,0}\exp(-\alpha_i(T_i-T_{ref}))$；总效率连乘 $C_{out}=C_{in}\cdot1000\cdot\prod(1-\eta_i)$。
- **约束（关键）**：
  1. 历史 $C_{out}\approx50$ 近乎恒定，拟合时因变量方差极小，参数估计不确定性大——**必须以物理单调性约束拟合**（效率随 $U$ 单调增、随 $T$ 单调减），否则外推不可靠。
  2. 前后电场参数差异显著（$U_{1,2}\approx58$kV vs $U_{3,4}\approx48$kV；$T_{1,2}\approx230$s vs $T_{3,4}\approx441$s），**各电场必须独立设边界与系数，不可混用统一范围**。
  3. 外推距离远（从 $50$ 到 $\leq5$ 是 10 倍降幅），需在报告中明确不确定性区间。

### 1.2.4 边界统计的约束

- **业务规则**：各电场 $U_{min},U_{max},T_{min},T_{max}$ 取历史该电场 min/max，并预留 $5\%$ 裕度供外推。
- **约束**：振打周期需额外设安全上限 $T_{crit,i}$（防极板积灰过厚），取历史 $T_i$ 的 $95$ 分位数或物理经验值；电压上限受变压器容量约束，外推不超过 $U_{max}\times1.1$。
