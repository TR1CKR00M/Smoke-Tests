# `diagnosis_new.py` 分析说明

## 1. 脚本用途

`diagnosis_new.py` 用于比较多个 MACE 势模型在含 Cr 替位缺陷的 MoS₂ 结构上的表现。它把分析分成三个相互补充的部分：

1. **预测误差**：比较模型在 300 K 训练数据子集和 1200 K 测试数据上的能量、原子力预测误差。
2. **缺陷区域响应**：用旋转不变的逐原子梯度乘激活分数，比较模型对 Cr 缺陷附近与远端基质的响应强度。
3. **结构特征与误差关联**：用 SOAP 描述符表示 1200 K 结构，以梯度提升树拟合能量误差，再用 TreeSHAP 统计高角动量 Cr 特征的归因占比。

脚本输出的是用于模型比较的汇总指标。逐原子梯度归因和 SHAP 是模型内部响应或拟合器归因分析，不能单独证明某个物理机制导致误差。

## 2. 输入文件与默认配置

默认情况下，脚本从当前工作目录读取：

- `defect_train_300K.xyz`：300 K 训练结构及参考能量、力。默认取最前面的 30 帧计算训练误差。
- `defect_test_1200K.xyz`：1200 K 测试结构及参考能量、力。全部帧用于测试误差和 SOAP/SHAP 分析。
- 四个模型文件：`mace_k128_L0_stagetwo.model`、`mace_k32_L0_stagetwo.model`、`mace_k128_L2_stagetwo.model`、`mace_k32_L2_stagetwo.model`。

XYZ 帧须包含 `REF_energy`（参考总能量）和 `REF_forces`（参考原子力），且元素种类、晶胞和周期性信息应与模型兼容。模型以 CPU 加载和运行；加载时会暂时将 TorchScript 子模块映射到 CPU，并在结束后恢复 `torch.jit.load`。

运行示例：

```bash
python scripts/analysis/diagnosis_new.py
```

指定数据、模型和训练帧数：

```bash
python scripts/analysis/diagnosis_new.py \
  --train data/processed/defect_train_300K.xyz \
  --test data/processed/defect_test_1200K.xyz \
  --models mace_k128_L0 mace_k32_L0 \
  --train-frames 30
```

`--models` 接受模型名前缀，脚本会自动从 `models/` 目录加载 `_stagetwo.model` 文件。可用 `python scripts/analysis/diagnosis_new.py --help` 查看参数。

## 3. 分析流程

### 3.1 读取结构并准备 SOAP 描述符

ASE 将 XYZ 文件读成原子结构列表。脚本对所有 1200 K 帧计算全局 SOAP 特征，参数为：

- 元素：Cr、Mo、S
- 截断半径 `r_cut=5.5 Å`
- 径向基组数 `n_max=6`
- 最大角动量 `l_max=3`
- `average="inner"`：将局部原子环境聚合为每帧的全局描述
- `periodic=True`

因此 SOAP 矩阵的一行对应一帧测试结构，一列对应一个结构特征。代码依照元素对、径向基组对和角动量通道构造特征名；只保留与 DScribe 实际输出列数相同的名称。

### 3.2 MACE 预测与误差

对每个模型，脚本分别评估最多 `--train-frames` 个训练帧和全部测试帧。每帧通过 MACE 的原子数据转换接口构造输入，并将浮点输入转换成模型权重的数据类型。

每帧的能量误差为总能量绝对误差按原子数归一化，并换算为 meV/atom：

\[
E_{\mathrm{err}} = \frac{|E_{\mathrm{pred}}-E_{\mathrm{ref}}|}{N}\times 1000
\]

每帧的力误差为所有原子、三个笛卡尔分量上的均方根误差，换算为 meV/Å：

\[
F_{\mathrm{RMSE}} = \sqrt{\operatorname{mean}\left[(\mathbf F_{\mathrm{pred}}-\mathbf F_{\mathrm{ref}})^2\right]}\times 1000
\]

输出列中的 `300K F-RMSE` 和 `1200K F-RMSE` 是相应帧级力 RMSE 的算术平均值。能量误差也逐帧计算，但表格不直接打印其平均值；1200 K 的逐帧能量误差会作为下一步 SHAP 分析的目标。

### 3.3 泛化差距（Overfit Ratio）

脚本用 1200 K 测试集平均力 RMSE 除以 300 K 训练子集平均力 RMSE：

\[
R_{\mathrm{overfit}} = \frac{\operatorname{mean}(F_{\mathrm{RMSE,1200K}})}{\operatorname{mean}(F_{\mathrm{RMSE,300K}})}
\]

比值大于 1 表示模型在高温测试集上的力误差高于训练子集；数值越大，两个数据集上的误差差距越大。由于 300 K 只使用训练文件开头的指定帧数，该指标依赖这些帧的顺序和代表性。它是训练/测试误差比，不是严格意义上经过独立验证的统计过拟合检验。

### 3.4 缺陷区域 Grad-CAM 选择性

脚本在模型最后一个 interaction block 上注册前向钩子，保存每个原子的输出激活 \(A_i\)。对预测总能量反向传播，得到每个原子的梯度 \(\nabla_{A_i}E\)。由于最后一层可能包含标量和高阶等变分量，不能把所有分量简单当作独立标量并跨原子平均绝对梯度。当前实现沿特征分量维对每个原子分别计算 L2 范数，定义逐原子梯度乘激活分数：

\[
\mathrm{CAM}_i=\left\|\nabla_{A_i} E\right\|_2\,\left\|A_i\right\|_2
\]

范数不跨原子计算。若特征按不可约表示块在旋转下正交变换，则特征及其梯度的 L2 范数对坐标旋转不变；合并所有块求整体范数也保持这一性质。此分数适用于混合等变特征，是逐原子的梯度范数乘激活范数归因，不是严格意义上按每个 irrep 分块的传统 Grad-CAM，也不是经典的 signed Grad-CAM + ReLU。它度量能量输出对该原子特征的局部敏感性与特征幅度的组合。

该公式变化可能改变 `S_def`，但“取绝对值必然使比值退化到 1”并不是普遍成立的数学结论；是否造成特定数据上的近 1 结果，需要在相同模型和帧上分别计算旧、新公式逐帧归因进行验证。Euler 齐次函数定理只有在特定齐次性条件成立时才可直接应用，MACE 最后一层的具体映射及交互依赖不能仅凭总能量可加性推断为该条件。

随后用 ASE 邻居列表以 3.5 Å 为截断距离划定区域：缺陷区包含所有 Cr 原子，以及距离任一 Cr 不超过 3.5 Å 的邻居；其余原子为 bulk 区。每帧的选择性定义为：

\[
S_{\mathrm{def}}=\frac{\operatorname{mean}(\mathrm{CAM}_{\mathrm{defect}})+10^{-8}}
{\operatorname{mean}(\mathrm{CAM}_{\mathrm{bulk}})+10^{-8}}
\]

表格中的 `Grad-CAM S_def` 是全部 1200 K 帧的该比值的算术平均。大于 1 表示最后一层逐原子归因分数在缺陷区平均更高；小于 1 则表示 bulk 区平均更高。它度量的是模型对能量预测的相对响应，不是模型准确率，也不表示 Cr 原子对能量的因果贡献。

### 3.5 SOAP、GBDT 与 TreeSHAP

对每个模型，脚本将 1200 K 每帧 SOAP 特征作为输入，将该模型对应帧的能量绝对误差（meV/atom）作为回归目标，拟合 `GradientBoostingRegressor`（50 棵树、最大深度 3、随机种子 42）。随后用 `shap.TreeExplainer` 计算训练样本上的 SHAP 值。

脚本取每个特征 SHAP 绝对值在测试帧上的平均值，表示该特征对 GBDT 误差拟合的平均归因幅度。它将元素对名称包含 `Cr` 且角动量通道为 `l2` 或 `l3` 的特征归为“Cr(l≥2)”组，并计算：

\[
P_{\mathrm{Cr},l\ge2}=100\times
\frac{\sum_{j\in\mathrm{Cr}(l\ge2)}\operatorname{mean}_i|\mathrm{SHAP}_{ij}|}
{\sum_j\operatorname{mean}_i|\mathrm{SHAP}_{ij}|+10^{-8}}
\]

输出的 `SHAP Cr(l>=2) Share` 是这部分特征在所有特征总绝对归因中的百分比。它回答的是“在该 GBDT 对该模型测试能量误差的拟合中，这组 SOAP 列占了多少平均 SHAP 绝对归因”，不是 MACE 模型自身的特征重要性，也不是高角动量物理效应的因果份额。SOAP 全局平均会压缩局部环境信息，SHAP 结果还会受特征相关性、数据规模和 GBDT 拟合质量影响。

## 4. 输出列解读

| 列名 | 含义 |
| --- | --- |
| `Model Name` | 被评估的模型名前缀 |
| `300K F-RMSE` | 300 K 训练子集的平均帧级力 RMSE，meV/Å |
| `1200K F-RMSE` | 1200 K 测试集的平均帧级力 RMSE，meV/Å |
| `Overfit Ratio` | 1200 K 与 300 K 平均力 RMSE 之比 |
| `Grad-CAM S_def` | 1200 K 测试集缺陷区相对 bulk 区的平均 CAM 比值 |
| `SHAP Cr(l>=2) Share` | GBDT/TreeSHAP 中 Cr 相关 l=2、3 SOAP 特征的归因占比 |

## 5. 结果比较时的注意事项

- 不同模型的误差指标可直接用于本脚本输入数据上的相对比较；测试集是 1200 K 高位移结构，主要反映该设定下的分布外表现。
- `Overfit Ratio` 的分母只来自训练集前若干帧，增减 `--train-frames` 或改变帧顺序会影响结果。
- Grad-CAM 的区域边界固定为 3.5 Å，SOAP 截断固定为 5.5 Å。更改边界或描述符参数会改变指标定义，比较时应保持一致。
- GBDT 在同一批 1200 K 帧上拟合并计算 SHAP，没有另设 SHAP 回归器验证集。因此 SHAP 百分比描述的是该批数据上的拟合归因，不应当作误差预测模型的外部泛化结果。
- 模型和数据必须具有可比的参考标签、单位、原子顺序/结构信息及周期性设置，否则误差或区域分析可能无效。
- 脚本只打印汇总值，不保存逐帧误差、SHAP 明细或图表。

## 6. 依赖

运行环境需提供 PyTorch、ASE、MACE、DScribe、NumPy、scikit-learn 和 SHAP，并能读取相应 MACE 模型文件及 XYZ 数据。
