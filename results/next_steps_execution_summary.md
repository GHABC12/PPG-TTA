# PPG-SpO₂ 下一阶段执行记录

## 范围与隔离

- 本轮只用服务器上已有 VitalDB 数据（23 个可用病例），没有下载其他数据集。
- PaPaGei 采用病例分组五折；每个外层测试折只用于最终评分，下一折作为验证折，剩余病例用于训练。
- 四种 baseline 策略使用同一折分、随机种子和训练折统计量。主比较关闭 morphology 辅助损失。
- 目标训练使用训练折 `y_mean/y_std` 标准化并反变换，数学上等价于训练折均值残差回归后还原均值；没有重复实现一个数学等价的“残差版本”。
- TTA 仍按 prequential 顺序执行（先预测当前批，再用当前批无标签信号更新后续批），每病例重置。实现中测试 SpO₂ 标签在全部预测与更新结束后才合并用于评分。

## Encoder 表征诊断

冻结 PaPaGei 表征 512 个特征通道的每折标准差中位数约为 0.262–0.307，近零方差特征比例为 0。因而没有证据表明所有通道都塌缩。协方差有效秩为 8.5–11.1（约为 512 维的 1.7%–2.2%），显示表征有效维度偏低或高度相关；该结果不能单独证明这是 SpO₂ baseline 落后的原因。详细结果见 [feature_diagnostics.json](feature_diagnostics.json)。

## SpO₂ baseline 对比

对比冻结 Linear Probe、LN+Head、末 4 个 encoder block 微调和 Full Fine-tune。主指标为病例等权 MAE。

| 策略 | 五折 OOF 病例 MAE | 五折 OOF 窗口 MAE |
| --- | ---: | ---: |
| Linear Probe | 0.934 | 0.845 |
| LN+Head | **0.904** | **0.800** |
| Last 4 Layers | 0.930 | 0.825 |
| Full Fine-tune | 0.928 | 0.827 |
| 训练折中位数常数基线 | **0.683** | 0.575 |

按每折验证病例 MAE 选择策略后，fold 0、1、3、4 选择 Last 4 Layers，fold 2 选择 Full Fine-tune；选择后外层 OOF 病例 MAE 为 0.924。与训练折中位数常数基线相比，误差高 0.242，病例 bootstrap 95% CI 为 [0.145, 0.337]。LN+Head 是表中较好的固定策略，但也未超过常数基线。策略选择仍是 23 病例、单随机种子的探索结果；验证折也参与早停。

结果见 [baseline_strategy_summary.json](baseline_strategy_summary.json)、[baseline_strategy_metrics.csv](baseline_strategy_metrics.csv) 与 [baseline_strategy_case_metrics.csv](baseline_strategy_case_metrics.csv)。当前样本标签集中在 100 附近，baseline 尚未修复，故不应声称获得了可推广的 SpO₂ 估计器。

## 选择性 TTA 小范围比较

在 fold 0 的验证折（4,921 个窗口）比较了 no-update、quality-only、一种 uncertainty gate 以及 quality/uncertainty 组合。验证病例 MAE 由 source 的 0.5116 降至 quality-only 的 0.5102，因此仅选择该配置用于 fold 0 外层测试；没有用测试标签选配置。MC Dropout gate 的方差阈值 0.01 只放行 2/4,921 个样本，按该阈值其行为近似 no-update；uncertainty-weighted 配置验证 MAE 为 0.5205。当前不确定性门限尚未校准出有用覆盖率。

选中的 quality-only 配置在 fold 0 外层测试（4 个病例）病例 MAE 为 0.9465，source-only 为 0.9486，差异很小；训练折中位数基线为 0.4728。此单折结果不构成 TTA 有效性的证据。比较记录见 `results/vitaldb_pilot/selective_tta_fold0/` 服务器输出；验证选择摘要单独保存在 [selective_tta_fold0_selection.json](selective_tta_fold0_selection.json)。

## 人工输入扰动

固定 fold 0 检查点与 TTA 配置，对 4 个测试病例（5,205 个窗口）执行 4 类扰动各 3 个强度。扰动施加在缓存中已逐窗归一化的波形上；这是模型输入端压力测试，不模拟归一化前的真实传感器增益或经滤波后的真实漂移。每窗扰动由病例、时间和强度确定性设种子，污染后重算质量分和形态量；标签保持原监护仪 SpO₂，仅用于最终评分。

source-only 的病例 MAE 相对 clean fold 0 的 0.9486 变化如下：

- gain 输入端倍率 0.75、1.25、1.5：0.9188、1.2664、1.5077；倍率高于 1 时误差明显变大。
- baseline drift 幅度 0.1、0.25、0.5 倍窗口标准差：0.9499、0.9558、0.9881。
- Gaussian noise 标准差 0.01、0.05、0.1 倍窗口标准差：0.9541、1.0091、1.0748。
- motion burst 幅度 0.5、1、2 倍窗口标准差：0.9509、0.9511、1.0096。

所选 quality-only TTA 对各档通常只带来约 0.000–0.002 的病例 MAE 变化，远小于源模型与中位数基线的差距。该扰动结果只来自一个外层折和四个病例，没有病例 bootstrap 区间；完整表见 [shift_stress_fold0.csv](shift_stress_fold0.csv)。

## 阶段结论

已实现训练策略对比、表征诊断、MC Dropout 质量/不确定性筛选接口、无 morphology 主 TTA 路径和确定性人工扰动评估。残差回归与现有训练目标等价。新基线仍明显输给常数中位数，TTA 改善接近零；下一步优先扩大 VitalDB 内的独立病例与低 SpO₂ 覆盖，而非在当前样本上继续扩大搜索。跨数据集真实域外评估因本轮“尽量只用 VitalDB”的数据约束未执行。
