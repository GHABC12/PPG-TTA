# Patient-Aware TTA 方案：病例级可行性执行记录

日期：2026-10-11  
数据：仅使用服务器已有 VitalDB；未下载新数据集。  
代码：`src/qmtta/{adapter,patient_calibration,augment,uncertainty,patient_tta}.py`、`scripts/evaluate_patient_adapter_tta.py`、`scripts/summarize_patient_adapter_tta.py`

## 结论

本轮完成五折病例级可行性试验，但**方案验收未通过**。完整 PATTA 在 clean 条件的 MAE 为 0.889，训练折中位数基线为 0.634；完整 PATTA 在五种条件下均未显著优于 Frozen source（23 病例配对 bootstrap 95% 区间均跨过 0），也没有任何 TTA 方法接近训练折中位数。当前结果不支持将 PATTA 作为有效 SpO₂ TTA，也不支持临床或患者级泛化结论。

## 适用范围

方案目标是患者感知适应，但当前 VitalDB 索引、轨道审计与折分只有 `case_id`，没有可验证的 patient ID。因此本轮仅能按 held-out case 做适应并重置状态，结论限定为**病例级（case-aware）适应可行性**；不能声称患者级互斥或跨手术患者泛化。阶段分析代理的主要发现是 patient ID 缺失；这是解释和后续设计的首要限制。

## 实现与评估协议

- 沿用五折 source encoder/head 与 LN+Head checkpoint。Encoder 和 SpO₂ Head 冻结，只训练 512→32→512 残差 Adapter；最后一层零初始化，初始映射为恒等映射。
- 每个测试病例单独初始化 Adapter 和优化器。按时间排序，前 32 个无标签窗口仅用于适应、不计入指标；其余数据逐批先预测，再使用当前批无标签波形更新。该病例的标签只在预测和更新完成后用于计分。
- 更新目标为增强视图预测一致性及病例内 prototype 紧致项，并加入权重 0.01 的恒等映射约束以抑制无标签表示塌缩。该约束是本轮实现中的稳定项，不属于原方案目标函数。
- MC Dropout 在 Adapter dropout 层采样 8 次；不确定性权重为 `exp(-variance / 1.0)`，方差以 source head 标准化目标单位计算。实测平均权重接近 1（约 0.998–1.000），因此本轮 uncertainty weighting 的实际作用很小。
- 消融：Consistency only、Patient prototype only、Uncertainty weighted consistency、Full PATTA。
- 人工输入压力：增益 0.8/1.2、0.1 Hz 低频漂移（相对标准差 0.1）、Gaussian noise（相对标准差 0.05）。扰动加在已滤波、逐窗 z-score 的输入上，只是缓存模型输入压力测试，不等于原始传感器偏移或临床域偏移。
- 对照：Frozen source、LN+Head、训练折中位数。指标先在病例内计算 MAE，再对 23 个病例宏平均。配对 bootstrap 以病例为单位重采样 2000 次，报告 `Frozen MAE − 方法 MAE` 的 95% 区间；正值代表方法相对 Frozen 更好。此区间反映病例抽样不确定性，不替代独立外部验证。
- 参数固定为 bottleneck=32、dropout=0.1、学习率=1e-4、每批 5 步、批量=32、warm-up=32 窗口、prototype 权重=0.1、MC 次数=8；未用外层测试标签调参。

## 主要结果

表中为病例宏平均 MAE，单位沿用模型目标单位。括号是 `Frozen MAE − Full PATTA MAE` 的病例配对 bootstrap 95% 区间。

| 条件 | Frozen | LN+Head | Full PATTA | Patient only | 训练折中位数 | Full PATTA 相对 Frozen 改善 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.910 | 0.876 | 0.889 | 0.899 | 0.634 | +0.021 [-0.019, +0.067] |
| 增益 0.8 | 0.827 | 0.805 | 0.833 | 0.813 | 0.634 | -0.005 [-0.051, +0.044] |
| 增益 1.2 | 0.986 | 0.879 | 0.963 | 0.963 | 0.634 | +0.025 [-0.028, +0.083] |
| 低频漂移 | 0.911 | 0.875 | 0.894 | 0.901 | 0.634 | +0.018 [-0.020, +0.060] |
| Gaussian noise | 0.916 | 0.880 | 0.897 | 0.904 | 0.634 | +0.019 [-0.012, +0.056] |

Consistency only 在增益 1.2 下的改善为 +0.047 [0.007, 0.088]，uncertainty weighted 为 +0.048 [0.009, 0.089]；但 clean 条件没有显著改善，且所有 TTA 的误差仍远高于中位数基线。不能据单一人工扰动上的相对改善判定适应成功。

## 诊断与风险

- **高：基线性能不足。** Frozen、LN+Head 与各 TTA 的误差均高于训练折中位数；继续优化 Adapter 前应先确认数据标签、输出单位、按折归一化及 baseline/head 训练设定。
- **高：患者身份不可验证。** VitalDB 当前索引只提供病例号。本轮没有患者互斥划分证据，需先找到可靠的 patient ID 映射，才能声称 patient-aware 或患者级泛化。
- **高：Prototype 目标易引起收缩。** Patient-only 与 Full PATTA 的 Adapter 表示改变量明显高于 consistency-only（clean 下约 0.366/0.357 对 0.011）；加入恒等约束后预测没有变成常数，但大的特征改变量仍值得警惕。不能把 prototype 收紧视为有效适应证据。
- **中：不确定性权重近似常数。** 当前归一化和温度下 `exp(-variance)` 接近 1，几乎没有筛选高不确定窗口；现有实验不能支持 uncertainty weighting 的收益机制。
- **中：扰动位置限制。** 缓存输入已丢失绝对幅值信息，增益/漂移/noise 测试不能代表原始 PPG 光学或真实临床变化；输入为单通道 `SNUADC/PLETH`，不是双波长 SpO₂ 光学测量。

## 文件记录

- `case_metrics.csv`：23 个病例 × 5 种输入条件 × 7 种方法的病例指标（805 行）。
- `fold_metrics.csv`：折级聚合指标。
- `summary_by_shift_method.csv`、`summary.json`：汇总表和机器可读摘要。
- `paired_bootstrap_summary.csv`：各条件、方法的病例配对 bootstrap 区间。
- `update_diagnostics.csv`：约 15,780 条适应更新与不确定性诊断。

## 下一步

1. 暂停将当前 PATTA 作为有效方法；优先审计 baseline/head、标签单位和 fold 配置，并确认中位数基线为何显著更好。
2. 获得可靠 patient ID 后重做按患者互斥的评估；在此之前统一使用“病例级”表述。
3. 若继续探索 prototype，先改为防止收缩的目标并在开发折固定方案，再用未触碰的 held-out folds 评估；保留 clean 与中位数基线作为硬门槛。
4. 不新增数据集；后续实验仍先限定在现有 VitalDB。
