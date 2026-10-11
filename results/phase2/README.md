# PPG SpO2 TTA Phase 2：阶段执行记录

日期：2026-10-11  
数据：服务器已有 VitalDB 1.0.0 与缓存；未下载其他数据集。  
代码目录：`/home/user/shiyunxiao/PPG/PPG-TTA/`

## 本轮范围

复用现有冻结 PaPaGei 表征缓存，检查表示与标签关系、VitalDB 轨道覆盖、PPG/监护仪标签的时间网格，以及 10/30/60 秒窗口的 SpO2 标签覆盖。对 HR 和呼吸率候选标签做了按病例分组的 Ridge 迁移探针。所有探针使用病例分组的五折划分：每折留一组测试病例，下一组用于 Ridge 正则参数选择，其余病例用于训练。

本轮数据为 23 个病例、25,638 个已有 10 秒窗口、512 维冻结 PaPaGei 表征。新增计算结果均写入本目录；原始 VitalDB 文件和模型缓存不纳入 Git。

## 关键发现

### SpO2 表征探针

| 探针 | 病例均值 MAE（百分点） | 窗口 MAE | 窗口 R² | 窗口 Spearman |
|---|---:|---:|---:|---:|
| Ridge，验证集选择 alpha | 1.074 | 1.010 | -0.407 | -0.009 |
| XGBoost，固定配置 | 1.006 | 0.926 | -0.453 | 0.067 |
| 训练集 SpO2 中位数基线（已有结果） | 0.683 | — | — | — |

病例级平均绝对误差与现有基线采用相同聚合口径。两个冻结表征探针都没有超过中位数基线；窗口级相关性也很弱。当前不扩大 SpO2 TTA 搜索。

病例均值嵌入的 PCA 前两主成分解释 82.4% 和 6.8% 方差；PC1 与病例中位 SpO2 的 Spearman 相关为 0.537。该分析只有 23 个病例，属于探索性描述，不能说明窗口级可预测性。PCA/t-SNE 图见 `representation/`。

### HR 与呼吸率候选迁移任务

| 标签 | 可用 10 秒窗口 | 病例 | Ridge 病例均值 MAE | 中位数基线病例均值 MAE | 窗口 R² |
|---|---:|---:|---:|---:|---:|
| Solar8000/HR | 24,378 / 25,638（95.1%） | 23 | 5.14 bpm | 11.68 bpm | 0.551 |
| Solar8000/RR_CO2 | 24,180 / 25,638（94.3%） | 23 | 2.38 次/分 | 2.57 次/分 | -0.504 |

HR 探针明显优于常数中位数基线；呼吸率只在病例 MAE 上略有改善，窗口 R² 仍为负。两者是同一数据源中的监护仪标签，不能视为独立临床参考，结果需要进一步做病例级分层与外部验证。

### 标签覆盖与时间轴

23 个已索引病例中，PPG、SpO2、Solar HR 和 PLETH HR 均有轨道；呼吸率候选 Solar VENT_RR 为 22/23，Solar RR_CO2 和 Primus RR_CO2 均为 23/23。BIS/SQI 为 21/23，但它是 BIS/EEG 质量信号，不是 PPG SQI 真值。

在每个已有 fold 选取 1 个病例（5 个样本）读取共同时间戳：时间戳单调递增，间隔最大误差为 0 秒；PPG 与 SpO2 时间格对齐。SpO2 在完整记录的 1 秒网格中通常只覆盖约 43%–49%，因此窗口标签仍须按有效样本比例过滤。

| 窗口长度 | 至少 5 个有效 SpO2 点 | 固定 50% 有效点 |
|---|---:|---:|
| 10 秒 | 93.9% | 93.9% |
| 30 秒 | 95.1% | 92.8% |
| 60 秒 | 95.8% | 91.7% |

旧规则“至少 5 个点”对 30/60 秒窗口明显更宽松，比较不同窗口时必须改用统一覆盖率。重算的 10 秒有效标签窗口为 25,646 个，与已有 embedding 索引 25,638 个相差 8 个；继续训练长窗口模型前需先确认这 8 个窗口来自边界或缓存筛选差异。

## 当前决策与后续

既有外层 OOF 结果中，最佳 LN+Head 策略病例 MAE 为 0.904，训练病例中位数基线为 0.683；Phase 2 的 SpO2 冻结表征探针也未推翻该比较。因此按研究计划继续暂缓扩大 SpO2 TTA。可把 HR 作为迁移探针的优先任务；呼吸率只保留为探索性任务。当前 SQI 标签不适用于 PPG 质量监督。

下一步优先事项：

1. 对齐并解释 10 秒索引与重算标签间的 8 个窗口差异。
2. 核对 HR/呼吸率轨道的记录间隔与测量来源，检查病例级分布和训练/测试折差异。
3. 在统一标签覆盖规则下，按研究计划决定是否提取 30/60 秒表征并与 10 秒结果比较。
4. 只有 SpO2 基线在病例级指标上稳定超过中位数基线后，再恢复 SpO2 TTA 扩展。

执行记录：随机森林探针在单线程训练阶段耗时过长，本轮中止，没有将其作为结果；XGBoost 早期一次运行因训练与预测预处理不一致而作废，修正后结果已重跑并写入正式摘要。

## 可复现脚本与产物

- 脚本：`scripts/phase2_representation_probe.py`
- 脚本：`scripts/audit_phase2_vital_tracks.py`
- 脚本：`scripts/audit_phase2_window_coverage.py`
- 脚本：`scripts/phase2_transfer_probes.py`
- 结果：`representation/probe_summary.json`、`representation/probe_fold_metrics.csv`
- 图表：`representation/case_centroid_pca.png`、`representation/case_centroid_tsne.png`
- 结果：`transferability/track_audit_summary.json`、`transferability/track_availability_by_case.csv`
- 结果：`transferability/timestamp_alignment_sample.csv`、`transferability/transfer_probe_summary.json`
- 标签窗口明细：`transferability/transfer_window_labels.csv`
- 结果：`label_coverage/window_coverage_summary.json`、`label_coverage/window_coverage_by_case.csv`
