# QM-TTA SpO₂ feasibility pilot

本仓库实现了 VitalDB-only 的 SpO₂ 可行性实验：数据审计、PaPaGei/SIGMA 线性探针、源尾微调、验证折调参和 TTA 组件消融均已完成。它是研究原型；VitalDB 监护仪 SpO₂ 不是血气 SaO₂，也不构成临床血氧测量验证。

## 环境配置

仓库不包含 VitalDB 原始信号、缓存、预训练权重或微调检查点。准备好本地数据、上游代码和权重后设置路径：

```bash
export VITALDB_DIR=/path/to/existing/vitaldb/1.0.0
export PAPAGEI_REPO=third_party/papagei-foundation-model
export PAPAGEI_CHECKPOINT=weights/papagei_s.pt
export SIGMA_REPO=third_party/SigmaPPG
export SIGMA_CHECKPOINT=weights/sigma.pth
PYTHONPATH=src
PY=python
```

PaPaGei、SIGMA 的权重及上游源码应按各自许可和官方说明单独获取；本仓库只记录 SIGMA 上游 commit 和兼容性补丁。

## 首轮数据审计

在已安装 PyTorch、vitaldb、SciPy、pandas、scikit-learn 的 Python 环境运行：

```bash
PYTHONPATH=src $PY scripts/audit_vitaldb.py \
  --data-dir "$VITALDB_DIR" \
  --out-dir results/vitaldb_pilot --max-cases 24
```

`--max-cases 24` 用于小规模可行性试验；设为 `0` 才会遍历全目录。每个成功病例只读取 PPG 和 SpO₂ 两条轨道，输出 10 秒非重叠窗口、统一到 125 Hz 并做 Chebyshev 0.5–12 Hz 滤波及窗口内 z-score。窗口标签取有效 `[70,100]` SpO₂ 的中位数，至少要求 5 个有效秒值；不会用标签补缺失值。窗口波形和无标签形态量写入 `../qmtta_cache/`，索引写到结果目录。

输出文件：

- `case_audit.csv`：病例轨道读取、有效值和窗口计数。
- `window_index.csv`：窗口时间、病例、SpO₂ 汇总、质量分和形态指标。
- `folds.json`：按病例分组的五折分配；窗口不会随机跨折。
- `summary.json`：病例/窗口覆盖和标签分布摘要。
- `data_audit.md`：人可读的数据覆盖与配对报告。

`sVRI`、`IPA` 与 skew-SQI 是探索性形态描述量，不是氧饱和度真值；质量分是固定的无标签启发式。IPA 检测失败会写入有效掩码。

## 首个模型与 TTA 流程

审计成功且产生至少 3 个有窗口病例后，可按以下顺序运行。缓存窗口不会再次读取 `.vital` 大文件：

```bash
export PYTHONPATH=src
PY=python
$PY scripts/extract_embeddings.py --index results/vitaldb_pilot/window_index.csv
$PY scripts/train_source.py --test-fold 0
$PY scripts/evaluate_tta.py --test-fold 0
```

PaPaGei adapter 从现有官方实现读取 `outputs[-1]` 的 512 维 pooled latent，并要求官方检查点严格匹配。source 阶段按病例折分，训练 SpO₂ 线性头和形态辅助头，验证折选择 epoch，测试折只在评估时读取标签。TTA 每个测试病例重置到 source encoder，只开放最后一个 ResNet block；损失由质量门控一致性、形态保持和参数近端约束组成。`EpisodicQualityMorphTTA.step()` 不接收 SpO₂ 标签参数。

`scripts/train_finetune.py` 现在提供 `ln_head`、`last_layers`（默认解冻末 4 个 block，可由 `--tail-blocks` 设置）和 `full` 三种策略；`train_source.py` 是 frozen linear probe。主策略比较默认关闭形态辅助损失（`--aux-morph-weight 0`），而病例折分、训练折统计、验证早停和外层测试隔离保持一致。目标使用训练折 `y_mean/y_std` 标准化并在预测时反变换，这已等价于训练折均值残差回归，无需重复建模。

```bash
python scripts/train_finetune.py --backbone papagei --strategy ln_head \
  --test-fold 0 --val-fold 1 --out-dir results/vitaldb_pilot/baselines/ln_head
python scripts/train_finetune.py --backbone papagei --strategy last_layers \
  --tail-blocks 4 --test-fold 0 --val-fold 1 \
  --out-dir results/vitaldb_pilot/baselines/last_layers
python scripts/train_finetune.py --backbone papagei --strategy full \
  --encoder-lr 3e-6 --test-fold 0 --val-fold 1 \
  --out-dir results/vitaldb_pilot/baselines/full
```

可用 `scripts/diagnose_features.py` 对每折训练病例的冻结 embedding 报告通道标准差和协方差有效秩。`evaluate_tta.py` 默认关闭 morphology loss；可选 MC Dropout `--uncertainty-mode gated|weighted`，只将 teacher 中 Dropout 子模块临时设为 train，BatchNorm 保持 eval；方差阈值与权重尺度必须通过验证折选定。所有 TTA 预测仍遵循先预测当前批、再基于无标签当前批更新后续批的 prequential 协议，病例边界重置状态。评估标签在全部预测/更新完成后才与预测合并。

`evaluate_tta.py` 还提供 `--shift gain|baseline_drift|gaussian_noise|motion --severity ...` 的确定性输入端扰动。扰动作用在已逐窗归一化的缓存波形上，属于模型输入压力测试，不代表归一化前真实传感器增益；扰动波形的质量分和形态量会重新计算。每次只指定一种强度，宜报告完整严重度曲线，不得用外层测试标签选强度。

SIGMA 主干使用独立的 20 秒、50 Hz 协议（Butterworth 0.5–8 Hz + `[-1,1]` min-max），每个 2 秒生成一个 token（`patch_size=100`），先建立独立索引再执行：

```bash
$PY scripts/audit_vitaldb.py --protocol sigma --fs 50 --window-s 20 --stride-s 20 \
  --out-dir results/vitaldb_sigma --max-cases 24
$PY scripts/extract_embeddings.py --backbone sigma \
  --index results/vitaldb_sigma/window_index.csv \
  --out results/vitaldb_sigma/embeddings.npz --sigma-patch-size 100
$PY scripts/train_source.py --backbone sigma --index results/vitaldb_sigma/window_index.csv \
  --embeddings results/vitaldb_sigma/embeddings.npz --out-dir results/vitaldb_sigma/source
```

当前官方源码存在两处与该公开检查点不兼容之处，已在 `third_party/SigmaPPG` 记录的工作树补丁中处理：禁用检查点不存在的 q/v attention bias，并仅为多通道输入扩展位置嵌入。adapter 还单独保留 CLS 位置并对 patch 位置插值。补丁和仓库 commit 见 `third_party/SigmaPPG_PATCH.md`、`commits.txt`。初始 20 秒 smoke 输入的 patch projection 权重不匹配；采用 100 点（2 秒）patch 后，权重骨干加载率达到 100%。

## 验证折选参与 source-tail 微调

`scripts/tune_tta.py` 对每个外层折单独在其登记的验证折比较 6 组 `lr × lambda_morph` 参数和零更新控制；固定 `quality_threshold=0.5`、`lambda_reg=1e-4`、在线批大小 4。选参指标是等权病例 MAE。选出的配置只在对应外层测试折评估一次：

```bash
PY=python
$PY scripts/tune_tta.py --backbone papagei --index results/vitaldb_pilot/window_index.csv \
  --source-dir results/vitaldb_pilot/source --model-kind lp \
  --out-dir results/vitaldb_pilot/tta_tuning/lp --run-test
$PY scripts/tune_tta.py --backbone sigma --index results/vitaldb_sigma/window_index.csv \
  --source-dir results/vitaldb_sigma/source --model-kind lp \
  --out-dir results/vitaldb_sigma/tta_tuning/lp --run-test
```

TTA 采用 prequential 在线评估：每批先用当前模型预测，再用该批无标签 PPG/形态量更新后续批次；每个病例开始时重置。`evaluate_tta.py` 会校验 `--role validation/test` 与检查点中的折号，SpO₂ 标签不会传入更新函数。验证折也用于 source 模型 early stopping，验证选参分数可能偏乐观；外层测试折仍只用于最终评分。完成各折后可用 `scripts/summarize_tuned.py` 汇总 OOF 和病例级指标。

source-tail 微调示例（SIGMA 使用对应 `vitaldb_sigma` 索引和输出目录）：

```bash
$PY scripts/train_finetune.py --backbone papagei --index results/vitaldb_pilot/window_index.csv \
  --test-fold 0 --val-fold 1 --out-dir results/vitaldb_pilot/finetune
```

## 实施边界

这个首轮执行阶段不下载 BIDMC 或其他数据集；目前仅复用 VitalDB。PaPaGei 的预训练暴露中包含 VitalDB 类数据，所以该设置不能称为完全未见过的外部域。TTA 的目标病例 SpO₂ 标签不得进入更新过程；测试时按病例重置模型状态。

## 首轮结果

抽样 24 例中 23 例有可用窗口。PaPaGei 10 秒协议产生 25,638 个窗口；SIGMA 20 秒协议产生 12,913 个窗口。两者都完成了按 case 分组的 5 折交叉验证（单随机种子），合计覆盖 23 个病例：

| Backbone | 方法 | OOF 窗口 MAE | 平均病例 MAE |
| --- | --- | ---: | ---: |
| PaPaGei-S | Source-only LP | 0.835 | 0.929 |
| PaPaGei-S | 训练折中位数常数 | 0.575 | 0.683 |
| PaPaGei-S | Episodic QM-TTA | 0.835 | 0.929 |
| SIGMA-PPG-Pro | Source-only LP | 0.916 | 1.026 |
| SIGMA-PPG-Pro | 训练折中位数常数 | 0.610 | 0.765 |
| SIGMA-PPG-Pro | Episodic QM-TTA | 0.927 | 1.039 |

单位为 SpO₂ 百分点。两个线性探针和 TTA 均未胜过训练折中位数常数基线；OOF `ΔMAE`（Source−TTA）为 PaPaGei −0.00005、SIGMA −0.01038。当前可行性结果不支持“QM-TTA 有效”的结论。逐折、OOF、病例级指标和模型 smoke 输出在 `results/` 下；尚未做多随机种子或病例级 bootstrap。

## 后续阶段：源尾微调和验证集调参

已完成两个 backbone 的 source-tail 微调（只训练最后一个 backbone block 和任务头），并在每个外层折内比较 6 个更新配置及 no-update 控制。每折配置只由其登记的验证折选择，选中配置随后在对应外层测试折评估一次。表格给出 5 折 OOF 窗口 MAE 和等权病例 MAE：

| Backbone | 方法 | OOF 窗口 MAE | 平均病例 MAE |
| --- | --- | ---: | ---: |
| PaPaGei-S | 线性探针 | 0.8454 | 0.9339 |
| PaPaGei-S | 线性探针 + 验证集选参 TTA | 0.8455 | 0.9340 |
| PaPaGei-S | 源尾微调 | 0.7975 | 0.9011 |
| PaPaGei-S | 源尾微调 + 验证集选参 TTA | 0.7975 | 0.9010 |
| PaPaGei-S | 训练折中位数常数 | 0.5749 | 0.6827 |
| SIGMA-PPG-Pro | 线性探针 | 0.9244 | 1.0331 |
| SIGMA-PPG-Pro | 线性探针 + 验证集选参 TTA | 0.9250 | 1.0337 |
| SIGMA-PPG-Pro | 源尾微调 | 0.8872 | 1.0160 |
| SIGMA-PPG-Pro | 源尾微调 + 验证集选参 TTA | 0.8893 | 1.0181 |
| SIGMA-PPG-Pro | 训练折中位数常数 | 0.6104 | 0.7651 |

源尾微调相对线性探针改善了 MAE，但四种学习模型仍都没有超过简单的训练折中位数常数。验证集选出的 TTA 相对对应 source 模型只带来极小、方向不一致的变化；因此当前实验不支持 TTA 的稳健收益，也不能据此宣称最优超参数。完整逐折结果及配置见 `results/tuned_all_metrics.csv`、`results/tuned_all_case_metrics.csv`、`results/tuned_all_summary.json`，各折选择记录在 `results/vitaldb_{pilot,sigma}/tta_tuning/{lp,finetune}/`。

调参示例：

```bash
$PY scripts/train_finetune.py --backbone papagei --index results/vitaldb_pilot/window_index.csv \
  --test-fold 0 --val-fold 1 --out-dir results/vitaldb_pilot/finetune
$PY scripts/tune_tta.py --backbone papagei --index results/vitaldb_pilot/window_index.csv \
  --source-dir results/vitaldb_pilot/finetune --model-kind finetune \
  --out-dir results/vitaldb_pilot/tta_tuning/finetune --run-test
$PY scripts/summarize_tuned.py --results-root results --backbones papagei,sigma \
  --model-kinds lp,finetune --out-prefix tuned_all
```

每折的验证集既用于 source checkpoint early stopping，也用于 TTA 配置选择，因而验证分数会有重复使用带来的乐观偏差；外层测试折没有参与选参。该试验只有 23 个病例、一个随机种子，且 SpO₂ 标签集中在 100 附近。评估采用先预测当前批、再用无标签信号更新的 prequential 协议，病例间重置适配状态。下一步应优先增加独立病例/随机种子、低氧覆盖和病例级置信区间；在此之前不应扩大模型或调参搜索来追逐这一小样本上的“最优值”。

组件消融另用固定的 `lr=1e-5`、`lambda_reg=1e-4`，形态组 `lambda_morph=0.1`，每个变体在全部外层测试折各评估一次。`consistency_only` 使用全样本均匀一致性；`quality_gated_consistency` 增加质量筛选/加权；`morphology_without_quality` 增加形态项但不使用质量机制；`full_qm` 同时使用质量和形态。结果见 `results/tta_ablation_metrics.csv` 和 `results/tta_ablation_summary.json`。PaPaGei full QM 的平均病例 MAE 相对 source-only 改善约 0.0002 点，病例 bootstrap 95% CI 跨过 0；SIGMA 的四种 TTA 消融都使平均病例 MAE 增加。这个消融没有产生足以支持稳健收益的证据。
