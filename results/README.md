# YOLOv8 车漆缺陷检测 — 训练结果

本目录包含使用 YOLOv8s 模型在车漆缺陷数据集上训练 **100 个 epoch** 后的完整结果。

## 检测类别

| 类别 ID | 英文名 | 中文名 | 说明 |
|---------|--------|--------|------|
| 0 | dirt | 脏污 | 车漆表面的污渍或杂质 |
| 1 | runs | 流挂 | 喷漆过程中产生的流淌痕迹 |
| 2 | scratch | 划痕 | 车漆表面的刮擦损伤 |
| 3 | water marks | 水渍 | 水滴蒸发后留下的痕迹 |

## 最终性能指标

| 指标 | 数值 |
|------|------|
| **Precision** | 0.636 |
| **Recall** | 0.677 |
| **mAP@50** | 0.624 |
| **mAP@50-95** | 0.404 |

---

## 训练曲线

训练过程中各项损失和指标的变化趋势：

![训练曲线](results.png)

> 上图展示了 100 个 epoch 内的 box_loss、cls_loss、dfl_loss 以及 Precision、Recall、mAP 的变化趋势。可以看出模型在训练过程中稳步收敛。

---

## 混淆矩阵

### 原始混淆矩阵

![混淆矩阵](confusion_matrix.png)

### 归一化混淆矩阵

![归一化混淆矩阵](confusion_matrix_normalized.png)

> 混淆矩阵展示了模型对各类缺陷的分类准确度。归一化版本更直观地反映了每个类别的识别率。

---

## PR 曲线与评估指标

### Precision-Recall 曲线

![PR曲线](BoxPR_curve.png)

> PR 曲线反映了模型在不同置信度阈值下 Precision 和 Recall 的权衡关系，曲线下方面积即为 AP 值。

### F1-Confidence 曲线

![F1曲线](BoxF1_curve.png)

> F1 曲线展示了在不同置信度阈值下的 F1 分数，用于选择最佳的推理置信度阈值。

### Precision-Confidence 曲线

![Precision曲线](BoxP_curve.png)

### Recall-Confidence 曲线

![Recall曲线](BoxR_curve.png)

---

## 数据集标签分布

![标签分布](labels.jpg)

> 展示了训练集中各类别的样本数量分布及标注框的空间分布情况。

---

## 训练样本可视化

以下为训练过程中的数据增强效果示例（含 Mosaic 拼接、色彩增强等）：

| 训练初期 | 训练初期 | 训练初期 |
|----------|----------|----------|
| ![batch0](train_batch0.jpg) | ![batch1](train_batch1.jpg) | ![batch2](train_batch2.jpg) |

---

## 验证集预测效果

左列为真实标注（Ground Truth），右列为模型预测结果（Predictions）：

| 真实标注 | 模型预测 |
|----------|----------|
| ![val0_label](val_batch0_labels.jpg) | ![val0_pred](val_batch0_pred.jpg) |
| ![val1_label](val_batch1_labels.jpg) | ![val1_pred](val_batch1_pred.jpg) |
| ![val2_label](val_batch2_labels.jpg) | ![val2_pred](val_batch2_pred.jpg) |

> 通过对比真实标注与模型预测，可以直观评估模型的检测精度和定位能力。

---

## 训练配置

- **模型**: YOLOv8s（基于 COCO 预训练权重）
- **输入尺寸**: 640×640
- **训练轮数**: 100 epochs
- **批大小**: 16
- **早停耐心值**: 20
- **优化器**: SGD (lr=0.01, momentum=0.937, weight_decay=0.0005)
- **训练设备**: NVIDIA P100 GPU (Kaggle)

## 文件说明

| 文件 | 说明 |
|------|------|
| `args.yaml` | 完整的训练超参数配置 |
| `results.csv` | 逐 epoch 的训练指标记录 |
| `results.png` | 训练曲线汇总图 |
| `confusion_matrix.png` | 混淆矩阵 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 |
| `labels.jpg` | 数据集标签分布可视化 |
| `Box*.png` | 各评估指标曲线 |
| `train_batch*.jpg` | 训练批次可视化 |
| `val_batch*_labels.jpg` | 验证集真实标注 |
| `val_batch*_pred.jpg` | 验证集模型预测 |
| `weights/` | 模型权重（best.pt / last.pt） |
