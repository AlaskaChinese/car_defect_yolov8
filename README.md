# Car Paint Defect Detection — YOLOv8

基于 YOLOv8 的汽车漆面缺陷检测模型，使用 Kaggle P100 GPU 训练。

---

## 模型说明

| 项目 | 内容 |
|------|------|
| 模型架构 | YOLOv8s（Ultralytics） |
| 任务类型 | 目标检测（边界框 + 类别分类） |
| 预训练权重 | COCO（yolov8s.pt）→ 微调 |
| 输入 | RGB 图像，640x640 |
| 输出 | 边界框坐标 + 类别 + 置信度 |
| 训练平台 | Kaggle（NVIDIA P100 GPU） |
| 训练配置 | 100 epochs, batch=16, patience=20（早停） |

### 与 UNet++ 方案的对比

| | UNet++（分割） | YOLOv8s（检测） |
|---|---|---|
| **输出** | 二值掩码（有/无缺陷） | 边界框 + 4 类缺陷名称 |
| **缺陷分类** | 不区分类别 | dirt, runs, scratch, water marks |
| **数据匹配度** | 低（需要像素级标注） | 高（数据集就是边界框格式） |
| **推理速度** | 较慢 | 快 |
| **实用性** | 适合精确标记区域 | 适合快速定位和分类 |

> **结论**：YOLOv8 在该数据集上表现更优，能区分缺陷类型，检测速度快，数据格式完全匹配。实际部署推荐使用 YOLOv8。

---

## 数据集

来源：[Roboflow - Final Year Car Paint Defect](https://universe.roboflow.com/poli-h7nww/final-year-car-paint-defect/dataset/1)（CC BY 4.0）

格式：YOLOv8（边界框标注），4 个缺陷类别：

| 类别 ID | 名称 | 说明 |
|---------|------|------|
| 0 | `dirt` | 脏污 |
| 1 | `runs` | 流挂 |
| 2 | `scratch` | 划痕 |
| 3 | `water marks` | 水渍 |

---

## 文件结构

```
car_defect_yolov8/
├── README.md                # 本文件
├── kaggle_train.ipynb       # Kaggle 训练 Notebook
├── best.pt                  # 训练好的 YOLOv8 权重（从 Kaggle 下载）
└── final year car paint defect.v1i.yolov8/
    ├── data.yaml            # 数据集配置
    ├── train/               # 训练集（images/ + labels/）
    ├── valid/               # 验证集
    └── test/                # 测试集
```

---

## 环境配置（Ubuntu 18.04）

### 安装 Python 3.8

Ubuntu 18.04 默认 Python 为 3.6，需手动安装：

```bash
sudo apt update
sudo apt install -y software-properties-common build-essential \
    curl git libssl-dev libffi-dev libjpeg-dev libpng-dev

# 添加 deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev

# 安装 pip
curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8
```

### 创建虚拟环境

```bash
cd /path/to/car_defect_yolov8
python3.8 -m venv .venv
source .venv/bin/activate
```

### 安装依赖

```bash
# PyTorch（根据硬件选择）
# 仅 CPU：
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU + CUDA 11.7：
# pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# YOLOv8
pip install ultralytics==8.0.196

# 国内镜像加速（对 ultralytics 等包生效）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
```

### 推荐版本汇总

| 软件 / 库 | 推荐版本 | 说明 |
|-----------|----------|------|
| Ubuntu | 18.04 LTS | 目标部署系统 |
| Python | **3.8.x** | deadsnakes PPA 安装 |
| PyTorch | **2.0.1** | 支持 Python 3.8 |
| Ultralytics | **8.0.196** | YOLOv8 框架 |
| CUDA（可选） | 11.7 / 11.8 | 仅 NVIDIA GPU 需要 |

---

## 训练（Kaggle 云端）

详见 `kaggle_train.ipynb`，步骤：

1. 登录 kaggle.com → **Datasets → New Dataset** → 上传数据集文件夹
2. **Code → New Notebook** → 添加数据集 → 开启 **GPU P100**
3. Settings → **Internet → On**（需手机验证）
4. 导入 `kaggle_train.ipynb`，**Run All**（约 30~40 分钟）
5. 训练完成后在 Output 下载 `best.pt` 和 `results.zip`

---

## 推理使用

### 基本用法

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("your_image.jpg", conf=0.25)

# 显示检测结果（弹出图片窗口）
results[0].show()

# 保存结果图片
results[0].save("output.jpg")
```

### 获取检测结果数据

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("your_image.jpg", conf=0.25)

for box in results[0].boxes:
    cls_id = int(box.cls)
    name   = results[0].names[cls_id]   # dirt / runs / scratch / water marks
    conf   = float(box.conf)            # 置信度
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{name}: {conf:.2f} @ ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
```

### 批量推理

```python
from ultralytics import YOLO

model = YOLO("best.pt")

# 整个文件夹
results = model("test/images/", conf=0.25, save=True)

# 结果自动保存到 runs/detect/predict/
```

### 命令行推理

```bash
yolo detect predict model=best.pt source=your_image.jpg conf=0.25
```

---

## 常见问题

**Q: 下载 PyTorch 很慢？**
A: 前往 https://download.pytorch.org/whl/torch/ 手动下载 `torch-2.0.1+cpu-cp38-cp38-linux_x86_64.whl`，然后 `pip install xxx.whl`。

**Q: 推理时报 `CUDA out of memory`？**
A: 使用 CPU 推理：`model.predict(source, device="cpu")`。

**Q: Kaggle Notebook 安装包报网络错误？**
A: 右侧 Settings → Internet → On（首次需手机验证）。

**Q: 想在本地 CPU 训练而不用 Kaggle？**
A: 可以，但速度很慢（约 2~5 小时）。修改训练命令中 `device=0` 为 `device="cpu"`，并将 `batch` 改为 8。
