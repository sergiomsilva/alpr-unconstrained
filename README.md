# 非受限场景下的车牌识别 (ALPR Unconstrained)

## PyCharm 2026.1 一键启动

1. **Clone** `https://github.com/blackjaxx/alpr-unconstrained.git`
2. PyCharm 自动识别 `.python-version`，**创建 venv** 即可
3. 打开 `pyproject.toml` → 点击 "Install requirements"
4. 终端运行 `bash get-networks.sh`
5. 编辑器右上角选择 **`ALPR WebUI`** 配置 → 点击 Run
6. 浏览器访问 `http://127.0.0.1:5000`

> 详细步骤见 [`PYCHARM.md`](./PYCHARM.md)

## 简介

本仓库是 ECCV 2018 论文 *"License Plate Detection and Recognition in Unconstrained Scenarios"* 的实现代码。

v2.0.0 现代化升级：Python 3.11+ / Keras 3 / TensorFlow 2.x / YOLOv8 / PaddleOCR，不再需要编译 Darknet C 代码。同时提供了 Flask Web 界面、命令行入口和 PyCharm 完整运行配置。

引用文献 (BibTeX) 详见 `README` 末尾或论文网页。

## 三种启动方式

| 场景 | 命令 |
|------|------|
| PyCharm 启动 (推荐) | Run Configurations → `ALPR WebUI` → ▶ Run |
| 命令行 Web 界面 | `python run.py` |
| 命令行三阶段流水线 | `bash run.sh -i samples/test -o /tmp/output -c /tmp/output/results.csv` |

## 环境要求

```bash
pip install -e .
```

自动安装：Keras 3 + TensorFlow 2.15+、Ultralytics YOLOv8、PaddleOCR、OpenCV 4.8+、Flask 3.0+、NumPy。

## 下载模型

```bash
bash get-networks.sh
```

YOLOv8 和 PaddleOCR 模型首次运行时自动下载。

## 三阶段流水线

```
原始图片 → YOLOv8 (车辆检测) → WPOD-Net (车牌检测) → PaddleOCR (字符识别) → 结果
```

| 阶段 | 引擎 | 替代原版 |
|------|------|----------|
| 车辆检测 | ultralytics/YOLOv8n | Darknet YOLO-VOC |
| 车牌检测 | Keras WPOD-Net | Darknet 自定义网络 |
| 字符识别 | PaddleOCR | Darknet 自定义网络 |

## 训练车牌检测器

```bash
mkdir models
python create-model.py eccv models/eccv-model-scratch
python train-detector.py --model models/eccv-model-scratch --name my-model \
    --train-dir samples/train-detector --output-dir models/my-trained-model/ \
    -op Adam -lr .001 -its 300000 -bs 64
```

## 关于 GPU / CPU

WPOD-Net (Keras/TF)、YOLOv8、PaddleOCR 自动识别 GPU 并加速推理。Apple Silicon Mac 用户安装 `tensorflow-metal` 后也可使用 GPU。

## 项目结构

```
alpr-unconstrained/
├── run.py                         PyCharm 入口 (web/cli)
├── webapp.py                      Flask Web 应用
├── templates/index.html           Web UI 页面
├── PYCHARM.md                     PyCharm 配置指南
├── vehicle-detection.py           阶段1: 车辆检测 (YOLOv8)
├── license-plate-detection.py     阶段2: 车牌检测 (WPOD-Net)
├── license-plate-ocr.py           阶段3: 字符识别 (PaddleOCR)
├── create-model.py                WPOD-Net 模型定义
├── train-detector.py              训练脚本
├── run.sh                         Bash 流水线
├── get-networks.sh                下载预训练模型
├── annotation-tool.py             标注工具
├── gen-outputs.py                 结果可视化
│
├── pyproject.toml                 依赖 + PyCharm 项目配置
├── .python-version                PyCharm 自动识别 Python 版本
├── .gitignore                     屏蔽 venv / .idea / 模型等
│
├── .idea/                         PyCharm 项目配置 (已纳入版本控制)
│   ├── alpr-unconstrained.iml
│   ├── modules.xml
│   └── runConfigurations/
│       ├── ALPR_WebUI.xml
│       └── ALPR_CLI.xml
│
└── src/                           核心库
    ├── label.py
    ├── loss.py
    ├── keras_utils.py
    ├── data_generator.py
    ├── sampler.py
    ├── projection_utils.py
    ├── drawing_utils.py
    └── utils.py
```

## 命令行脚本对照表

| 命令 | 说明 |
|------|------|
| `python run.py` | 启动 Web (端口 5000) |
| `python run.py cli <input> <output>` | 三阶段流水线 |
| `alpr-webui` | 安装后控制台入口 |
| `python train-detector.py ...` | 训练车牌检测器 |
| `bash run.sh -i ... -o ... -c ...` | Bash 流水线 |
| `bash get-networks.sh` | 下载预训练模型 |
