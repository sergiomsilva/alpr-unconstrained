# PyCharm 2026.1 启动部署指南

本项目已经为 **PyCharm 2026.1** 完整配置好所有运行/调试环境，您只需要按下面步骤操作即可一键启动。

## 1. 环境准备

| 软件 | 版本 | 说明 |
|------|------|------|
| PyCharm | **2026.1** Professional 或 Community | Community 免费版足够 |
| Python | **3.11** | 项目根目录有 `.python-version` 自动识别 |
| Git | 最新版 | 拉取代码 |

> 已在仓库提交 `.gitignore`、`pyproject.toml`、`.python-version`，因此 PyCharm 打开后会**自动识别**项目类型。

## 2. 打开项目

1. 启动 PyCharm 2026.1
2. 选择 **Get from VCS** (或 `File > New > Project from Version Control`)
3. 输入仓库地址：
   ```
   https://github.com/blackjaxx/alpr-unconstrained.git
   ```
4. 选择本地目录 > 点击 **Clone**
5. PyCharm 自动识别为 Python 项目，并打开配置文件

## 3. 配置 Python 解释器 (venv)

PyCharm 2026.1 内置 `.python-version` 自动识别功能，步骤：

1. 打开项目后，PyCharm 右下角会提示 `Python 3.11 interpreter not found`
2. 点击提示 → **Add New Interpreter > Virtualenv > OK**
   （或：`Settings (Ctrl+Alt+S)` → `Project > Python Interpreter` → **Add Interpreter** → `Virtualenv` → `Location: <项目目录>/.venv`）
3. PyCharm 自动创建 venv 并激活

## 4. 安装依赖

方式 **A** — PyCharm 编辑器自动提示（推荐）：

打开 `pyproject.toml`，PyCharm 会高亮 `dependencies` 部分，**右上角会弹出"Install requirements"通知**，点击即可一键安装。

方式 **B** — 终端 (PyCharm 内置 Terminal)：

```bash
pip install --upgrade pip
pip install -e .
```

> `pip install -e .` 会以可编辑模式安装，所有代码修改**即时生效，无需重新安装**。

## 5. 下载模型权重

终端中执行：

```bash
bash get-networks.sh
```

这会下载 WPOD-Net 车牌检测模型（~12MB）。YOLOv8 和 PaddleOCR 在首次运行时自动下载。

## 6. 启动 Web 界面 — 两种方式

### 方式 A：PyCharm 图形化 Run 配置 (推荐)

本仓库已在 `.idea/runConfigurations/` 目录预置运行配置：

1. 右上角工具栏 → **Run Configurations** 下拉框 → 选择 **`ALPR WebUI`**
2. 点击绿色三角形 ▶ **Run**（或 🐞 调试按钮启动 Debug）
3. PyCharm 控制台会显示：
   ```
    * Running on http://127.0.0.1:5000
    * Running on http://192.168.x.x:5000
   ```
4. 浏览器访问 `http://127.0.0.1:5000`

> 这个路径支持 **断点调试**：`webapp.py` 或 `src/keras_utils.py` 中的代码可以被中断、单步执行。

### 方式 B：终端命令

```bash
python run.py
```

## 7. 三阶段流水线 CLI（也可 Run）

另一个预置运行配置 **`ALPR CLI`** 或终端命令：

```bash
python run.py cli samples/test /tmp/output
```

## 8. 常见问题

### Q1: `ModuleNotFoundError: No module named 'src'`
解决方案：右键 `src` 文件夹 → `Mark Directory as > Sources Root`。

### Q2: `TF / Keras 加载失败`
- Apple Silicon (M1/M2/M3) Mac 需安装 `tensorflow-metal`：
  ```bash
  pip install tensorflow-metal
  ```
- GPU 用户确保已安装 CUDA 12.x + cuDNN

### Q3: 模型下载慢
YOLOv8 (~6MB) 和 PaddleOCR (~150MB) 会在首次启动时下载。如需使用代理：
```bash
export HTTPS_PROXY=http://127.0.0.1:7890
python run.py
```

### Q4: Web 界面 Chinese 字符显示为方框
终端字符问题不影响 Web。Web 界面本身是完全中文的（`templates/index.html`）。

## 9. 部署到生产环境

> 仅供参考，如需 Docker/Nginx 部署，告诉我即可。

```bash
# 使用 Gunicorn 
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 "webapp:app"
```

---

## 🎯 一句话总结

```
Clone 仓库 → 打开 PyCharm → PyCharm 识别 .python-version → 创建 venv → 
点 "Install requirements" → bash get-networks.sh → 
右上角制 ALPR WebUI 配置 → 点击 ▶ Run → 浏览器 http://127.0.0.1:5000
```

---

有问题告诉我，我会进一步更新本指南。
