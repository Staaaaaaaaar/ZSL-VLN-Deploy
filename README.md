# ZSL-VLN-Deploy

## 环境依赖安装

### 1. 机器人控制环境（推荐独立）

```powershell
conda env create -f envs/robot-control.yml
conda activate robot-control
```

### 2. 模型服务环境（推荐独立）

```powershell
conda env create -f envs/model-service.yml
conda activate model-service
```

### 3. Ubuntu 22.04 相机流依赖（FFmpeg + GStreamer）

```powershell
sudo apt update
sudo apt install -y ffmpeg \
										gstreamer1.0-tools \
										gstreamer1.0-plugins-base \
										gstreamer1.0-plugins-good \
										gstreamer1.0-plugins-bad \
										gstreamer1.0-plugins-ugly \
										gstreamer1.0-libav
```

### 4. OpenCV 环境说明（关键）

若希望在 Python 中使用 `cv2.CAP_GSTREAMER`，通常不要直接使用 `pip install opencv-python`。
多数 pip 预编译 wheel 不带 GStreamer 支持，会在 `cv2.getBuildInformation()` 中显示 `GStreamer: NO`。

建议在机器人控制环境中使用 conda-forge OpenCV：

```powershell
conda install -n robot-control -c conda-forge -y \
	opencv gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```

验证：

```powershell
conda run -n robot-control python -c "import cv2; s=cv2.getBuildInformation(); print('\n'.join([l for l in s.splitlines() if 'Video I/O' in l or 'GStreamer' in l or 'FFMPEG' in l]))"
```

目标是看到：`FFMPEG: YES` 且 `GStreamer: YES`。

## 本地模型推理服务

已提供基于 vLLM 的本地模型服务脚本，默认配置在 `configs/model_service.json`。

### 1. 配置模型路径

编辑 `configs/model_service.json` 中的 `model_path`，可使用本地目录或 HuggingFace 仓库名。
本仓库默认本地路径为 `models/Qwen2.5-VL-3B_rl_rxr_4000_step350`。

### 2. 启动服务

后台启动并等待服务就绪：

```powershell
python scripts/start_model_service.py
```

仅查看将执行的 vLLM 命令：

```powershell
python scripts/start_model_service.py --dry-run
```

### 3. 单次图片推理测试

```powershell
python scripts/test_model_service_infer.py --image <local_image_path> --instruction "Walk forward and stop at the doorway."
```

### 4. 连接到机器人运行脚本

默认模型端点是 `http://127.0.0.1:8003/v1`，与你当前 `configs/default.json` 保持一致。
服务就绪后可直接运行：

```powershell
python scripts/run_single_step.py
```

## 机器人侧基础测试脚本

### 1. 相机取流测试

```powershell
python scripts/test_robot_camera.py
```

### 2. 运动控制测试

```powershell
python scripts/test_robot_motion.py
```

