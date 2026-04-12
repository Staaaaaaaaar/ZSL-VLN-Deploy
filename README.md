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

### 3. Ubuntu 22.04 GStreamer 依赖

```powershell
sudo apt update
sudo apt install -y gstreamer1.0-tools \
                    gstreamer1.0-plugins-base \
                    gstreamer1.0-plugins-good \
                    gstreamer1.0-plugins-bad \
                    gstreamer1.0-plugins-ugly \
                    gstreamer1.0-libav
```

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

