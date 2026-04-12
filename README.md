# ZSL-VLN-Deploy

## 本地模型推理服务

已提供基于 vLLM 的本地模型服务脚本，默认配置在 `robot_deploy/configs/model_service.json`。

### 1. 配置模型路径

编辑 `robot_deploy/configs/model_service.json` 中的 `model_path`，可使用本地目录或 HuggingFace 仓库名。

### 2. 启动服务

后台启动并等待服务就绪：

```powershell
python scripts/start_model_service.py --wait-ready
```

前台启动（日志直接输出到当前终端）：

```powershell
python scripts/start_model_service.py --foreground
```

仅查看将执行的 vLLM 命令：

```powershell
python scripts/start_model_service.py --dry-run
```

### 3. 检查服务

```powershell
python scripts/check_model_service.py
```

### 4. 停止后台服务

```powershell
python scripts/stop_model_service.py
```

### 5. 连接到机器人运行脚本

默认模型端点是 `http://127.0.0.1:8003/v1`，与你当前 `robot_deploy/configs/default.json` 保持一致。
服务就绪后可直接运行：

```powershell
python scripts/run_single_step.py
```

## 说明

- 本地 vLLM 服务通常需要 Linux + NVIDIA GPU 环境，Windows 原生环境可考虑 WSL2。
- 若你使用 LoRA/Adapter 权重，请在 `extra_args` 中补充对应参数。
