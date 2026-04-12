# ZSL-VLN-Deploy API 文档

本文档整理当前项目核心模块、类和关键方法，按执行链路组织：
图像获取 -> 模型推理 -> 动作解析 -> 控制下发。

## 1. Core 抽象与类型

### 模块
- `robot_deploy/core/contracts.py`
- `robot_deploy/core/types.py`

### 抽象接口

#### `RobotAdapter`
统一机器人控制接口。

关键方法：
- `connect(endpoint: RobotEndpoint) -> None`
- `check_connection() -> bool`
- `stand_up() -> None`
- `lie_down() -> None`
- `send_motion(cmd: MotionCommand) -> None`
- `stop() -> None`
- `read_state() -> RobotState`
- `close() -> None`

#### `ModelAdapter`
统一模型推理接口。

关键方法：
- `infer(request: NavigationRequest) -> ModelResponse`
- `close() -> None`

#### `ActionInterface`
统一动作文本解析与控制映射接口。

关键方法：
- `parse_model_response(response_text: str) -> list[NavigationAction]`
- `to_motion_commands(actions: list[NavigationAction]) -> list[MotionCommand]`

### 关键数据类
- `RobotEndpoint`: SDK 连接参数（本机 IP/端口，狗端 IP）
- `NavigationRequest`: 推理输入（instruction + image + metadata）
- `NavigationAction`: 归一化动作（前进/左转/右转/停止）
- `MotionCommand`: 机器人速度命令（vx/vy/yaw_rate + duration）
- `RuntimeSafetyLimits`: 运行时限幅
- `RuntimeStepResult`: 单步执行结果

## 2. Robot 侧

### 模块
- `robot_deploy/robot/zsl_highlevel.py`
- `robot_deploy/robot/video.py`

### `ZSLHighLevelRobot`
基于 zsibot SDK 的高层控制实现。

关键方法：
- `connect(...)`: 调用 SDK `initRobot`
- `check_connection()`: 兼容 `checkConnect/checkConnection` 两种 SDK 名称
- `stand_up()/lie_down()/stop()`
- `send_motion(cmd)`: 下发速度并按 `duration_sec` 自动停止
- `read_state()`: 读取电量、姿态、机体速度

### `FFmpegCameraStream`
基于 OpenCV FFmpeg 后端的 RTSP 相机流读取器（与运控解耦）。

关键方法：
- `start() / stop()`
- `read_latest(timeout_sec=...) -> FramePacket | None`
- `stats() -> dict`

设计要点：
- 后台线程持续拉流
- 主流程只取“最新帧”避免缓冲积压
- 通过 `OPENCV_FFMPEG_CAPTURE_OPTIONS` 注入低延迟选项（`fflags=nobuffer`、`flags=low_delay` 等）

## 3. Model 侧

### 模块
- `robot_deploy/model/activevln.py`

### `ActiveVLNOpenAIModel`
OpenAI-compatible 推理客户端。

关键方法：
- `infer(request_data)`: 调用 `/chat/completions`
- `_resolve_model_name()`: 调用 `/models` 自动选择模型 id
- `_build_payload(...)`: 构造 system + user 多模态消息
- `_try_encode_image(image)`: 将 PIL/路径/ndarray 转为 base64 JPEG

关键配置参数：
- `base_url`
- `action_space` (`r2r`/`rxr`)
- `max_tokens`, `temperature`, `top_p`, `timeout_sec`

## 4. Action 解析侧

### 模块
- `robot_deploy/interface/action_interface.py`

### `ActiveVLNActionInterface`
把模型文本解析为离散动作，并映射为速度命令。

关键方法：
- `parse_model_response(...)`: 文本 -> `NavigationAction[]`
- `to_motion_commands(...)`: 动作 -> `MotionCommand[]`

解析规则：
- 支持 `r2r` 与 `rxr` 两种动作步长
- 解析 `forward/left/right/stop`
- 自动量化到允许步长（例如 25/50/75cm）
- 对异常输出回退为 `stop`

## 5. Runtime 总控

### 模块
- `robot_deploy/runtime/controller.py`

### `RuntimeController`
单机器人单进程总控。

关键方法：
- `startup(endpoint)`: 建连 + 站立
- `run_once(request_data)`: 推理 -> 解析 -> 安全限幅 -> 执行
- `shutdown()`: 停止、趴下、关闭资源

安全机制：
- 命令限幅（速度、角速度、持续时长）
- 异常时急停（可配置）
- 单轮最大动作条数限制

## 6. Scripts

### `scripts/run_single_step.py`
单次闭环入口。

流程：
1. 读取配置
2. 从 RTSP 获取一帧
3. 构造 `NavigationRequest`
4. 调用 `RuntimeController.run_once`

### `scripts/test_robot_camera.py`
相机连通与可视化测试。

### `scripts/test_robot_motion.py`
基础动作测试（连接、站立、前进、转向、停止）。

### `scripts/start_model_service.py`
按 `configs/model_service.json` 启动 vLLM。

### `scripts/test_model_service_infer.py`
本地图片 + 指令 的单次推理测试，输出：
- 模型原始文本
- 解析后的动作列表

## 7. 配置文件

### `configs/default.json`
机器人、相机、runtime 安全参数、单步请求。

### `configs/model_service.json`
vLLM 服务参数（模型路径、端口、多模态限制、长度限制等）。
