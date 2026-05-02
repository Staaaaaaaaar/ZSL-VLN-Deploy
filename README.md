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

```
# ========== 1. 安装系统级依赖（GStreamer、FFmpeg、GTK等） ==========
sudo apt-get update
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libxvidcore-dev \
    libx264-dev \
    libtbb-dev \
    cmake \
    ninja-build \
    pkg-config \
    git

# ========== 2. 创建并激活 Conda 环境 ==========
conda create -n opencv_gst python=3.9 -y
conda activate opencv_gst

# ========== 3. 在 Conda 环境中安装 NumPy ==========
pip install numpy

# ========== 4. 克隆 OpenCV 源码（不需要 contrib） ==========
mkdir -p ~/opencv_gst_build && cd ~/opencv_gst_build
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.8.0   # 可换成其他版本号
cd ..

# ========== 5. 配置 CMake（禁用所有下载依赖，仅保留视频 I/O 必需模块） ==========
cd opencv
mkdir -p build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python) \
    -D PYTHON3_EXECUTABLE=$(which python) \
    -D PYTHON3_INCLUDE_DIR=$(python -c "from sysconfig import get_path; print(get_path('include'))") \
    -D PYTHON3_PACKAGES_PATH=$(python -c "from sysconfig import get_path; print(get_path('platlib'))") \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D WITH_GSTREAMER=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GTK=ON \
    -D WITH_IPP=OFF \
    -D WITH_ADE=OFF \
    -D WITH_OPENEXR=OFF \
    -D WITH_PROTOBUF=OFF \
    -D BUILD_IPP_IW=OFF \
    -D BUILD_ITT=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_DOCS=OFF \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    ..

# ========== 6. 编译并安装 ==========
make -j$(nproc)
make install

# ========== 7. 验证安装 ==========
python -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())" | grep -A 2 "GStreamer"
```

验证：

```powershell
python -c "import cv2; s=cv2.getBuildInformation(); print('\n'.join([l for l in s.splitlines() if 'Video I/O' in l or 'GStreamer' in l or 'FFMPEG' in l]))"
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

连续闭环 VLN（持续取图 + 多轮推理 + 动作队列执行）可运行：

```powershell
python scripts/run_continuous_vln.py
```

相关预算参数在 `configs/default.json` 的 `runtime_episode` 字段中配置。
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

