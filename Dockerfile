# 庙算平台陆战兵棋AI开发环境
# 基于 Ubuntu 20.04 + Python 3.10

FROM python:3.10-slim

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    vim \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 创建工作目录
WORKDIR /workspace

# 复制 SDK 安装包并安装
COPY land_wargame_sdk/land_wargame_train_env-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/land_wargame_train_env-*.whl \
    && rm /tmp/land_wargame_train_env-*.whl

# 复制项目代码
COPY land_wargame_sdk/ai /workspace/ai
COPY land_wargame_sdk/run_offline_games.py /workspace/

# 复制数据（如果Data已解压）
COPY land_wargame_sdk/Data /workspace/data

# 创建日志目录
RUN mkdir -p /workspace/logs/replays

# 设置 Python 路径
ENV PYTHONPATH=/workspace:$PYTHONPATH

# 默认命令
CMD ["python", "run_offline_games.py"]
