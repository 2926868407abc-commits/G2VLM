#!/bin/bash
# ============================================================
# G2VLM 部署脚本 - 精简版
# 适用环境:已选 PyTorch 2.5.1 / Python 3.12 / CUDA 12.4 镜像
# 适用 GPU:A800-80GB(也兼容 4090/A100/V100 等 Ampere/Ada 卡)
# ============================================================

set -e

# ---------- 0. 配置 ----------
# 数据盘路径(AutoDL 一般是 /root/autodl-tmp,其他平台改这里)
DATA_DIR="/root/autodl-tmp"

# HF 国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# ---------- 1. 检查环境 ----------
echo "=========================================="
echo "Step 1: 环境检查"
echo "=========================================="
nvidia-smi
echo ""
echo "数据盘空间:"
df -h "$DATA_DIR"
echo ""

# 验证镜像自带 PyTorch 能用
echo "镜像自带 PyTorch:"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# ---------- 2. Clone 代码 ----------
echo ""
echo "=========================================="
echo "Step 2: Clone 代码"
echo "=========================================="
cd "$DATA_DIR"
if [ -d "G2VLM" ]; then
    echo "⚠️  G2VLM 已存在,跳过"
else
    git clone https://github.com/InternRobotics/G2VLM
fi
cd G2VLM

# ---------- 3. 建 Python 3.10 环境(装到数据盘)----------
echo ""
echo "=========================================="
echo "Step 3: 建 Python 3.10 conda 环境"
echo "=========================================="
ENV_PATH="$DATA_DIR/envs/g2vlm"

source $(conda info --base)/etc/profile.d/conda.sh

if [ -d "$ENV_PATH" ]; then
    echo "⚠️  环境已存在,直接激活"
else
    conda create -p "$ENV_PATH" python=3.10 -y
fi
conda activate "$ENV_PATH"
echo "Python 路径:$(which python)"
python --version

# ---------- 4. 装 PyTorch 2.5.1 (cu121,兼容 CUDA 12.4 driver) ----------
echo ""
echo "=========================================="
echo "Step 4: 装 PyTorch 到新 conda 环境"
echo "=========================================="
# 注意:虽然系统 Python 3.12 已经有 PyTorch,但我们在 Python 3.10 conda env 里需要重装一次
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('PyTorch in conda env:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# ---------- 5. 装项目依赖 ----------
echo ""
echo "=========================================="
echo "Step 5: 装依赖"
echo "=========================================="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "没找到 requirements.txt,装常见依赖"
    pip install transformers accelerate einops pillow numpy opencv-python \
                huggingface_hub safetensors plyfile trimesh
fi
pip install -U "huggingface_hub[cli]"

# ---------- 6. 下模型(5-8GB,慢慢等) ----------
echo ""
echo "=========================================="
echo "Step 6: 下载模型权重"
echo "=========================================="
CKPT_DIR="$DATA_DIR/G2VLM/checkpoints/G2VLM-2B-MoT"
if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR 2>/dev/null)" ]; then
    echo "⚠️  模型已存在,跳过"
else
    huggingface-cli download InternRobotics/G2VLM-2B-MoT \
        --local-dir "$CKPT_DIR" \
        --local-dir-use-symlinks False
fi
ls -lh "$CKPT_DIR"

# ---------- 7. 跑 demo ----------
echo ""
echo "=========================================="
echo "Step 7: 跑 demo"
echo "=========================================="
cd "$DATA_DIR/G2VLM"
python inference_chat.py

echo ""
echo "=========================================="
echo "🎉 完成!结果:$DATA_DIR/G2VLM/examples/result.ply"
echo "=========================================="
echo ""
echo "下次进来跑:"
echo "  conda activate $ENV_PATH"
echo "  cd $DATA_DIR/G2VLM"
echo "  python inference_chat.py --image_path /path --question \"...\""
