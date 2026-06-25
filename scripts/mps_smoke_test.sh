#!/usr/bin/env bash
#
# Mac (MPS) 冒烟测试：用玩具规模模型 + 合成数据，把
#   pretrain (训练→存权重)  →  eval_llm (加载→推理)
# 整条链路在 MPS 上真实跑一遍，验证设备/混合精度/checkpoint 适配没问题。
#
# 不需要下载任何数据集，几秒钟跑完。输出是乱码属正常——
# 这里只验证「能跑通」，不验证生成质量。
#
# 用法：
#   bash scripts/mps_smoke_test.sh
#
# 前置：仓库根目录有 .venv，且 torch 支持 MPS（torch.backends.mps.is_available()）。

set -euo pipefail

# 切到仓库根目录（无论从哪里调用脚本）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
DATA="$ROOT/dataset/pretrain_smoke.jsonl"
WEIGHT="pretrain_smoke"

# 退出时清理临时产物（数据、权重、续训文件）
cleanup() {
    rm -f "$DATA"
    rm -rf "$ROOT/out" "$ROOT/checkpoints"
}
trap cleanup EXIT

echo "==> [0/3] 检查 MPS 可用性"
"$PY" -c "import torch; assert torch.backends.mps.is_available(), 'MPS 不可用'; print('    MPS OK, torch', torch.__version__)"

echo "==> [1/3] 生成合成数据 (128 行) -> $DATA"
"$PY" - <<'PYEOF'
import json, os
texts = [
    "人工智能是研究如何让计算机模拟人类智能的科学。",
    "机器学习是人工智能的一个重要分支。",
    "深度学习使用多层神经网络来学习数据特征。",
    "自然语言处理让机器能够理解和生成人类语言。",
    "今天天气很好，适合出去散步和运动。",
    "苹果和香蕉都是常见又健康的水果。",
    "北京是中国的首都，历史文化非常悠久。",
    "学习编程需要不断地练习和动手实践。",
]
path = os.path.join("dataset", "pretrain_smoke.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for i in range(128):
        f.write(json.dumps({"text": texts[i % len(texts)]}, ensure_ascii=False) + "\n")
PYEOF

echo "==> [2/3] 在 MPS 上训练玩具模型 (1.26M, 16 步)"
# --device/--dtype 不传，走默认逻辑自动选中 mps + float32
# --num_workers 0 避免 Mac 上 DataLoader 多进程问题
# PYTORCH_ENABLE_MPS_FALLBACK=1 让 MPS 缺失的算子回退 CPU
( cd trainer && PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" train_pretrain.py \
    --data_path "$DATA" \
    --hidden_size 128 --num_hidden_layers 2 --max_seq_len 128 \
    --batch_size 8 --num_workers 0 --accumulation_steps 1 \
    --epochs 1 --log_interval 2 --save_interval 1000 \
    --save_weight "$WEIGHT" )

echo "==> [3/3] 加载权重跑推理 (eval_llm --auto, 默认 cpu)"
"$PY" eval_llm.py --auto --weight "$WEIGHT" \
    --hidden_size 128 --num_hidden_layers 2 --max_new_tokens 20

echo "==> 冒烟测试通过 ✅  (临时产物将自动清理)"
