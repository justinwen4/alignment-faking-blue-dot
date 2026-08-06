#!/bin/bash
# Setup + start vLLM for Condition 1 adapter on a 2x80GB RunPod box.
set -euo pipefail
cd /workspace
export HF_HOME=/workspace/hf
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
mkdir -p "$HF_HOME" /workspace/models/condition_1_adapter /workspace/logs

echo "[1/4] Install vllm + together..."
pip install -U pip
pip install -U "vllm" together zstandard huggingface_hub

echo "[2/4] Ensure Condition 1 adapter present..."
python3 <<'PY'
import io
import tarfile
from pathlib import Path

import zstandard as zstd
from together import Together

out = Path("/workspace/models/condition_1_adapter")
marker = out / "adapter_model.safetensors"
if marker.exists():
    print("Adapter already present:", marker, marker.stat().st_size)
else:
    print("Downloading adapter checkpoint from Together...")
    resp = Together().fine_tuning.content(ft_id="ft-f1baf12a-cba0", checkpoint="adapter")
    data = resp.read() if hasattr(resp, "read") else resp.content
    print(f"compressed bytes={len(data)}")
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(data)) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(path=out)
    assert marker.exists(), "adapter_model.safetensors missing after extract"
    print("Extracted OK:", marker, marker.stat().st_size)

# Point PEFT config at the public HF base used by vLLM.
import json

cfg_path = out / "adapter_config.json"
cfg = json.loads(cfg_path.read_text())
cfg["base_model_name_or_path"] = "meta-llama/Llama-3.3-70B-Instruct"
cfg_path.write_text(json.dumps(cfg, indent=2))
print("Patched adapter base -> meta-llama/Llama-3.3-70B-Instruct")
PY

echo "[3/4] HF login..."
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential || true

echo "[4/4] Start vLLM (TP=2, max-model-len=16000, kv fp8)..."
if pgrep -f "vllm serve" >/dev/null 2>&1; then
  echo "vLLM already running"
else
  nohup vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --enable-lora \
    --lora-modules condition1=/workspace/models/condition_1_adapter \
    --max-lora-rank 64 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 16000 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 2 \
    --port 8000 \
    > /workspace/logs/vllm.log 2>&1 &
  echo "vLLM pid $!"
fi

echo "Waiting for /v1/models ..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/v1/models >/tmp/models.json 2>/dev/null; then
    echo "READY"
    cat /tmp/models.json
    exit 0
  fi
  sleep 10
done
echo "TIMEOUT waiting for vLLM; last log lines:"
tail -n 80 /workspace/logs/vllm.log || true
exit 1
