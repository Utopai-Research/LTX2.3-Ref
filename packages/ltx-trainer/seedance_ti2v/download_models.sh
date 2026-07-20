#!/usr/bin/env bash
# Download the two checkpoints required to fine-tune LTX-2.3 (TI2V LoRA).
#   1) LTX-2.3 base "dev" transformer + VAEs (single .safetensors, ~44 GB bf16)
#   2) Gemma-3 text encoder directory (~24 GB)
#
# Both repos are gated on Hugging Face. Accept the licenses in a browser and make
# sure you are authenticated first, e.g.:
#     export HF_TOKEN=hf_xxx                 # or: huggingface-cli login
# then run this script.
set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-/home/zheng/videogen/checkpoints}"
PY="${PY:-/home/zheng/ltx_seedance_videos/.venv/bin/python}"

LTX_DIR="${CKPT_ROOT}/LTX-2.3"
GEMMA_DIR="${CKPT_ROOT}/gemma-3-12b-it-qat-q4_0-unquantized"

echo ">> LTX-2.3 base checkpoint  -> ${LTX_DIR}"
echo ">> Gemma-3 text encoder     -> ${GEMMA_DIR}"
mkdir -p "${LTX_DIR}" "${GEMMA_DIR}"

"${PY}" - "$LTX_DIR" "$GEMMA_DIR" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download

ltx_dir, gemma_dir = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

print("== Downloading LTX-2.3 base checkpoint (ltx-2.3-22b-dev.safetensors) ==", flush=True)
hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-22b-dev.safetensors",
    local_dir=ltx_dir,
    token=token,
)

print("== Downloading Gemma-3 text encoder (full repo) ==", flush=True)
snapshot_download(
    repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
    local_dir=gemma_dir,
    token=token,
    # weights + tokenizer + config; skip the original gguf to save space
    ignore_patterns=["*.gguf"],
)
print("== Done ==")
PY

echo
echo "Downloaded:"
echo "  model_path        = ${LTX_DIR}/ltx-2.3-22b-dev.safetensors"
echo "  text_encoder_path = ${GEMMA_DIR}"
echo "These match the paths in ltx2.3_ti2v_lora.yaml."
