#!/usr/bin/env bash
# v2 VIDEO latents at higher resolution: 1280x704 (16:9-ish, both /32 -> latent 40x22).
# v1 was 768x448 (latent 24x14); v2 is ~2.6x the spatial tokens per frame.
#
# Only VIDEO latents are recomputed here. Text conditions (30G) and audio latents (382M)
# are resolution-INDEPENDENT, so the v2 root symlinks them from v1 (see run_preprocess_v2_setup.sh)
# instead of re-encoding. Uses process_videos.py (video-only) rather than process_dataset.py
# (which would wastefully redo the 30G of Gemma text embeddings).
#
# Same native-length frame buckets as v1 (4-15s @ 24fps). Sharded over 8 H200s; resumable.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> packages/ltx-trainer

CKPT_ROOT="${CKPT_ROOT:-/home/zheng/videogen/checkpoints}"
MODEL_PATH="${MODEL_PATH:-${CKPT_ROOT}/LTX-2.3/ltx-2.3-22b-dev.safetensors}"
DATASET="${DATASET:-/home/zheng/seedance_video_v0/train_ti2v.jsonl}"
OUT_DIR="${OUT_DIR:-/home/zheng/seedance_video_v0/.precomputed_ti2v_16x9_v2/latents}"
NUM_GPUS="${NUM_GPUS:-8}"

LOG_DIR="${LOG_DIR:-/home/zheng/seedance_video_v0/runs/logs}"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/preprocess_v2_video_$(date +%Y%m%d_%H%M%S).log"

# 1280x704 at every native 24fps duration present in the 16:9 set (W x H x F).
BUCKETS="1280x704x97;1280x704x121;1280x704x145;1280x704x193;1280x704x217;1280x704x241;1280x704x265;1280x704x289;1280x704x313;1280x704x337;1280x704x361"

echo ">> dataset : ${DATASET}"
echo ">> output  : ${OUT_DIR}"
echo ">> buckets : ${BUCKETS}"
echo ">> log     : ${LOG}"

PYTHONUNBUFFERED=1 uv run --no-sync accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
  scripts/process_videos.py "${DATASET}" \
    --resolution-buckets "${BUCKETS}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUT_DIR}" \
    --video-column media_path \
    --batch-size 1 \
    --vae-tiling 2>&1 | tee "${LOG}"

echo ">> done -> ${OUT_DIR}  (log: ${LOG})"
