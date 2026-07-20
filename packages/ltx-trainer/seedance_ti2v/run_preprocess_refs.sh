#!/usr/bin/env bash
# Preprocess per-clip reference images (Seedance ref_images) -> reference_image_latents (exp2).
# Encodes each ref image (1-9/clip) as a 1-frame VAE latent at REF_RES (default 512x512, center-crop,
# since refs are mixed AR). Keyed by the clip's video path so it pairs with latents/conditions/audio.
# Sharded over 8 GPUs; resumable. NO ffmpeg/audio needed here.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> packages/ltx-trainer

CKPT_ROOT="${CKPT_ROOT:-/home/zheng/videogen/checkpoints}"
MODEL_PATH="${MODEL_PATH:-${CKPT_ROOT}/LTX-2.3/ltx-2.3-22b-dev.safetensors}"
DATASET="${DATASET:-/home/zheng/seedance_video_v0/dataset.jsonl}"        # has ref_images + video
MANIFEST="${MANIFEST:-/home/zheng/seedance_video_v0/train_ti2v.jsonl}"   # restrict to train clips
OUT_DIR="${OUT_DIR:-/home/zheng/seedance_video_v0/.precomputed_ti2v_16x9/reference_image_latents}"
# Resolution is AUTOMATIC per-image aspect bucketing now (16:9->1376x768, 9:16->768x1376, other->768x768).
# RESIZE_MODE = how each image fits its bucket: crop (cover+center-crop, ~0 loss for matched aspect) | pad.
RESIZE_MODE="${RESIZE_MODE:-crop}"
NUM_GPUS="${NUM_GPUS:-8}"
LOG_DIR="${LOG_DIR:-/home/zheng/seedance_video_v0/runs/logs}"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/preprocess_refs_$(date +%Y%m%d_%H%M%S).log"

echo ">> dataset : ${DATASET} (manifest ${MANIFEST})"
echo ">> output  : ${OUT_DIR} (aspect-bucketed, ${RESIZE_MODE})"
echo ">> log     : ${LOG}"

PYTHONUNBUFFERED=1 uv run --no-sync accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
  scripts/process_reference_images.py "${DATASET}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUT_DIR}" \
    --manifest "${MANIFEST}" \
    --resize-mode "${RESIZE_MODE}" 2>&1 | tee "${LOG}"

echo ">> done -> ${OUT_DIR}  (log: ${LOG})"
