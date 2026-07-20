#!/usr/bin/env bash
# Preprocess the 16:9 train split into VAE latents + Gemma text embeddings + AUDIO latents.
# Each clip is center-cropped to 768x448 and kept at its NATIVE length (4-15s):
# the bucket matcher assigns every clip to the largest frame-bucket <= its frame
# count, so a 15s clip -> 361 frames, a 12s clip -> 289, an 8s clip -> 193, etc.
# --with-audio extracts each clip's own audio track and encodes it (stereo, 44.1kHz);
# clips without an audio track get video+text only and are dropped from AV training.
# Runs sharded across all 8 H200s via accelerate. Resumable: existing .pt are skipped.
#
# Logs (stdout+stderr) are tee'd to a timestamped file under runs/logs/.
set -euo pipefail

cd "$(dirname "$0")/.."   # -> packages/ltx-trainer (so scripts/ resolves)

# FFmpeg shared libs for torchaudio (audio extraction needs torchcodec+ffmpeg). Reuses the
# ffmpeg PyAV already bundles in the venv via a SONAME shim. See seedance_ti2v/README.md.
VENV_DIR="${VENV_DIR:-/home/zheng/ltx_seedance_videos/.venv}"
export LD_LIBRARY_PATH="${VENV_DIR}/ffmpeg-shim:${VENV_DIR}/lib/python3.13/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

CKPT_ROOT="${CKPT_ROOT:-/home/zheng/videogen/checkpoints}"
MODEL_PATH="${MODEL_PATH:-${CKPT_ROOT}/LTX-2.3/ltx-2.3-22b-dev.safetensors}"
TEXT_ENCODER="${TEXT_ENCODER:-${CKPT_ROOT}/gemma-3-12b-it-qat-q4_0-unquantized}"

DATASET="${DATASET:-/home/zheng/seedance_video_v0/train_ti2v.jsonl}"
OUT_DIR="${OUT_DIR:-/home/zheng/seedance_video_v0/.precomputed_ti2v_16x9}"
NUM_GPUS="${NUM_GPUS:-8}"

LOG_DIR="${LOG_DIR:-/home/zheng/seedance_video_v0/runs/logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log"

# One bucket per native 24fps duration present in the 16:9 set (W x H x F).
BUCKETS="768x448x97;768x448x121;768x448x145;768x448x193;768x448x217;768x448x241;768x448x265;768x448x289;768x448x313;768x448x337;768x448x361"

echo ">> dataset : ${DATASET}"
echo ">> model   : ${MODEL_PATH}"
echo ">> gemma   : ${TEXT_ENCODER}"
echo ">> output  : ${OUT_DIR}"
echo ">> buckets : ${BUCKETS}"
echo ">> audio   : enabled (--with-audio)"
echo ">> log     : ${LOG}"

PYTHONUNBUFFERED=1 uv run --no-sync accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
  scripts/process_dataset.py "${DATASET}" \
    --resolution-buckets "${BUCKETS}" \
    --model-path "${MODEL_PATH}" \
    --text-encoder-path "${TEXT_ENCODER}" \
    --output-dir "${OUT_DIR}" \
    --caption-column caption \
    --video-column media_path \
    --batch-size 1 \
    --vae-tiling \
    --with-audio 2>&1 | tee "${LOG}"

echo
echo "Preprocessing complete. latents/ + conditions/ + audio_latents/ under: ${OUT_DIR}"
echo "Full log: ${LOG}"
