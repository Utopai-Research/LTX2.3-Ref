#!/usr/bin/env bash
# =============================================================================
# cue_prod v2 REFERENCE-IMAGE latents (aspect-bucketed: 16:9->1376x768,
# 9:16->768x1376, other->768x768). Single node (8xH200) — refs are light and this
# runs CONCURRENTLY with the 4-node all-in-one preprocess (different node, same
# OUT_DIR, different subdir reference_image_latents/ -> no collision).
# Submit:  sbatch run_preprocess_cueprod_refs.sbatch
# =============================================================================
#SBATCH --job-name=prep_cue_refs
#SBATCH --partition=all
#SBATCH --nodelist=gpu-h200-48
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=160
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/home/zheng/seedance_video_v0/runs/logs/slurm_%x_%j.out

set -euo pipefail

REPO=/home/zheng/ltx_seedance_videos
cd "${REPO}/packages/ltx-trainer"

MODEL_PATH=/home/zheng/videogen/checkpoints/LTX-2.3/ltx-2.3-22b-dev.safetensors
ROOT=/home/zheng/seedance_cue_prod
OUT_DIR=${ROOT}/.precomputed_ti2v_16x9_v2/reference_image_latents

echo ">> cue_prod ref-image latents | $(hostname) | out=${OUT_DIR}"

# dataset.jsonl has ref_images + video; restrict to the train split via --manifest.
PYTHONUNBUFFERED=1 uv run --no-sync accelerate launch --num_processes 8 --multi_gpu \
  scripts/process_reference_images.py "${ROOT}/dataset.jsonl" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUT_DIR}" \
    --manifest "${ROOT}/train_ti2v.jsonl" \
    --resize-mode crop

echo ">> cue_prod ref-image latents finished -> ${OUT_DIR}"
