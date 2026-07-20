#!/usr/bin/env bash
# =============================================================================
# v2 VIDEO latents (1280x704) across 4 nodes (32xH200). Preprocessing is
# embarrassingly parallel (per-clip VAE encode, no gradient comm), so this does
# NOT use NCCL/FSDP rendezvous. Instead each node processes a DISJOINT 1/4 shard
# of the dataset (train_ti2v.v2shard{0..3}of4.jsonl, round-robin split) with its
# own local 8-GPU accelerate launch. Outputs all land in the same OUT_DIR keyed
# by media_path, so no collisions.
# Submit ONCE:  sbatch run_preprocess_v2_video_4node.sbatch
# =============================================================================
#SBATCH --job-name=prep_v2_vid
#SBATCH --partition=all
#SBATCH --nodelist=gpu-h200-03,gpu-h200-35,gpu-h200-48,gpu-h200-02
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=160
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/home/zheng/seedance_video_v0/runs/logs/slurm_%x_%j.out

set -euo pipefail

# Exported so srun propagates them to each node; referenced QUOTED on the remote side
# (the BUCKETS string contains ';' which must NOT be split by the remote shell).
export REPO=/home/zheng/ltx_seedance_videos
export MODEL_PATH=/home/zheng/videogen/checkpoints/LTX-2.3/ltx-2.3-22b-dev.safetensors
export OUT_DIR=/home/zheng/seedance_video_v0/.precomputed_ti2v_16x9_v2/latents
export BUCKETS="1280x704x97;1280x704x121;1280x704x145;1280x704x193;1280x704x217;1280x704x241;1280x704x265;1280x704x289;1280x704x313;1280x704x337;1280x704x361"

echo ">> v2 video preprocess | nodes=${SLURM_JOB_NODELIST} | out=${OUT_DIR}"

# One task per node; $SLURM_NODEID (0..3) selects the shard. No cross-node comm.
# Body is single-quoted so all $VARs expand on the REMOTE bash; keep "$BUCKETS" quoted there.
srun --kill-on-bad-exit=1 bash -c '
  cd "${REPO}/packages/ltx-trainer"
  SHARD=/home/zheng/seedance_video_v0/train_ti2v.v2shard${SLURM_NODEID}of4.jsonl
  echo "[node $SLURM_NODEID] $(hostname) -> shard ${SHARD} ($(wc -l < "$SHARD") clips)"
  PYTHONUNBUFFERED=1 uv run --no-sync accelerate launch --num_processes 8 --multi_gpu \
    scripts/process_videos.py "$SHARD" \
      --resolution-buckets "$BUCKETS" \
      --model-path "$MODEL_PATH" \
      --output-dir "$OUT_DIR" \
      --video-column media_path \
      --batch-size 1 \
      --vae-tiling
'

echo ">> v2 video preprocess finished -> ${OUT_DIR}"
