#!/usr/bin/env bash
# =============================================================================
# cue_prod v2 (HIRES) ALL-IN-ONE preprocess across 4 nodes (32xH200):
#   video latents @1280x704 + Gemma text conditions + audio latents, in one pass.
# Unlike the seedance_video_v0 v2 run (which symlinked conditions/audio from v1),
# cue_prod is a NEW dataset -> everything is computed fresh via process_dataset.py.
# Embarrassingly parallel: each node processes a DISJOINT 1/4 shard with its own
# local 8-GPU accelerate; all outputs land in the same OUT_DIR keyed by media_path.
# Submit ONCE:  sbatch run_preprocess_cueprod_4node.sbatch
# =============================================================================
#SBATCH --job-name=prep_cue
#SBATCH --partition=all
#SBATCH --nodelist=gpu-h200-02,gpu-h200-03,gpu-h200-23,gpu-h200-35
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=160
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/home/zheng/seedance_video_v0/runs/logs/slurm_%x_%j.out

set -euo pipefail

export REPO=/home/zheng/ltx_seedance_videos
export MODEL_PATH=/home/zheng/videogen/checkpoints/LTX-2.3/ltx-2.3-22b-dev.safetensors
export TEXT_ENCODER=/home/zheng/videogen/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
export ROOT=/home/zheng/seedance_cue_prod
export OUT_DIR=/home/zheng/seedance_cue_prod/.precomputed_ti2v_16x9_v2
# 1280x704 at every native 24fps duration present (W x H x F) — SAME buckets as seedance v2.
export BUCKETS="1280x704x97;1280x704x121;1280x704x145;1280x704x193;1280x704x217;1280x704x241;1280x704x265;1280x704x289;1280x704x313;1280x704x337;1280x704x361"

# ffmpeg shim so torchaudio/torchcodec work in the uv venv (--with-audio needs it) — on every node.
export VENV_DIR="${REPO}/.venv"
export LD_LIBRARY_PATH="${VENV_DIR}/ffmpeg-shim:${VENV_DIR}/lib/python3.13/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

echo ">> cue_prod all-in-one preprocess | nodes=${SLURM_JOB_NODELIST} | out=${OUT_DIR}"

# One task per node; $SLURM_NODEID (0..3) selects the shard. No cross-node comm.
srun --kill-on-bad-exit=1 bash -c '
  cd "${REPO}/packages/ltx-trainer"
  SHARD=${ROOT}/train_ti2v.shard${SLURM_NODEID}of4.jsonl
  echo "[node $SLURM_NODEID] $(hostname) -> shard ${SHARD} ($(wc -l < "$SHARD") clips)"
  PYTHONUNBUFFERED=1 uv run --no-sync accelerate launch --num_processes 8 --multi_gpu \
    scripts/process_dataset.py "$SHARD" \
      --resolution-buckets "$BUCKETS" \
      --model-path "$MODEL_PATH" \
      --text-encoder-path "$TEXT_ENCODER" \
      --output-dir "$OUT_DIR" \
      --caption-column caption \
      --video-column media_path \
      --batch-size 1 \
      --vae-tiling \
      --with-audio
'

echo ">> cue_prod all-in-one preprocess finished -> ${OUT_DIR}"
