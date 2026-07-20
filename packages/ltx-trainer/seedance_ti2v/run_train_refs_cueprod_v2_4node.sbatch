#!/usr/bin/env bash
# =============================================================================
# cue_prod v2 (HIRES) reference-image FULL fine-tune across 4 nodes (32xH200) on the
# LARGE dataset (~13.4k clips, ~5.4x the seedance_video_v0 run). FSDP/HSDP, global
# batch 32, lr 2e-5. Same recipe as the seedance v2 4-node run; only the data root
# differs (cue_prod precompute). Config ltx2.3_reference_cueprod_v2.yaml. Submit ONCE:
#   sbatch run_train_refs_cueprod_v2_4node.sbatch
# =============================================================================
#SBATCH --job-name=train_cue_4n
#SBATCH --partition=all
#SBATCH --nodelist=gpu-h200-02,gpu-h200-03,gpu-h200-23,gpu-h200-35
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=160
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --output=/home/zheng/seedance_video_v0/runs/logs/slurm_%x_%j.out

set -euo pipefail

REPO=/home/zheng/ltx_seedance_videos
CONFIG="${CONFIG:-seedance_ti2v/ltx2.3_reference_cueprod_v2.yaml}"
NUM_NODES=4
GPUS_PER_NODE=8
NUM_PROCS=$((NUM_NODES * GPUS_PER_NODE))   # 32

# Rendezvous: first node in the allocation is the FSDP main process.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29507}"   # distinct from other runs' ports

# NCCL over the RoCE fabric (roce_rail1..8 = 8x 400Gbit RoCEv2; GID index 3 = IPv4).
# bond0 carries only the TCP bootstrap. CROSS_NIC OFF: this node set is index-aligned.
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="roce_rail1,roce_rail2,roce_rail3,roce_rail4,roce_rail5,roce_rail6,roce_rail7,roce_rail8"
export NCCL_IB_GID_INDEX="3"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ffmpeg shim so torchaudio/torchcodec work in the uv venv (audio strategy needs it).
VENV_DIR="${VENV_DIR:-${REPO}/.venv}"
export LD_LIBRARY_PATH="${VENV_DIR}/ffmpeg-shim:${VENV_DIR}/lib/python3.13/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

echo ">> cue_prod v2 4-node FULL FT (large dataset) | nodes=${SLURM_JOB_NODELIST} | global_batch=${NUM_PROCS}"
echo ">> master=${MASTER_ADDR}:${MASTER_PORT} | config=${CONFIG}"

# One accelerate launcher per node; $SLURM_NODEID (0..3) is the FSDP machine_rank.
srun --kill-on-bad-exit=1 bash -c '
  cd '"${REPO}/packages/ltx-trainer"'
  echo "[node $SLURM_NODEID / '"${NUM_NODES}"'] $(hostname) starting"
  PYTHONUNBUFFERED=1 uv run --no-sync accelerate launch \
    --config_file configs/accelerate/fsdp_hsdp.yaml \
    --num_machines '"${NUM_NODES}"' \
    --num_processes '"${NUM_PROCS}"' \
    --machine_rank $SLURM_NODEID \
    --main_process_ip '"${MASTER_ADDR}"' \
    --main_process_port '"${MASTER_PORT}"' \
    scripts/train.py '"${CONFIG}"' --disable-progress-bars
'

echo ">> cue_prod v2 4-node finished."
