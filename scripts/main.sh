#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=8
#SBATCH -o output/logs/%x_%j_%n.out
#SBATCH --constraint=VRAM_48GB|VRAM_96GB|VRAM_80GB|VRAM_141GB|A100_80G
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --account=kfragki2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ayushjain1144@gmail.com


conda init
source ~/.bashrc
conda activate /home/ayushj2/miniconda3/envs/alltracker

get_cuda_device_count() {
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    else
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
    fi
}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(dirname "$PWD"):$PYTHONPATH"
# Run from the repo root; the repo directory is the Python package.
PKG="$(basename "$PWD")"
echo "PYTHONPATH=$PYTHONPATH"

# Never inherit NCCL_SHM_DISABLE=1 from the submit shell — with P2P off, SHM
# is the normal single-node NCCL path.
unset NCCL_SHM_DISABLE

export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=WARN

export HF_HOME="/data/hf_cache/"
export TMPDIR=/tmp/$USER
export TORCH_HOME=/tmp/$USER/torch
export CUDA_CACHE_PATH=/tmp/$USER/cuda
export TRITON_CACHE_DIR=/tmp/$USER/triton
export HF_HOME=/tmp/$USER/hf
# Shared DINOv3 weights (fbaipublicfiles often returns HTTP 403 from compute nodes).
# Dinov3Encoder reads this and skips torch.hub download when the file exists.
export DINOV3_WEIGHTS_PATH="${DINOV3_WEIGHTS_PATH:-/data/user_data/ayushj2/dinov3_pretrain_vits16.pth}"
mkdir -p $TMPDIR $TORCH_HOME $CUDA_CACHE_PATH $TRITON_CACHE_DIR $HF_HOME
# Prefetch into node-local TORCH_HOME under the URL basename so any residual
# hub lookups also hit cache if DINOV3_WEIGHTS_PATH is unset on a worker.
if [[ -f "$DINOV3_WEIGHTS_PATH" ]]; then
  mkdir -p "$TORCH_HOME/hub/checkpoints"
  _dino_hub_name="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
  if [[ ! -f "$TORCH_HOME/hub/checkpoints/$_dino_hub_name" ]]; then
    cp -n "$DINOV3_WEIGHTS_PATH" "$TORCH_HOME/hub/checkpoints/$_dino_hub_name" || true
  fi
fi

NUM_GPUS=${NUM_GPUS:-$(get_cuda_device_count)}
NUM_WORKERS=${NUM_WORKERS:-12}
NUM_MACHINES=${NUM_MACHINES:-1}

USE_SLURM=${USE_SLURM:-0}
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo "MASTER_PORT="$MASTER_PORT

EXTRA_HYDRA_ARGS=()

add_extra_hydra_args_file() {
    local args_file="$1"
    if [[ -z "$args_file" || ! -f "$args_file" ]]; then
        return 0
    fi

    echo "Loading extra Hydra args from $args_file"
    local line trimmed
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        trimmed="${line#"${line%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"

        if [[ -z "$trimmed" || "${trimmed:0:1}" == "#" ]]; then
            continue
        fi

        EXTRA_HYDRA_ARGS+=("$trimmed")
    done < "$args_file"
}

load_extra_hydra_args() {
    if [[ -n "${TAPIP3D_EXTRA_ARGS_FILE:-}" ]]; then
        add_extra_hydra_args_file "$TAPIP3D_EXTRA_ARGS_FILE"
    fi

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        add_extra_hydra_args_file "output/slurm_overrides/${SLURM_JOB_ID}.args"
    fi

    if [[ ${#EXTRA_HYDRA_ARGS[@]} -gt 0 ]]; then
        echo "Extra Hydra args:"
        printf '  %q\n' "${EXTRA_HYDRA_ARGS[@]}"
    fi
}

if [[ "$USE_SLURM" -eq 1 ]]; then
    SLURM_ARG="launcher=slurm"
    CMD="srun accelerate"

    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo "MASTER_ADDR="$MASTER_ADDR

    export RANK=${SLURM_PROCID}
    echo "NODES="$NODES
    echo "RANK="$RANK
    echo "GPUS_PER_NODE="$GPUS_PER_NODE
    echo "SLURM_NODEID="$SLURM_NODEID
else
    SLURM_ARG=""
    CMD="accelerate"
fi

load_extra_hydra_args

if [ $NUM_GPUS -gt 1 ]; then
    $CMD launch --main_process_port $MASTER_PORT --gpu_ids all --multi_gpu --num_machines $NUM_MACHINES \
        --num_processes $NUM_GPUS -m "$PKG.main_point_track" --config-name track_everything slurm_job_id=$SLURM_JOB_ID \
        "$@" "${EXTRA_HYDRA_ARGS[@]}"
else
    $CMD launch --main_process_port $MASTER_PORT --gpu_ids all --num_machines $NUM_MACHINES \
        --num_processes $NUM_GPUS -m "$PKG.main_point_track" --config-name track_everything slurm_job_id=$SLURM_JOB_ID \
        "$@" "${EXTRA_HYDRA_ARGS[@]}"
fi

if [[ $? == 124 ]]; then
  scontrol requeue $SLURM_JOB_ID
fi
