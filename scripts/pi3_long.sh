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
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus-per-node=8
#SBATCH -o output/logs/%x_%j_%n.out
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --account=kfragki2

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
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(dirname "$PWD"):$PYTHONPATH"
# Run from the repo root; the repo directory is the Python package.
PKG="$(basename "$PWD")"
echo "PYTHONPATH=$PYTHONPATH"

NUM_GPUS=${NUM_GPUS:-$(get_cuda_device_count)}
NUM_MACHINES=${NUM_MACHINES:-1}
USE_SLURM=${USE_SLURM:-0}
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo "MASTER_PORT=$MASTER_PORT"

if [[ "$USE_SLURM" -eq 1 ]]; then
    CMD="srun accelerate"
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo "MASTER_ADDR=$MASTER_ADDR"
    export RANK=${SLURM_PROCID}
else
    CMD="accelerate"
fi

if [ $NUM_GPUS -gt 1 ]; then
    $CMD launch --main_process_port $MASTER_PORT --gpu_ids all --multi_gpu --num_machines $NUM_MACHINES \
        --num_processes $NUM_GPUS -m "$PKG.annotation.pi3_long" "$@"
else
    $CMD launch --main_process_port $MASTER_PORT --gpu_ids all --num_machines $NUM_MACHINES \
        --num_processes $NUM_GPUS -m "$PKG.annotation.pi3_long" "$@"
fi

if [[ $? == 124 ]]; then
    scontrol requeue $SLURM_JOB_ID
fi
