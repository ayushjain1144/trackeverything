
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


configure_shared() {
    export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    export OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${NAME}_${TIMESTAMP}"
}

configure_slurm() {
    export DEBUGRUN=0;
    export NUM_GPUS=1;
    export NUM_MACHINES=1;
    export SLURM=0;
    export USE_SLURM=0;
    export PREFIX="bash";
    export PREFIX_ARGS="";
    export NAME="local"
    export PREFIX_ARGS=()
    local PARTITION="general"
    local QOS="normal"
    local ARRAY_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --partition=*)
                PARTITION="${1#*=}"
                if [[ "$PARTITION" == "array" ]]; then
                    QOS="array_qos"
                    ARRAY_ARGS="--array=0-0%1"
                elif [[ "$PARTITION" == "preempt" ]]; then
                    QOS="preempt_qos"
                elif [[ "$PARTITION" == "debug" ]]; then
                    QOS="debug_qos"
                elif [[ "$PARTITION" == "katef" ]]; then
                    QOS="katef_qos"
                fi
                shift
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done
    
    export NUM_GPUS=${NUM_GPUS:-8}
    export NUM_MACHINES=${NUM_MACHINES:-1}
    export NAME=${NAME:-"alltracker"}
    export COMMENT=${COMMENT:-"alltracker"}
    export NUM_GPUS=$NUM_GPUS
    export NUM_MACHINES=$NUM_MACHINES
    export USE_SLURM=1
    export PREFIX="sbatch"
    # Do NOT set NCCL_SHM_DISABLE=1: with P2P disabled, SHM is the normal
    # single-node path; disabling it forces a slow socket transport.
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export NCCL_DEBUG=WARN
    unset NCCL_SHM_DISABLE
    export TIME=${TIME:-"1-0:00"}
    
    
    PREFIX_ARGS=(
        "--qos=$QOS"
        "--job-name=$NAME"
        "--partition=$PARTITION"
        $([[ -n $ARRAY_ARGS ]] && echo "$ARRAY_ARGS")  # Removed extra quotes
        $(hostname | grep -q "babel" && echo "--account=kfragki2")
        "--nodes=$NUM_MACHINES"
        "--gpus-per-node=$NUM_GPUS"
        "--ntasks-per-node=$NUM_GPUS"
        "--time=$TIME"
        "--comment=$COMMENT"
    )
    # Remove empty elements from array
    PREFIX_ARGS=("${PREFIX_ARGS[@]/#empty/}")
    export PREFIX_ARGS

    echo "Args: $PREFIX" "${PREFIX_ARGS[@]}"
    configure_shared
}

configure_local() {
    export DEBUGRUN=0;
    export TAPIP3D_LOCAL_DEBUG=1;
    export NUM_GPUS=1;
    export NUM_MACHINES=1;
    export SLURM=0;
    export USE_SLURM=0;
    export PREFIX="bash";
    export PREFIX_ARGS="";
    export NAME="local"
    export PREFIX_ARGS=()
    configure_shared
}

clear_exports() {
    unset SLURM; unset NAME; unset EVAL_PERIOD; unset NUM_GPUS; unset NUM_MACHINES; unset PREFIX; unset DEBUGRUN; unset TAPIP3D_LOCAL_DEBUG; unset PREFIX_ARGS;
}

# DINOv3 weights are gated (the torch.hub download returns HTTP 403), so the
# dinov3 encoder loads from a local checkpoint via this env var instead. Matches
# dinov3_name=dinov3_vits16 in configs/model/encoder/dinov3.yaml. Override by
# exporting DINOV3_WEIGHTS_PATH before sourcing this file.
export DINOV3_WEIGHTS_PATH="${DINOV3_WEIGHTS_PATH:-/data/user_data/ayushj2/dinov3_pretrain_vits16.pth}"
