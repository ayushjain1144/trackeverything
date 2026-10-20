#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

usage() {
    echo "Usage: $0 <slurm_job_id> <hydra_override> [<hydra_override> ...]"
    echo
    echo "Example:"
    echo "  $0 123456 model.no_window_overlap=false train.lr=1e-5"
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

JOB_ID="$1"
shift

if [[ "$JOB_ID" == *"/"* ]]; then
    echo "Invalid Slurm job id: $JOB_ID" >&2
    exit 1
fi

OVERRIDE_DIR="${TAPIP3D_SLURM_OVERRIDES_DIR:-output/slurm_overrides}"
ARGS_FILE="${OVERRIDE_DIR}/${JOB_ID}.args"

mkdir -p "$OVERRIDE_DIR"
: > "$ARGS_FILE"

for arg in "$@"; do
    printf '%s\n' "$arg" >> "$ARGS_FILE"
done

echo "Wrote extra Hydra args to $ARGS_FILE"
echo "Contents:"
while IFS= read -r line || [[ -n "$line" ]]; do
    printf '  %s\n' "$line"
done < "$ARGS_FILE"
