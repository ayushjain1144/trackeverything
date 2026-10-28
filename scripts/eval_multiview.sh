#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Evaluate a checkpoint on a multiview dataset at a chosen number of cameras.
#
#   CKPT=checkpoints/model.pth DATASET=panoptic VIEWS=4 FRAMES=16 \
#     bash scripts/eval_multiview.sh
#
# Query-mode metrics only. Sparse tracking is what makes several cameras fit: a
# window holds model.seq_len * VIEWS images, so the feature cloud grows with the
# camera count. The same flags are used at every view count so the numbers are
# comparable.
set -euo pipefail

CKPT="${CKPT:-checkpoints/model.pth}"
DATASET="${DATASET:-panoptic}"
VIEWS="${VIEWS:-4}"
FRAMES="${FRAMES:-16}"
SEQ_LEN="${SEQ_LEN:-16}"
VOXEL="${VOXEL:-0.02}"
TAG="${TAG:-}"

case "$DATASET" in
  panoptic) DS_KEY=multiview_panoptic_studio_eval ;;
  dexycb)   DS_KEY=multiview_dexycb_eval ;;
  *) echo "DATASET must be panoptic or dexycb, got '$DATASET'" >&2; exit 1 ;;
esac

# views_to_return are positions in the sorted view list, so 0..VIEWS-1.
VIEW_LIST="[$(seq -s, 0 $((VIEWS - 1)))]"

echo "=== $DATASET | $VIEWS view(s) $VIEW_LIST | $FRAMES frames" \
     "| seq_len=$SEQ_LEN | voxel=$VOXEL ==="

NUM_GPUS="${NUM_GPUS:-1}" bash scripts/main.sh \
  experiment_id="multiview_${DATASET}_v${VIEWS}_f${FRAMES}${TAG}" \
  train.checkpoint="$CKPT" \
  train.resume_load_optimizer=false \
  eval_only=true \
  wandb.enable=false \
  model.name=track_everything_multiview \
  model.seq_len="$SEQ_LEN" \
  model.voxel_size="$VOXEL" \
  model.use_only_query_points=true \
  model.sparse_tracking=true \
  eval_datasets="['$DS_KEY']" \
  "test_datasets.$DS_KEY.provider_config.config.views_to_return=$VIEW_LIST" \
  train.eval_modes="['query']" \
  train.visualize_eval_modes="['query']" \
  train.bidirectional_eval=false \
  train.visualize_nbatches=0 \
  ++eval.dataset_overrides.frames_per_sample="$FRAMES" \
  ++eval.dataset_overrides.compute_dense_trajs_3d=false \
  ++eval.dataset_overrides.compute_tmaps=false \
  ++eval.dataset_overrides.num_dense_sample_points=0 \
  ~eval.dataset_overrides.eval_on_key_frames \
  train.num_workers=0 \
  train.prefetch_factor=null \
  "$@"
