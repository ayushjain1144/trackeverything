# Dataset Preparation & Processing

This document explains how to set up the datasets used for training and evaluating TrackEverything, including our pre-processing scripts, precomputed geometry annotations, and the corrections for known annotation defects in Point Odyssey. Please download the datasets following the instructions in each dataset's respective README.

---

## 1. Directory Setup & Path Configuration

All dataset roots are centralized in [`configs/paths.yaml`](../configs/paths.yaml). Every training, evaluation, and annotation config reads these entries via Hydra's `${data_path:<key>}` resolver. 

Edit `configs/paths.yaml` to point each key to your local directory:

```yaml
# configs/paths.yaml
kubric_root: /path/to/kubric
cotracker3_kubric_root: /path/to/CoTracker3_Kubric
pod_root: /path/to/point_odyssey
dynamic_replica_train_root: /path/to/dynamic_replica
dynamic_replica_val_root: /path/to/dynamic_replica
tapvid3d_root: /path/to/tapvid3d
```

---

## 2. Point Odyssey (POD) Preparation

Point Odyssey contains long sequences with annotations stored in `anno.npz` files.

### Splitting Annotations for Fast Loading

Loading large compressed `.npz` files during training creates significant CPU and I/O bottlenecks. We provide a utility to split each sequence's `anno.npz` into separate memory-mappable `.npy` files (`trajs_2d.npy`, `trajs_3d.npy`, `visibs.npy`, `valids.npy`, `intrinsics.npy`, `extrinsics.npy`):

```bash
python scripts/split_pod_annotations.py /path/to/point_odyssey/train
python scripts/split_pod_annotations.py /path/to/point_odyssey/val
```

This writes an `anno_split/` directory inside each sequence folder alongside `anno.npz`. The dataloader automatically detects and memory-maps these `.npy` arrays if present.

---

## 3. Point Odyssey Annotation Fixes

PointOdyssey annotations have a few errors, which we detail in supplementary of our paper.This codebase includes built-in fixes for these defects, active by default.

All corrections are **enabled by default** using the bundled metadata manifest located at:
```
datasets/utils/pod_corrections_data/pod_corrections.json
```
No extra flags or downloads are required.
---

## 4. Precomputing Geometry Annotations (VGGT-Ω)

For benchmark evaluation, precomputed depth and camera poses are generated using [VGGT-Omega](http://vggt-omega.github.io/).

You can generate these annotations using `annotation/vggt_omega_long.py` (or `scripts/vggt_omega_long.sh` for multi-GPU / SLURM jobs):

```bash
python -m trackeverything.annotation.vggt_omega_long \
  --provider_cfg adt_minival \
  --output_path /path/to/annotations
```

Pre-configured provider configs live in `annotation/provider_configs/` (e.g. `adt_minival.yaml`, `pstudio_minival.yaml`, `drivetrack_minival.yaml`, `pod_train.yaml`, `pod_val.yaml`).
