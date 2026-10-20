#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert PointOdyssey anno.npz files into individual .npy files for true mmap.

Usage:
    python scripts/split_pod_annotations.py /path/to/point_odyssey/train
    python scripts/split_pod_annotations.py /path/to/point_odyssey/val

This creates an `anno_split/` directory alongside each `anno.npz` containing
one .npy file per array key. The original anno.npz is left untouched.
"""

import argparse
import os
import sys

import numpy as np


ARRAY_KEYS = ("trajs_2d", "trajs_3d", "visibs", "valids", "intrinsics", "extrinsics")


def split_one(video_dir: str) -> None:
    npz_path = os.path.join(video_dir, "anno.npz")
    out_dir = os.path.join(video_dir, "anno_split")

    if not os.path.isfile(npz_path):
        print(f"  SKIP (no anno.npz): {video_dir}")
        return

    if os.path.isdir(out_dir) and all(
        os.path.isfile(os.path.join(out_dir, f"{k}.npy")) for k in ARRAY_KEYS
    ):
        print(f"  SKIP (already split): {video_dir}")
        return

    print(f"  Splitting: {video_dir} ... ", end="", flush=True)
    anns = np.load(npz_path)
    os.makedirs(out_dir, exist_ok=True)
    for key in ARRAY_KEYS:
        arr = anns[key]
        np.save(os.path.join(out_dir, f"{key}.npy"), arr)
    print(f"done ({len(ARRAY_KEYS)} arrays)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", help="Root directory containing video subdirs (e.g. .../train)")
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"Error: {data_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    video_dirs = sorted(
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    print(f"Found {len(video_dirs)} video directories in {data_root}")

    for vd in video_dirs:
        split_one(vd)

    print("All done.")


if __name__ == "__main__":
    main()
