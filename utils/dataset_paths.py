# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Central dataset-path resolution.

All dataset/annotation configs reference local dataset locations through a
single editable file, ``configs/paths.yaml``. This module loads that file and
registers an OmegaConf resolver so any config can interpolate a root with
``${data_path:<key>}`` (e.g. ``${data_path:kubric_root}``).

Importing this module registers the resolver as a side-effect. It is imported
by every Hydra compose entry point (the main train/eval driver, the dataset
loader, and the annotation provider) so the resolver is available regardless of
which Hydra search path is active.
"""

from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

PATHS_FILE = Path(__file__).resolve().parent.parent / "configs" / "paths.yaml"


@lru_cache(maxsize=1)
def _paths():
    return OmegaConf.load(PATHS_FILE)


def get_path(key: str) -> str:
    """Return the configured root for ``key`` from ``configs/paths.yaml``."""
    cfg = _paths()
    if key not in cfg:
        raise KeyError(
            f"Unknown dataset path key '{key}'. Add it to {PATHS_FILE} "
            f"(available keys: {sorted(cfg.keys())})."
        )
    return str(cfg[key])


OmegaConf.register_new_resolver("data_path", get_path, replace=True)
