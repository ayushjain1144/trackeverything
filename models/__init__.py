# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing
from pathlib import Path
from typing import Tuple, Union

import ipdb
import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict

logger = logging.getLogger(__name__)
st = ipdb.set_trace

CONFIG_ROOT = Path(__file__).parent.parent / "configs" / "model"

def load_config(cfg: Union[DictConfig, str]) -> DictConfig:
    if isinstance(cfg, str):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(
            version_base=None, config_dir=str(CONFIG_ROOT.absolute())
        ):
            cfg = compose(config_name=cfg)
    return typing.cast(DictConfig, cfg)


def from_config(cfg: Union[DictConfig, str], **kwargs) -> nn.Module:
    cfg = load_config(cfg)
    assert all(key not in kwargs for key in cfg.keys()), "Overwriting model config is not allowed"
    kwargs.update(cfg)
    name = kwargs.pop("name")
    # "track_everything_clean" is the name stored in older checkpoints.
    if name in ("track_everything", "track_everything_clean"):
        from .track_everything import TrackEverything
        return TrackEverything(**kwargs)
    raise ValueError(f"Model {name} not supported")


def from_pretrained(ckpt_path: Union[str, Path]) -> Tuple[nn.Module, DictConfig]:
    assert Path(ckpt_path).exists(), f"Checkpoint {ckpt_path} does not exist"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
        # Weights include the full encoder; skip separate DinoV3 download.
        model_cfg = cfg.get("model", None)
        if model_cfg is not None and "encoder" in model_cfg:
            with open_dict(cfg):
                cfg["model"]["encoder"]["pretrained"] = False
        model = from_config(cfg["model"], image_size=cfg.resolution)
        missing, unexpected = model.load_state_dict(ckpt["weight"], strict=True)
        if missing:
            raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")
    else:
        raise ValueError("Unsupported checkpoint")

    return model, cfg
