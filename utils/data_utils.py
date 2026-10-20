# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import fsspec
import numpy as np
from PIL import Image

import ipdb
st = ipdb.set_trace


def read_gen_normal(file_name_):
    ext = os.path.splitext(file_name_)[-1]
    # with fsspec.open(file_name_, "rb") as file_name:
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name_)
        # img.load()
        # return img
        # return Image.open(file_name)
    elif ext == ".bin" or ext == ".raw" or ext == ".npy" or ext == ".npz":
        return np.load(file_name_, allow_pickle=True)
        # return {k: v.copy() for k, v in data.items()}
        # return np.load(file_name, allow_pickle=True)
    else:
        raise NotImplementedError("extension not supported: %s" % ext)


def read_gen(file_name_):
    ext = os.path.splitext(file_name_)[-1]
    with fsspec.open(file_name_, "rb") as file_name:
        if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
            # img = Image.open(file_name)
            # img.load()
            # return img
            return Image.open(file_name)
        elif ext == ".bin" or ext == ".raw" or ext == ".npy" or ext == ".npz":
            # data = np.load(file_name, allow_pickle=True)
            # return {k: v.copy() for k, v in data.items()}
            return np.load(file_name, allow_pickle=True)
        else:
            raise NotImplementedError("extension not supported: %s" % ext)

        # elif ext == '.flo':
        #     return readFlow(file_name).astype(np.float32)
        # elif ext == '.pfm':
        #     flow = readPFM(file_name).astype(np.float32)
        #     if len(flow.shape) == 2:
        # #         return flow
        #     else:
        #         return flow[:, :, :-1]
        # elif ext == '.flo5':
        #     return readFlo5Flow(file_name)
        # return imageio.imread(file_name)[:,:,0:3]
