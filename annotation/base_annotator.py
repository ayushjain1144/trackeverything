# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np

from ..datasets.datatypes import RawSliceData


class BaseAnnotator:
    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]: ...

    def to(self, device: str) -> None: ...
