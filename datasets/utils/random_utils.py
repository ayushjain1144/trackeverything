# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import numpy as np
from typing import List, Sequence, Union

class RandomMapping:
    def __init__(self, seed: Union[int, str]):
        self.seed = seed

    def randint(self, value: Union[int, str], start: int, end: int) -> int:
        combined = f"hash_randint_{self.seed}_{value}"
        
        hash_object = hashlib.sha256(combined.encode())
        hash_hex = hash_object.hexdigest()
        hash_int = int(hash_hex, 16)

        assert start < end, "start must be less than end"
        range = end - start
        return start + (hash_int % range)
    
    def random(self, value: Union[int, str]) -> float:
        combined = f"hash_random_{self.seed}_{value}"
        
        hash_object = hashlib.sha256(combined.encode())
        hash_hex = hash_object.hexdigest()
        hash_int = int(hash_hex, 16)
        MOD = 2 ** 128 + 1

        return (hash_int % MOD) / (MOD - 1)

class WeightedSampler:
    def __init__(self, weights: Sequence[Union[int, float]]):
        self.weights = np.array(weights, dtype=np.float64)
        self.cumulative_weights = np.cumsum(self.weights)
        self.total_weight = self.cumulative_weights[-1]

    def sample(self, rng_or_seed: Union[np.random.Generator, float]) -> int:
        random_value = rng_or_seed if isinstance(rng_or_seed, float) else rng_or_seed.random()
        random_value = random_value * self.total_weight
        return int (np.searchsorted(self.cumulative_weights, random_value))
