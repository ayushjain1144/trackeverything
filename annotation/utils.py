# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fcntl
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TypeVar

import h5py
import ipdb
import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir

from iopath.common.file_io import PathManager
from omegaconf import open_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import Sampler
from tqdm import tqdm

from ..datasets.datatypes import RawSliceData
from .base_annotator import BaseAnnotator

path_manager = PathManager()

CONFIG_PATH = Path(__file__).parent / "provider_configs"

T_co = TypeVar("T_co", covariant=True)
st = ipdb.set_trace


class UnevenDistributedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        skip_idxs: Optional[List[int]] = None,
        reverse_order: bool = False,
    ) -> None:
        if num_replicas is None:
            if not torch.dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if skip_idxs is not None:
            self.skip_idxs = set(skip_idxs)
        else:
            self.skip_idxs = set()

        self.num_samples = len(self.dataset) // self.num_replicas
        if len(self.dataset) % self.num_replicas != 0:
            if (
                self.rank % self.num_replicas
                <= (len(self.dataset) - 1) % self.num_replicas
            ):
                self.num_samples += 1

        self.shuffle = shuffle
        self.seed = seed

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # subsample
        indices = indices[self.rank % self.num_replicas :: self.num_replicas]

        self.indices = [idx for idx in indices if idx not in self.skip_idxs]
        if reverse_order:
            self.indices.reverse()
        self.num_samples = len(self.indices)

    def __iter__(self) -> Iterator[T_co]:
        assert len(self.indices) == self.num_samples
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def load_annotations(
    annotation_path: str,
    key: str,
    seq_id: int,
    start: int,
    length: int,
    stride: int,
    seq_name: str,
    name: str,
    split: str,
    annotator_name: str = "pi3",
) -> np.ndarray:
    # confirm_file = os.path.join(annotation_path, "confirm_dir", f"{seq_id}.confirm")
    # assert path_manager.exists(
    #     confirm_file
    # ), f"Confirm file {confirm_file} does not exist"
    
    # if "/" in seq_name:
    #     seq_name = seq_name.split("/")[-1]
    
    if "/" in seq_name:
        seq_name_prefix = seq_name.split("/")[-1]
    elif '.' in seq_name:
        seq_name_prefix = seq_name.split(".")[0]
    else:
        seq_name_prefix = seq_name
    if name == "dynamic_replica":
        h5_file = os.path.join(annotation_path, f"{split}/{seq_name_prefix}_source_left/{seq_name_prefix}_{annotator_name}.h5")
    elif name == "point_odyssey":
        h5_file = os.path.join(annotation_path, f"{split}/{seq_name_prefix}/{seq_name_prefix}_{annotator_name}.h5")
    elif 'tapvid3d' in name or 'pstudio' in name or 'adt' in name or 'drivetrack' in name:
        h5_file = os.path.join(annotation_path, f"{seq_name_prefix}_{annotator_name}.h5")
    else:
        h5_file = os.path.join(annotation_path, f"{seq_name_prefix}/{seq_name_prefix}_{annotator_name}.h5")

    # h5_file = os.path.join(annotation_path, f"{seq_id}.h5")
    # h5_file = os.path.join(annotation_path, f"{seq_name}/{seq_name}_pi3.h5")
    with path_manager.open(h5_file, "rb") as f_handle:
        with h5py.File(f_handle, "r") as f:
            ret: np.ndarray = f[key][start : start + length * stride : stride]  # type: ignore
    return ret  # type: ignore


class SeqDataset(Dataset):
    def __init__(self, provider, num_threads):
        self.provider = provider
        self.num_threads = num_threads
        self.seq_lens = provider.seq_lens
        if self.seq_lens is None:
            print("loading seq_lens, this may take a while")
            self.seq_lens = provider.load_seq_lens()

    def __len__(self):
        return len(self.seq_lens)

    def __getitem__(self, idx):
        seq_len = self.seq_lens[idx]
        # seq_len = 600
        try:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                data: RawSliceData = self.provider.load_slice(
                    idx,
                    start=0,
                    length=seq_len,
                    stride=1,
                    rng=np.random.default_rng(seed=0),
                    executor=executor,
                )  # type: ignore
        except Exception as e:
            print(f"Failed to load {idx} because of {e}")
            return idx, None
        return idx, data


def dummy_collate(batch):
    return batch[0]


def generate_annotations(
    output_path: str,
    annotator: BaseAnnotator,
    provider_cfg: str,
    num_dataloader_threads: int = 4,
    num_dataloader_workers: int = 8,
    mod10: Optional[int] = None,
    _idx: Optional[int] = None,
    annotator_name: str = "megasam",
):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        device = f"cuda:{local_rank % torch.cuda.device_count()}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"

    print(f"Rank: {rank}, Local Rank: {local_rank}, Using device: {device}")

    annotator.to(device)

    # Load provider config
    with initialize_config_dir(
        version_base=None, config_dir=str(CONFIG_PATH.absolute())
    ):
        cfg = compose(config_name=provider_cfg)  # type: ignore
    name = cfg.name
    override_anno = cfg.override_anno
    with open_dict(cfg):
        del cfg["name"]
        del cfg["override_anno"]

    from ..datasets.providers.base_provider import BaseDataProvider
    provider = BaseDataProvider.from_config(cfg, name=name, override_anno=override_anno)
    output_dir_path = output_path
    path_manager.mkdirs(output_dir_path)

    dataset = SeqDataset(provider, num_dataloader_threads)

    skip_idxs = []
    seq_ids = range(len(dataset))
    if rank == 0:
        seq_ids = tqdm(seq_ids, desc="Scanning existing confirms")
    for seq_id in seq_ids:
        if annotator_name == "megasam":
            confirm_path = os.path.join(output_dir_path, f"confirm_dir/{seq_id}.confirm")
        else:
            confirm_path = os.path.join(
                output_dir_path, f"confirm_dir/{annotator_name}_{seq_id}.confirm"
            )
        if path_manager.exists(confirm_path):
            skip_idxs.append(seq_id)

    if world_size > 1:
        sampler = UnevenDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            skip_idxs=skip_idxs,
            reverse_order=True,
        )
    else:
        sampler = UnevenDistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=True,
            skip_idxs=skip_idxs,
            reverse_order=True,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_dataloader_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=dummy_collate,
    )
    if rank == 0:
        dataloader = tqdm(dataloader)

    if _idx is not None:
        assert world_size == 1
        dataloader = [dataset[_idx]]

    failed_seqs = []

    for dummy_batch in dataloader:
        seq_id, data = dummy_batch

        if data is None:
            failed_seqs.append(seq_id)
            continue
        if mod10 is not None and (int(seq_id)) % 10 != mod10:
            continue
        
        seq_name = data.seq_name

        if "/" in seq_name:
            seq_name_prefix = seq_name.split("/")[-1]
        elif '.' in seq_name:
            seq_name_prefix = seq_name.split(".")[0]
        else:
            seq_name_prefix = seq_name
        if name == "dynamic_replica":
            split = cfg.split
            seq_output_path = os.path.join(output_dir_path, f"{split}/{seq_name_prefix}_source_left/{seq_name_prefix}_{annotator_name}.h5")
        elif name == "point_odyssey":
            split = cfg.split
            seq_output_path = os.path.join(output_dir_path, f"{split}/{seq_name_prefix}/{seq_name_prefix}_{annotator_name}.h5")
        elif 'tapvid3d' in name or 'pstudio' in name or 'adt' in name or 'drivetrack' in name:
            seq_output_path = os.path.join(output_dir_path, f"{seq_name_prefix}_{annotator_name}.h5")
        else:
            seq_output_path = os.path.join(output_dir_path, f"{seq_name_prefix}/{seq_name_prefix}_{annotator_name}.h5")
        
        if annotator_name == 'megasam':
            confirm_path = os.path.join(output_dir_path, f"confirm_dir/{seq_id}.confirm")
        else:
            confirm_path = os.path.join(output_dir_path, f"confirm_dir/{annotator_name}_{seq_id}.confirm")
        
        if path_manager.exists(confirm_path):
            continue # doesn't repeat data collection / generation.

        # Create parent directory for seq_output_path
        seq_parent_dir = os.path.dirname(seq_output_path)
        path_manager.mkdirs(seq_parent_dir)
        # try:
        annotation: Dict[str, np.ndarray] = annotator(data)

        with path_manager.open(seq_output_path, "wb") as lock_file:
            # fcntl.flock(lock_file, fcntl.LOCK_EX)
            # try:

            # print(f"\nAre you sure you want to overwrite files?")
            with h5py.File(lock_file, "w") as f:
                for key, value in annotation.items():
                    assert isinstance(
                        value, np.ndarray
                    ), "Annotation must be numpy arrays"
                    assert (
                        key in data.__dict__
                        or f"gt_{key}" in data.__dict__
                        or key == "depths"
                    ), f"Key {key} not in data"
                    f.create_dataset(
                        key,
                        shape=value.shape,
                        dtype=value.dtype,
                        compression="gzip",
                    )
                    f[key][:] = value  # type: ignore
                f.flush()

            # finally:
                # fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        # Create confirm file
        path_manager.mkdirs(os.path.dirname(confirm_path))
        with path_manager.open(confirm_path, "w") as f:
            f.write("")
        # except Exception as e:
        #     failed_seqs.append(seq_id)
        #     print(f"Failed to annotate {seq_id}: {e}")
        #     continue

    if len(failed_seqs) > 0:
        print(f"Failed to annotate {len(failed_seqs)} sequences: {failed_seqs}")
        # save failed sequences to a local file with rank as suffix
        with open(f"failed_seqs_{rank}.txt", "w") as f:
            f.write("\n".join(str(seq_id) for seq_id in failed_seqs))
    else:
        print("All sequences annotated successfully")
