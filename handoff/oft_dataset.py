"""An in-memory bridge to OFT's existing RLDSBatchTransform; no upstream files modified."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .data import DATASET_NAME, action_chunk, load_manifest, normalize


class LocalOFTDataset(IterableDataset):
    def __init__(self, data_root_dir, data_mix, batch_transform, resize_resolution=None,
                 shuffle_buffer_size=256000, train=True, image_aug=False, **kwargs):
        super().__init__()
        if data_mix != DATASET_NAME:
            raise ValueError(f"Local dataset must be named {DATASET_NAME}, got {data_mix}")
        if image_aug:
            raise ValueError("Image augmentation is disabled for the first experiment; do not silently ignore it")
        self.root = Path(data_root_dir)
        self.train = train
        self.rows = [r for r in load_manifest(self.root) if r["split"] == ("train" if train else "calibration")]
        if not self.rows:
            raise ValueError("Requested demonstration split is empty")
        self.batch_transform = batch_transform
        self.dataset_statistics = json.loads((self.root / "dataset_statistics.json").read_text())
        self.stats = self.dataset_statistics[DATASET_NAME]
        self.length = sum(r["length"] for r in self.rows)
        self.weights = np.asarray([r["length"] for r in self.rows], dtype=float) / self.length
        self.resize_resolution = resize_resolution

    def __len__(self):
        return self.length

    def __iter__(self):
        worker = get_worker_info()
        rank = int(os.environ.get("RANK", "0"))
        seed = torch.initial_seed() + rank * 100003 + (worker.id if worker else 0)
        rng = np.random.default_rng(seed % (2**63 - 1))
        cache = {}

        def frame(index, timestep):
            row = self.rows[index]
            if index not in cache:
                if len(cache) >= 2:
                    cache.pop(next(iter(cache)))
                cache[index] = {name: np.load(self.root / row["path"] / f"{name}.npy", mmap_mode="r")
                                for name in ("global_rgb", "wrist_rgb", "proprio", "actions")}
            arrays = cache[index]
            primary, wrist = arrays["global_rgb"][timestep], arrays["wrist_rgb"][timestep]
            if self.resize_resolution is not None:
                # OFT supplies (height, width); fail rather than train on a mismatched camera stream.
                desired = tuple(self.resize_resolution[:2])
                if len(desired) == 2 and primary.shape[:2] != desired:
                    raise ValueError(f"Prepared image shape {primary.shape[:2]} != OFT resolution {desired}")
            return self.batch_transform({
                "dataset_name": DATASET_NAME,
                "observation": {"image_primary": primary[None], "image_wrist": wrist[None],
                                "proprio": normalize(arrays["proprio"][timestep], self.stats["proprio"])[None]},
                "task": {"language_instruction": row["instruction"].encode()},
                "action": normalize(action_chunk(arrays["actions"], timestep, 8), self.stats["action"]),
            })

        if self.train:
            while True:
                index = int(rng.choice(len(self.rows), p=self.weights))
                yield frame(index, int(rng.integers(self.rows[index]["length"])))
        else:
            for index, row in enumerate(self.rows):
                if worker and index % worker.num_workers != worker.id:
                    continue
                for timestep in range(row["length"]):
                    yield frame(index, timestep)
