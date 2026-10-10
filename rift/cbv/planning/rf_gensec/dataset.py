"""Offline transition datasets compatible with FREA's grouped HDF5 files."""

from pathlib import Path

import h5py
import numpy as np
import torch

from .risk_calculator import RiskCalculator


REQUIRED_FIELDS = ("obs", "next_obs", "actions", "dones")


class RFGensecDataset:
    """Load FREA/RIFT transitions into CPU memory and sample on a target device."""

    def __init__(self, path, actor_rows=3, calculate_missing_risk=True,
                 risk_calculator=None, split="all", split_seed=0,
                 split_fractions=(0.7, 0.15, 0.15), drop_incomplete=True,
                 risk_bin_edges=None, risk_bin_fractions=None):
        self.path = Path(path)
        self.actor_rows = int(actor_rows)
        self.data = self._read(self.path)
        missing = [name for name in REQUIRED_FIELDS if name not in self.data]
        if missing:
            raise ValueError(f"dataset is missing required fields: {', '.join(missing)}")
        self.source_length = len(self.data["obs"])
        self.split = split
        self.episode_count = None
        if split != "all":
            indices, self.episode_count = self._split_indices(
                self.data["dones"], split, split_seed, split_fractions,
                drop_incomplete)
            self.data = {name: value[indices] for name, value in self.data.items()}
        self.length = len(self.data["obs"])
        if any(len(value) != self.length for value in self.data.values()):
            raise ValueError("dataset fields do not have the same length")
        if "total_risk" not in self.data:
            if "risk" in self.data:
                self.data["total_risk"] = self.data["risk"].astype(np.float32)
            elif calculate_missing_risk:
                calculator = risk_calculator or RiskCalculator()
                self.data["total_risk"] = calculator.calculate_total_risk(
                    self.data["obs"], actor_rows=self.actor_rows)[0]
            else:
                raise ValueError("dataset has neither total_risk nor risk")
        self.risk_bin_edges = None
        self.risk_bin_fractions = None
        self.risk_bin_pools = None
        if risk_bin_edges is not None or risk_bin_fractions is not None:
            self.configure_risk_sampling(risk_bin_edges, risk_bin_fractions)

    def configure_risk_sampling(self, edges, fractions):
        """Configure stratified sampling over continuous total-risk intervals."""
        if edges is None or fractions is None:
            raise ValueError("risk_bin_edges and risk_bin_fractions must be provided together")
        edges = np.asarray(edges, dtype=np.float64)
        fractions = np.asarray(fractions, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("risk_bin_edges must be a strictly increasing one-dimensional sequence")
        if len(fractions) != len(edges) - 1 or np.any(fractions < 0):
            raise ValueError("risk_bin_fractions must have len(edges)-1 non-negative values")
        if not np.isclose(fractions.sum(), 1.0):
            raise ValueError("risk_bin_fractions must sum to one")
        risk = self.data["total_risk"].reshape(-1)
        bin_ids = np.digitize(risk, edges[1:-1], right=False)
        pools = [np.flatnonzero(bin_ids == index) for index in range(len(fractions))]
        requested_empty = [index for index, (pool, fraction) in enumerate(zip(pools, fractions))
                           if fraction > 0 and not len(pool)]
        if requested_empty:
            raise ValueError(f"risk sampling requested empty bins: {requested_empty}")
        self.risk_bin_edges = edges
        self.risk_bin_fractions = fractions
        self.risk_bin_pools = pools

    def risk_bin_counts(self):
        if self.risk_bin_pools is None:
            return None
        return [len(pool) for pool in self.risk_bin_pools]

    @staticmethod
    def _split_indices(dones, split, seed, fractions, drop_incomplete):
        names = ("train", "validation", "test")
        if split not in names:
            raise ValueError(f"split must be 'all' or one of {names}, got {split!r}")
        fractions = np.asarray(fractions, dtype=np.float64)
        if fractions.shape != (3,) or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("split_fractions must contain three non-negative values summing to one")
        done_indices = np.flatnonzero(np.asarray(dones).reshape(-1) > 0.5)
        starts = np.r_[0, done_indices[:-1] + 1]
        episodes = [np.arange(start, end + 1) for start, end in zip(starts, done_indices)]
        trailing_start = int(done_indices[-1] + 1) if len(done_indices) else 0
        if not drop_incomplete and trailing_start < len(dones):
            episodes.append(np.arange(trailing_start, len(dones)))
        if not episodes:
            raise ValueError("no complete episodes found from dones")
        order = np.random.default_rng(seed).permutation(len(episodes))
        first = int(len(order) * fractions[0])
        second = first + int(len(order) * fractions[1])
        partitions = np.split(order, (first, second))
        selected = partitions[names.index(split)]
        if not len(selected):
            raise ValueError(f"episode split {split!r} is empty")
        return np.sort(np.concatenate([episodes[index] for index in selected])), len(selected)

    @staticmethod
    def _read(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        chunks = {}
        with h5py.File(path, "r") as handle:
            groups = [value for value in handle.values() if isinstance(value, h5py.Group)]
            containers = groups or [handle]
            for container in containers:
                for name, value in container.items():
                    if isinstance(value, h5py.Dataset):
                        chunks.setdefault(name, []).append(np.asarray(value, dtype=np.float32))
        return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}

    @property
    def obs_shape(self):
        return tuple(self.data["obs"].shape[1:])

    @property
    def action_dim(self):
        actions = self.data["actions"]
        return 1 if actions.ndim == 1 else actions.shape[-1]

    def sample(self, batch_size, device="cpu"):
        batch_size = int(batch_size)
        if self.risk_bin_pools is None:
            indices = np.random.randint(0, self.length, size=batch_size)
        else:
            raw_counts = self.risk_bin_fractions * batch_size
            counts = np.floor(raw_counts).astype(int)
            remainder_order = np.argsort(-(raw_counts - counts))
            counts[remainder_order[:batch_size - counts.sum()]] += 1
            indices = np.concatenate([
                np.random.choice(pool, size=count, replace=count > len(pool))
                for pool, count in zip(self.risk_bin_pools, counts) if count
            ])
            np.random.shuffle(indices)
        return {
            name: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
            for name, value in self.data.items()
        }

    def summary(self):
        return {
            "path": str(self.path), "length": self.length,
            "source_length": self.source_length, "split": self.split,
            "episode_count": self.episode_count,
            "risk_bin_edges": (self.risk_bin_edges.tolist()
                               if self.risk_bin_edges is not None else None),
            "risk_bin_counts": self.risk_bin_counts(),
            "obs_shape": self.obs_shape, "action_dim": self.action_dim,
            "fields": sorted(self.data),
        }
