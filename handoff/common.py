from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager
from typing import Any

import numpy as np


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_id(state) -> str:
    return hashlib.sha256(np.round(np.asarray(state, dtype=np.float64), 8).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: Path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}; repair a truncated last line before resuming") from exc


def append_jsonl(path: Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    edges = np.asarray(cfg["radius_edges"])
    if edges.ndim != 1 or len(edges) < 2 or edges[0] <= 0 or not np.all(np.diff(edges) > 0):
        raise ValueError("radius_edges must be positive and strictly increasing")
    if cfg["action_chunk"] != 8:
        raise ValueError("This OFT LIBERO adapter requires action_chunk=8")
    for key in ("sectors", "execute_steps", "max_steps", "max_attempt_factor", "image_size", "max_neighbors"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["execute_steps"] > cfg["action_chunk"]:
        raise ValueError("Invalid sectors or action execution horizon")
    for key in ("counts", "posture_counts"):
        if set(cfg[key]) != {"train", "calibration", "test"} or any(not isinstance(v, int) or v < 1 for v in cfg[key].values()):
            raise ValueError(f"{key} must contain positive train/calibration/test counts")
    return cfg


@contextmanager
def run_lock(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".collect.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another collector is writing this run; use a separate output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def group_split(group: str, seed: int) -> str:
    value = int(fingerprint([seed, group])[:13], 16) / 16**13
    return "train" if value < 0.7 else "calibration" if value < 0.85 else "test"


def assert_disjoint(rows):
    assignments = {}
    policies = set()
    for row in rows:
        split = row["split"]
        if split not in {"train", "calibration", "test"}:
            raise ValueError(f"Unknown split: {split}")
        for field in ("scene_id", "initial_state_id", "pair_id"):
            key = row.get(field)
            if key is not None:
                previous = assignments.setdefault((field, key), split)
                if previous != split:
                    raise ValueError(f"Split leakage: {field}={key} occurs in {previous} and {split}")
        if row.get("policy_id"):
            policies.add(row["policy_id"])
    if len(policies) > 1:
        raise ValueError("A capability experiment must use exactly one frozen policy version")


def checkpoint_id(root: Path) -> str:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Use a local frozen checkpoint directory (download remote snapshots first)")
    paths = sorted(p for p in root.rglob("*") if p.suffix in {".json", ".py", ".pt", ".bin", ".safetensors"})
    if not paths or not any(p.suffix in {".pt", ".bin", ".safetensors"} for p in paths):
        raise ValueError("No checkpoint weights found")
    return fingerprint([(str(p.relative_to(root)), file_sha256(p)) for p in paths])
