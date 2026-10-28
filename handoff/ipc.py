"""Atomic file IPC lets simulation and OFT use different Python environments."""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np

from .common import atomic_json, atomic_npz


class FilePolicyClient:
    def __init__(self, directory, timeout=120):
        self.root = Path(directory)
        self.timeout = timeout
        self.ready = json.loads((self.root / "ready.json").read_text())
        self.policy_id = self.ready["policy_id"]
        self.server_id = self.ready["server_id"]
        self.reset("uninitialized")

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.first = True

    def predict(self, obs):
        identifier = uuid.uuid4().hex
        request = self.root / "requests" / f"{identifier}.npz"
        response = self.root / "responses" / f"{identifier}.npz"
        error = response.with_suffix(".json")
        atomic_npz(request, global_rgb=obs["global_rgb"], wrist_rgb=obs["wrist_rgb"], proprio=obs["proprio"],
                   instruction=np.array(obs["instruction"]), episode_id=np.array(self.episode_id),
                   reset=np.array(self.first), server_id=np.array(self.server_id))
        self.first = False
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if error.exists():
                    raise RuntimeError(json.loads(error.read_text())["error"])
                if response.exists():
                    with np.load(response, allow_pickle=False) as result:
                        if str(result["policy_id"].item()) != self.policy_id or str(result["server_id"].item()) != self.server_id:
                            raise RuntimeError("OFT server or checkpoint changed during a frozen-policy experiment")
                        return result["actions"].copy()
                time.sleep(0.05)
            raise TimeoutError(f"OFT inference timed out after {self.timeout}s")
        finally:
            request.unlink(missing_ok=True)
            response.unlink(missing_ok=True)
            error.unlink(missing_ok=True)


def serve(policy, directory, poll_interval=0.05, stop=None):
    root = Path(directory)
    for name in ("requests", "responses"):
        (root / name).mkdir(parents=True, exist_ok=True)
    server_id = uuid.uuid4().hex
    atomic_json(root / "ready.json", {"policy_id": policy.policy_id, "server_id": server_id,
                                       "action_convention": "LIBERO-native", "chunk_shape": [8, 7]})
    print(f"OFT service ready: {root}, checkpoint={policy.policy_id}", flush=True)
    try:
        while stop is None or not stop.is_set():
            for path in sorted((root / "requests").glob("*.npz")):
                response = root / "responses" / path.name
                try:
                    with np.load(path, allow_pickle=False) as request:
                        if str(request["server_id"].item()) != server_id:
                            raise RuntimeError("Request belongs to an earlier OFT server instance")
                        if bool(request["reset"].item()):
                            policy.reset(str(request["episode_id"].item()))
                        observation = {name: request[name].copy() for name in ("global_rgb", "wrist_rgb", "proprio")}
                        observation["instruction"] = str(request["instruction"].item())
                    actions = np.asarray(policy.predict(observation))
                    if actions.shape != (8, 7) or not np.isfinite(actions).all():
                        raise ValueError("Invalid action chunk")
                    # A timed-out client removes its request; do not leave an orphan response.
                    if path.exists():
                        atomic_npz(response, actions=actions, policy_id=np.array(policy.policy_id), server_id=np.array(server_id))
                except Exception as exc:
                    if path.exists():
                        atomic_json(response.with_suffix(".json"), {"error": f"{type(exc).__name__}: {exc}"})
                finally:
                    path.unlink(missing_ok=True)
            time.sleep(poll_interval)
    finally:
        (root / "ready.json").unlink(missing_ok=True)
