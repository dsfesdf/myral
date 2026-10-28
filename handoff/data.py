from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import atomic_json, assert_disjoint, file_sha256, fingerprint, group_split, read_jsonl

DATASET_NAME = "libero_handoff"


def crop_start(distances, widths, contacts, distance=0.12, open_width=0.04, preroll=5):
    hits = np.flatnonzero((np.asarray(distances) <= distance) & (np.asarray(widths) >= open_width) & ~np.asarray(contacts, dtype=bool))
    return max(0, int(hits[0]) - preroll) if len(hits) else None


def libero_to_oft_actions(actions):
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.0001:
        raise ValueError("Expected bounded LIBERO actions [T,7], with -1=open and +1=close")
    actions[:, -1] = (1.0 - actions[:, -1]) / 2.0  # OFT: 1=open, 0=closed.
    return actions


def oft_to_libero_actions(actions):
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Expected OFT action chunk [T,7]")
    actions[:, -1] = -np.sign(2 * actions[:, -1] - 1)
    if np.max(np.abs(actions)) > 1.0001:
        raise ValueError("OFT action head returned out-of-range unnormalized LIBERO actions")
    return actions


def summary(values, mask=None):
    result = {"mean": values.mean(0), "std": values.std(0), "min": values.min(0), "max": values.max(0),
              "q01": np.quantile(values, 0.01, axis=0), "q99": np.quantile(values, 0.99, axis=0)}
    if mask is not None:
        result["mask"] = mask
    return result


def normalize(values, stats):
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    # Match upstream OFT normalize_proprio, including its epsilon and constant channels.
    normalized = np.clip(2 * (np.asarray(values) - lo) / (hi - lo + 1e-8) - 1, -1, 1)
    mask = np.asarray(stats.get("mask", np.ones_like(lo, dtype=bool)))
    return np.where(mask, normalized, values).astype(np.float32)


def action_chunk(actions, start, chunk_size=8):
    indices = np.minimum(np.arange(start, start + chunk_size), len(actions) - 1)
    return np.asarray(actions[indices]).copy()


def load_manifest(root):
    root = Path(root).resolve()
    rows = list(read_jsonl(root / "manifest.jsonl"))
    if not rows:
        raise ValueError("Empty demonstration manifest")
    assert_disjoint(rows)
    for row in rows:
        if not (root / row["path"] / "metadata.json").exists() or not (root / row["path"] / "model.xml").exists():
            raise ValueError(f"Missing episode: {row['path']}")
    return rows


def compute_statistics(root):
    root = Path(root)
    train = [r for r in load_manifest(root) if r["split"] == "train"]
    if not train:
        raise ValueError("No training scenes; collect additional distinct scenes instead of leaking held-out ones")
    actions = np.concatenate([np.load(root / r["path"] / "actions.npy") for r in train])
    proprio = np.concatenate([np.load(root / r["path"] / "proprio.npy") for r in train])
    stats = {DATASET_NAME: {"action": summary(actions, [True] * 6 + [False]), "proprio": summary(proprio),
                            "num_trajectories": len(train), "num_transitions": len(actions)}}
    atomic_json(root / "dataset_statistics.json", stats)
    return stats


def prepare(raw_hdf5, output, cfg, limit=None, adapter_factory=None):
    import h5py
    from .sim import LiberoAdapter

    root = Path(output).resolve()
    if (root / "manifest.jsonl").exists():
        raise ValueError("Output already has a manifest. Choose a fresh directory to preserve provenance.")
    root.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(Path(raw_hdf5))
    adapter = (adapter_factory or LiberoAdapter)(cfg)
    rows, rejected = [], []
    try:
        with h5py.File(raw_hdf5, "r") as source:
            demos = source["data"]
            names = sorted(demos, key=lambda n: int(n.rsplit("_", 1)[-1]))
            for demo_name in names[:limit]:
                demo = demos[demo_name]
                states = np.asarray(demo["states"])
                native_actions = np.asarray(demo["actions"])
                if "model_file" not in demo.attrs:
                    raise ValueError(f"{demo_name}: missing recorded model XML; cannot restore fixed fixtures faithfully")
                adapter.load_model(demo.attrs["model_file"])
                if len(states) not in (len(native_actions), len(native_actions) + 1):
                    raise ValueError(f"{demo_name}: actions and states are not timestep-aligned")
                adapter.reset(states[0])
                scene = adapter.scene_id()
                distances, widths, contacts = [], [], []
                for state in states[:len(native_actions)]:
                    adapter.set_state(state)
                    position, _ = adapter.eef_pose()
                    target = adapter.scene()[cfg["target_object"]]["position"]
                    distances.append(np.linalg.norm(position - target))
                    widths.append(adapter.width())
                    contacts.append(adapter.has_target_contact())
                cut = crop_start(distances, widths, contacts, cfg["crop_distance"], cfg["open_width"], cfg["crop_preroll"])
                if cut is None:
                    rejected.append({"demo": demo_name, "reason": "no_open_pregrasp_within_crop_distance"})
                    continue
                adapter.reset(states[-1])
                if len(states) == len(native_actions):
                    adapter.step(native_actions[-1])
                if not adapter.success():
                    rejected.append({"demo": demo_name, "reason": "replay_did_not_reach_environment_success"})
                    continue
                final_state = adapter.state()
                images, wrists, proprio = [], [], []
                for state in states[cut:len(native_actions)]:
                    adapter.set_state(state)
                    obs = adapter.observe()
                    images.append(obs["global_rgb"])
                    wrists.append(obs["wrist_rgb"])
                    proprio.append(obs["proprio"])
                identifier = fingerprint([source_hash, demo_name, cut])[:20]
                episode = root / "episodes" / identifier
                episode.mkdir(parents=True, exist_ok=False)
                arrays = {"global_rgb": images, "wrist_rgb": wrists, "proprio": proprio,
                          "actions": libero_to_oft_actions(native_actions[cut:]),
                          "native_actions": native_actions[cut:], "states": states[cut:len(native_actions)],
                          "scene_state": states[0], "final_state": final_state}
                for name, array in arrays.items():
                    np.save(episode / f"{name}.npy", np.asarray(array), allow_pickle=False)
                (episode / "model.xml").write_text(adapter.model_xml)
                row = {"episode_id": identifier, "scene_id": scene, "split": group_split(scene, cfg["seed"]),
                       "path": str(episode.relative_to(root)), "length": len(images), "crop_start": cut,
                       "source_demo": demo_name, "source_sha256": source_hash, "instruction": adapter.task.language}
                atomic_json(episode / "metadata.json", row)
                rows.append(row)
                print(f"prepared {demo_name}: start={cut}, frames={len(images)}, split={row['split']}", flush=True)
    finally:
        adapter.close()
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    atomic_json(root / "preparation.json", {"schema_version": 2, "config": cfg, "source_sha256": source_hash, "accepted": len(rows), "rejected": rejected,
                                           "image_convention": "LIBERO raw render rotated 180 degrees; uint8 RGB",
                                           "proprio_convention": "eef_xyz, eef_axis_angle, two_gripper_joint_positions",
                                           "action_convention": "LIBERO arm deltas; OFT gripper 1=open,0=closed"})
    compute_statistics(root)
    return {"episodes": len(rows), "rejected": len(rejected),
            "scene_counts": {s: len({r['scene_id'] for r in rows if r['split'] == s}) for s in ("train", "calibration", "test")}}


def replay_check(data, count=10):
    from .sim import LiberoAdapter

    root = Path(data)
    cfg = json.loads((root / "preparation.json").read_text())["config"]
    adapter = LiberoAdapter(cfg)
    results = []
    try:
        for row in load_manifest(root)[:count]:
            path = root / row["path"]
            adapter.load_model((path / "model.xml").read_text())
            states = np.load(path / "states.npy")
            adapter.reset(states[0])
            obs = adapter.observe()
            image_error = max(float(np.max(np.abs(obs[key].astype(int) - np.load(path / f"{key}.npy", mmap_mode="r")[0].astype(int))))
                              for key in ("global_rgb", "wrist_rgb"))
            proprio_error = float(np.max(np.abs(obs["proprio"] - np.load(path / "proprio.npy")[0])))
            actions = np.load(path / "native_actions.npy")
            converted = oft_to_libero_actions(np.load(path / "actions.npy"))
            action_error = float(np.max(np.abs(converted - actions)))
            succeeded = False
            for action in actions:
                succeeded, done = adapter.step(action)
                if succeeded or done:
                    break
            results.append({"episode_id": row["episode_id"], "success": succeeded,
                            "image_max_error": image_error, "proprio_max_error": proprio_error, "action_max_error": action_error})
    finally:
        adapter.close()
    return {"passed": bool(results) and all(r["success"] and r["image_max_error"] <= 1 and r["proprio_max_error"] < 1e-4 and r["action_max_error"] < 1e-4 for r in results),
            "episodes": results}
