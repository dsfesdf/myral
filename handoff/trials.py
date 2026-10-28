from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .common import (append_jsonl, atomic_json, atomic_npz, assert_disjoint, file_sha256,
                     fingerprint, read_jsonl, run_lock, state_id)
from .data import load_manifest
from .geometry import feature_names, region_id, requested_pose, sample_region


def plan_starts(data, output, cfg, phase="position"):
    root, destination = Path(data).resolve(), Path(output).resolve()
    if destination.exists():
        raise ValueError("Choose a new plan filename; plans are immutable")
    rows = load_manifest(root)
    preparation = json.loads((root / "preparation.json").read_text())
    for field in ("task", "suite", "target_object", "goal_object"):
        if preparation["config"][field] != cfg[field]:
            raise ValueError(f"Prepared data and experiment disagree on {field}")
    scenes = {split: {} for split in ("train", "calibration", "test")}
    for row in rows:
        scenes[row["split"]].setdefault(row["scene_id"], row)
    if any(not group for group in scenes.values()):
        raise ValueError("All splits need distinct scenes. Prepare more demonstrations; do not split frames randomly.")
    rng = np.random.default_rng(cfg["seed"])
    counts = cfg["counts"] if phase == "position" else cfg["posture_counts"]
    attempts = cfg["max_attempt_factor"] if phase == "position" else 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    scene_hashes = {row["episode_id"]: (file_sha256(root / row["path"] / "scene_state.npy"),
                                      file_sha256(root / row["path"] / "model.xml")) for row in rows}
    with destination.open("w") as stream:
        for split, count in counts.items():
            available = sorted(scenes[split])
            for ring in range(len(cfg["radius_edges"]) - 1):
                for sector in range(cfg["sectors"]):
                    for index in range(count * attempts):
                        scene = available[index % len(available)]
                        row = scenes[split][scene]
                        xy = sample_region(rng, ring, sector, cfg)
                        pair = fingerprint([cfg["seed"], phase, scene, ring, sector, index, xy])[:24]
                        conditions = [(cfg["height"], 0)] if phase == "position" else [(h, y) for h in cfg["heights"] for y in cfg["yaw_degrees"]]
                        for height, yaw in conditions:
                            candidate = {"scene_id": scene, "split": split, "region": region_id(ring, sector),
                                         "ring": ring, "sector": sector, "xy": xy, "height": height, "yaw_degrees": yaw,
                                         "phase": phase, "pair_id": pair, "seed": int(rng.integers(0, 2**31)),
                                         "scene_state_path": str(root / row["path"] / "scene_state.npy"),
                                         "scene_model_path": str(root / row["path"] / "model.xml"),
                                         "scene_state_sha256": scene_hashes[row["episode_id"]][0],
                                         "scene_model_sha256": scene_hashes[row["episode_id"]][1]}
                            candidate["candidate_id"] = fingerprint(candidate)[:24]
                            stream.write(json.dumps(candidate) + "\n")
                            total += 1
    atomic_json(destination.with_suffix(".meta.json"), {"config": cfg, "phase": phase,
                "manifest_sha256": file_sha256(root / "manifest.jsonl"), "planned_attempts": total,
                "plan_sha256": file_sha256(destination)})
    return {"planned_attempts": total, "path": str(destination)}


def rollout(adapter, policy, candidate_id, cfg):
    policy.reset(candidate_id)
    for start in range(0, cfg["max_steps"], cfg["execute_steps"]):
        actions = np.asarray(policy.predict(adapter.observe()))
        if actions.shape != (cfg["action_chunk"], 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid policy chunk: {actions.shape}")
        for offset in range(min(cfg["execute_steps"], cfg["max_steps"] - start)):
            success, done = adapter.step(actions[offset])
            steps = start + offset + 1
            if success:
                return 1, steps, "success"
            if done:
                return 0, steps, "environment_done"
    return 0, cfg["max_steps"], "timeout"


def trial_key(row):
    return row["split"], row["region"], row["height"], row["yaw_degrees"]


def collect(plan, output, ipc, max_trials=None, timeout=120, adapter_factory=None, policy=None):
    with run_lock(output):
        return _collect_locked(plan, output, ipc, max_trials, timeout, adapter_factory, policy)


def _collect_locked(plan, output, ipc, max_trials=None, timeout=120, adapter_factory=None, policy=None):
    from .ipc import FilePolicyClient
    from .sim import LiberoAdapter

    plan, output = Path(plan).resolve(), Path(output).resolve()
    metadata = json.loads(plan.with_suffix(".meta.json").read_text())
    if metadata["plan_sha256"] != file_sha256(plan):
        raise ValueError("Plan changed after it was generated")
    cfg = metadata["config"]
    policy = policy or FilePolicyClient(ipc, timeout)
    output.mkdir(parents=True, exist_ok=True)
    run = {"policy_id": policy.policy_id, "plan_sha256": metadata["plan_sha256"],
           "config": cfg, "feature_names": feature_names(cfg["max_neighbors"])}
    run_path = output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Cannot resume with a different plan, feature schema, or checkpoint")
    atomic_json(run_path, run)
    log = output / "trials.jsonl"
    previous = list(read_jsonl(log)) if log.exists() else []
    assert_disjoint(previous)
    completed = {r["candidate_id"] for r in previous}
    counts = Counter(trial_key(r) for r in previous if r["valid"] and r.get("success") in (0, 1))
    quotas = cfg["counts"] if metadata["phase"] == "position" else cfg["posture_counts"]
    adapter = (adapter_factory or LiberoAdapter)(cfg)
    new_trials = 0
    try:
        for candidate in read_jsonl(plan):
            key = trial_key(candidate)
            if candidate["candidate_id"] in completed or counts[key] >= quotas[candidate["split"]]:
                continue
            for path_key, hash_key in (("scene_state_path", "scene_state_sha256"), ("scene_model_path", "scene_model_sha256")):
                if file_sha256(Path(candidate[path_key])) != candidate[hash_key]:
                    raise ValueError(f"Scene source changed after planning: {candidate[path_key]}")
            state = np.load(candidate["scene_state_path"], allow_pickle=False)
            adapter.load_model(Path(candidate["scene_model_path"]).read_text())
            adapter.reset(state, zero_velocity=True)
            target = adapter.scene()[cfg["target_object"]]
            position, rotation = requested_pose(target["position"], target["rotation"], candidate["xy"],
                                                candidate["height"], adapter.rim_offset(), candidate["yaw_degrees"])
            valid, reason, details = adapter.initialize(state, position, rotation, candidate["seed"])
            actual_state = adapter.state()
            actual_pos, actual_rot = adapter.eef_pose()
            snapshot = Path("states") / f"{candidate['candidate_id']}.npz"
            atomic_npz(output / snapshot, initial_state=actual_state, scene_state=state, model_xml=np.array(adapter.model_xml))
            result = {**candidate, "policy_id": policy.policy_id, "valid": bool(valid), "success": None,
                      "initial_state_id": fingerprint([state_id(actual_state[1:]), fingerprint(adapter.model_xml)]), "state_path": str(snapshot),
                      "requested_position": position, "requested_rotation": rotation,
                      "actual_position": actual_pos, "actual_rotation": actual_rot,
                      "joints": adapter.sim.data.qpos[adapter.joint_qpos].copy(),
                      "features": None, "steps": 0, "termination": reason, "initialization": details}
            if valid:
                result["features"] = adapter.features().tolist()
                success, steps, termination = rollout(adapter, policy, candidate["candidate_id"], cfg)
                result.update(success=success, steps=steps, termination=termination)
                counts[key] += 1
            append_jsonl(log, result)
            completed.add(candidate["candidate_id"])
            new_trials += 1
            print(f"{new_trials}: {candidate['split']} {candidate['region']} valid={valid} result={result['success']} ({result['termination']})", flush=True)
            if max_trials is not None and new_trials >= max_trials:
                break
    finally:
        adapter.close()
    return {"new_trials": new_trials, "valid_counts": {str(k): v for k, v in counts.items()}}


def load_trials(path, independent=True):
    rows = list(read_jsonl(path))
    assert_disjoint(rows)
    seen_candidates = set()
    for row in rows:
        if row["candidate_id"] in seen_candidates:
            raise ValueError(f"Duplicate candidate log entry: {row['candidate_id']}")
        seen_candidates.add(row["candidate_id"])
        if row["valid"]:
            if row.get("success") not in (0, 1) or row.get("features") is None:
                raise ValueError("Valid trials require a binary outcome and pre-execution features")
            if not np.isfinite(row["features"]).all():
                raise ValueError("Nonfinite pre-execution features")
        elif row.get("success") is not None:
            raise ValueError("Invalid initializations cannot carry a local-operation failure label")
    if not independent:
        return rows
    seen_states, result = set(), []
    for row in rows:
        key = row.get("initial_state_id")
        if row["valid"] and key in seen_states:
            continue
        if row["valid"]:
            if key is None:
                raise ValueError("Missing initial_state_id; cannot remove deterministic duplicates")
            seen_states.add(key)
        result.append(row)
    return result
