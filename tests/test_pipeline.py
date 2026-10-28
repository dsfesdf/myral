from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from handoff.common import append_jsonl, assert_disjoint, atomic_json, group_split, load_config, run_lock
from handoff.data import (DATASET_NAME, action_chunk, compute_statistics, crop_start,
                          libero_to_oft_actions, normalize, oft_to_libero_actions)
from handoff.geometry import capability_features, feature_names, requested_pose, sample_region
from handoff.metrics import probability_metrics, wilson
from handoff.report import posture_summary, region_summary
from handoff.trials import load_trials, plan_starts, rollout

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def configuration():
    return load_config(ROOT / "configs" / "pilot.json")


def sample_trial(index, split="train", label=1, valid=True):
    return {"candidate_id": f"{split}-{index}", "scene_id": f"scene-{split}-{index // 8}",
            "initial_state_id": f"state-{split}-{index}", "pair_id": f"pair-{split}-{index}",
            "policy_id": "synthetic-unit-test-policy", "split": split, "region": "r0_s0", "ring": 0,
            "sector": 0, "height": 0.12, "yaw_degrees": 0, "phase": "position", "valid": valid,
            "success": label if valid else None, "features": [0.] * len(feature_names()) if valid else None,
            "termination": "success" if valid else "ik_failure"}


def make_prepared(root):
    cfg = configuration()
    rows = []
    for index, split in enumerate(("train", "calibration", "test")):
        episode = root / "episodes" / split
        episode.mkdir(parents=True)
        row = {"episode_id": split, "scene_id": f"scene-{split}", "split": split,
               "path": f"episodes/{split}", "length": 3, "instruction": "put bowl on plate"}
        actions = np.zeros((3, 7), dtype=np.float32)
        actions[:, :6] = 0.1 if split == "train" else 0.9
        actions[:, 6] = [0, 1, 1]
        for name, array in {"actions": actions, "proprio": np.full((3, 8), index),
                            "global_rgb": np.zeros((3, 4, 4, 3), dtype=np.uint8),
                            "wrist_rgb": np.zeros((3, 4, 4, 3), dtype=np.uint8),
                            "scene_state": np.arange(10)}.items():
            np.save(episode / f"{name}.npy", array)
        atomic_json(episode / "metadata.json", row)
        (episode / "model.xml").write_text("<mujoco/>")
        rows.append(row)
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    atomic_json(root / "preparation.json", {"config": cfg})
    compute_statistics(root)
    return cfg


class DataTests(unittest.TestCase):
    def test_crop_rejects_contact_and_closed_gripper(self):
        self.assertEqual(crop_start([.3, .1, .1, .1], [.08, .02, .08, .08], [False, False, True, False], preroll=1), 2)
        self.assertIsNone(crop_start([.3], [.08], [False]))
        self.assertEqual(crop_start([.1], [.08], [False]), 0)

    def test_gripper_roundtrip_and_tail_chunk(self):
        native = np.zeros((3, 7), dtype=np.float32)
        native[:, -1] = [-1, 1, -1]
        normalized = libero_to_oft_actions(native)
        np.testing.assert_array_equal(normalized[:, -1], [1, 0, 1])
        np.testing.assert_array_equal(oft_to_libero_actions(normalized), native)
        chunk = action_chunk(normalized, 2)
        self.assertEqual(chunk.shape, (8, 7))
        np.testing.assert_array_equal(chunk, np.repeat(normalized[-1:], 8, axis=0))

    def test_statistics_exclude_held_out_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_prepared(Path(tmp))
            stats = json.loads((Path(tmp) / "dataset_statistics.json").read_text())[DATASET_NAME]
            np.testing.assert_allclose(stats["action"]["mean"][:6], .1)
            np.testing.assert_allclose(stats["proprio"]["mean"], 0)
            self.assertFalse(stats["action"]["mask"][-1])

    def test_normalization_matches_oft_with_constant_channel(self):
        actual = normalize([[.5, 0, 1]], {"q01": [0, 0, 0], "q99": [1, 0, 1], "mask": [True, True, False]})
        np.testing.assert_allclose(actual, [[0, -1, 1]], atol=1e-7)


class GeometryTests(unittest.TestCase):
    def test_area_uniform_annulus_and_top_down_orientation(self):
        cfg = configuration()
        rng = np.random.default_rng(5)
        points = np.array([sample_region(rng, 0, 0, cfg) for _ in range(10000)])
        radius2 = np.sum(points**2, axis=1)
        self.assertTrue((radius2 >= .04**2).all() and (radius2 < .08**2).all())
        self.assertAlmostEqual(radius2.mean(), (.04**2 + .08**2) / 2, delta=5e-5)
        position, rotation = requested_pose([1, 2, 3], np.eye(3), [.1, 0], .12, .03, 15)
        np.testing.assert_allclose(position, [1.1, 2, 3.15])
        np.testing.assert_allclose(rotation[:, 2], [0, 0, -1])

    def test_features_are_invariant_to_scene_translation_and_yaw(self):
        target = {"name": "bowl", "position": np.array([1., 2., 0]), "rotation": np.eye(3), "size": [.1, .1, .05]}
        goal = {"name": "plate", "position": np.array([1.2, 2., 0]), "rotation": np.eye(3), "size": [.2, .2, .02]}
        eef = np.array([1., 2.1, .2])
        first = capability_features(eef, np.eye(3), np.zeros(7), .08, target, goal, [])
        transform = Rotation.from_euler("z", .7).as_matrix()
        translation = np.array([4, -3, 1])
        def moved(obj):
            return {**obj, "position": transform @ obj["position"] + translation, "rotation": transform @ obj["rotation"]}
        second = capability_features(transform @ eef + translation, transform, np.zeros(7), .08, moved(target), moved(goal), [])
        np.testing.assert_allclose(first, second, atol=1e-7)
        self.assertEqual(len(first), len(feature_names()))


class IntegrityTests(unittest.TestCase):
    def test_only_one_collector_can_write_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(tmp):
                with self.assertRaisesRegex(RuntimeError, "Another collector"):
                    with run_lock(tmp):
                        pass
            with run_lock(tmp):
                pass

    def test_plan_rejects_changed_scene_assets(self):
        from handoff.trials import collect
        class Adapter:
            def close(self): pass
        class Policy:
            policy_id = "frozen-test-policy"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_prepared(root)
            cfg["counts"] = {s: 1 for s in cfg["counts"]}
            plan_starts(root, root / "plan.jsonl", cfg)
            for path in (root / "episodes").glob("*/model.xml"):
                path.write_text("<mujoco model='changed'/>")
            with self.assertRaisesRegex(ValueError, "Scene source changed"):
                collect(root / "plan.jsonl", root / "run", None, adapter_factory=lambda cfg: Adapter(), policy=Policy())

    def test_scene_state_pair_and_policy_leakage_are_rejected(self):
        first, second = sample_trial(0), sample_trial(1, "test")
        for field in ("scene_id", "initial_state_id", "pair_id"):
            changed = {**second, field: first[field]}
            with self.assertRaisesRegex(ValueError, "leakage"):
                assert_disjoint([first, changed])
        with self.assertRaisesRegex(ValueError, "frozen policy"):
            assert_disjoint([first, {**second, "policy_id": "another"}])
        self.assertEqual(group_split("scene-1", 42), group_split("scene-1", 42))

    def test_invalid_is_not_failure_and_repeated_state_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "trials.jsonl"
            first = sample_trial(0)
            append_jsonl(log, first)
            append_jsonl(log, {**sample_trial(1), "initial_state_id": first["initial_state_id"]})
            append_jsonl(log, sample_trial(2, valid=False))
            self.assertEqual(len(load_trials(log)), 2)
            append_jsonl(log, {**sample_trial(3, valid=False), "success": 0})
            with self.assertRaisesRegex(ValueError, "Invalid initializations"):
                load_trials(log)

    def test_plans_pair_positions_without_crossing_scene_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_prepared(root)
            cfg["posture_counts"] = {s: 1 for s in cfg["counts"]}
            plan = root / "plan.jsonl"
            plan_starts(root, plan, cfg, "posture")
            rows = [json.loads(line) for line in plan.read_text().splitlines()]
            self.assertEqual(len(rows), 24 * 3 * 9)
            same = [r for r in rows if r["pair_id"] == rows[0]["pair_id"]]
            self.assertEqual(len(same), 9)
            self.assertEqual(len({tuple(r["xy"]) for r in same}), 1)
            assert_disjoint(rows)


class ExecutionTests(unittest.TestCase):
    def test_receding_horizon_and_timeout_boundary(self):
        class Policy:
            def __init__(self): self.resets, self.calls = [], 0
            def reset(self, episode): self.resets.append(episode)
            def predict(self, obs):
                self.calls += 1
                return np.repeat(np.arange(8)[:, None] / 8, 7, axis=1)
        class Adapter:
            def __init__(self): self.actions = []
            def observe(self): return {}
            def step(self, action):
                self.actions.append(action)
                return False, False
        adapter, policy = Adapter(), Policy()
        cfg = {**configuration(), "max_steps": 3}
        result = rollout(adapter, policy, "one", cfg)
        self.assertEqual(result, (0, 3, "timeout"))
        self.assertEqual(policy.resets, ["one"])
        self.assertEqual(policy.calls, 2)
        np.testing.assert_allclose([a[0] for a in adapter.actions], [0, 1/8, 0])

    def test_atomic_file_service_preserves_episode_reset(self):
        from handoff.ipc import FilePolicyClient, serve
        class Policy:
            policy_id = "test-checkpoint"
            def __init__(self): self.resets = []
            def reset(self, episode): self.resets.append(episode)
            def predict(self, obs): return np.zeros((8, 7))
        with tempfile.TemporaryDirectory() as tmp:
            policy, stop = Policy(), threading.Event()
            thread = threading.Thread(target=serve, args=(policy, tmp), kwargs={"stop": stop, "poll_interval": .001})
            thread.start()
            try:
                deadline = time.monotonic() + 3
                while not (Path(tmp) / "ready.json").exists() and time.monotonic() < deadline:
                    time.sleep(.005)
                client = FilePolicyClient(tmp, timeout=3)
                client.reset("episode-one")
                observation = {"global_rgb": np.zeros((4, 4, 3), dtype=np.uint8), "wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                               "proprio": np.zeros(8), "instruction": "place bowl"}
                self.assertEqual(client.predict(observation).shape, (8, 7))
                client.predict(observation)
                self.assertEqual(policy.resets, ["episode-one"])
            finally:
                stop.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())


class StatisticsTests(unittest.TestCase):
    def test_empty_is_not_zero_success_and_probability_endpoints(self):
        cfg = configuration()
        rows = [sample_trial(0, "test", valid=False)]
        summary = region_summary(rows, rows, cfg)
        self.assertIsNone(summary[0]["success_rate"])
        self.assertEqual(summary[0]["status"], "no_valid_starts")
        self.assertEqual(summary[1]["status"], "unobserved")
        self.assertEqual(wilson(0, 0), (None, None))
        lo, hi = wilson(5, 10)
        self.assertLess(lo, .5)
        self.assertGreater(hi, .5)
        metrics = probability_metrics([0, 1], [0, 1])
        self.assertEqual(metrics["brier"], 0)
        self.assertEqual(metrics["ece"], 0)

    def test_posture_effects_use_complete_matched_pairs(self):
        cfg = configuration()
        rows = []
        for height in cfg["heights"]:
            for yaw in cfg["yaw_degrees"]:
                rows.append({**sample_trial(len(rows), "test"), "phase": "posture", "pair_id": "one",
                             "height": height, "yaw_degrees": yaw, "success": int(yaw == 0)})
        summary = posture_summary(rows, cfg, "test")
        self.assertEqual(summary["complete_pairs"], 1)
        self.assertEqual(summary["conditions"][0]["paired_difference_from_nominal"], -1)
        self.assertEqual(posture_summary(rows[:-1], cfg, "test")["complete_pairs"], 0)


@unittest.skipUnless(HAS_TORCH, "PyTorch is optional in the data-only environment")
class TorchIntegrationTests(unittest.TestCase):
    def test_oft_batch_bridge_preserves_horizon_and_proprio_shape(self):
        from handoff.oft_dataset import LocalOFTDataset
        with tempfile.TemporaryDirectory() as tmp:
            make_prepared(Path(tmp))
            dataset = LocalOFTDataset(tmp, DATASET_NAME, lambda batch: batch, resize_resolution=(4, 4), train=False)
            frame = next(iter(dataset))
            self.assertEqual(frame["action"].shape, (8, 7))
            self.assertEqual(frame["observation"]["proprio"].shape, (1, 8))
            self.assertEqual(frame["task"]["language_instruction"], b"put bowl on plate")

    def test_capability_training_and_sealed_report(self):
        import torch
        from handoff.capability import CapabilityPredictor, fit
        from handoff.report import report
        torch.set_num_threads(1)
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "trials.jsonl"
            for split, count in (("train", 128), ("calibration", 32), ("test", 64)):
                for index in range(count):
                    signal = float(rng.uniform(-1, 1))
                    row = sample_trial(index, split, int(signal > 0))
                    row["features"][0] = signal
                    append_jsonl(log, row)
            schema = {"config": configuration(), "feature_names": feature_names(), "policy_id": "synthetic-unit-test-policy"}
            atomic_json(root / "run.json", schema)
            fit(log, root / "model", epochs=12, batch_size=32, learning_rate=.002, patience=8)
            predictor = CapabilityPredictor(root / "model" / "model.pt")
            rows = load_trials(log)
            test = [r for r in rows if r["split"] == "test"]
            p, std = predictor.predict([r["features"] for r in test])
            self.assertLess(probability_metrics([r["success"] for r in test], p)["brier"], .20)
            self.assertTrue((std >= 0).all())
            with self.assertRaisesRegex(ValueError, "different policy"):
                predictor.validate_run({**schema, "policy_id": "new-policy"}, test)
            result = report(log, root / "report", root / "model" / "model.pt")
            self.assertEqual(result["independent_valid_trials"], 64)
            self.assertTrue((root / "report" / "regions.svg").exists())
            self.assertTrue((root / "report" / "reliability.png").exists())


if __name__ == "__main__":
    unittest.main()
