"""Opt-in tests against downloaded public demonstrations, never a policy benchmark."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from handoff.common import load_config, read_jsonl
from handoff.data import load_manifest
from handoff.geometry import requested_pose

DATA = os.environ.get("HANDOFF_SIM_DATA")


@unittest.skipUnless(DATA, "Set HANDOFF_SIM_DATA to prepared real data to run renderer/physics tests")
class RealSimulationTests(unittest.TestCase):
    def setUp(self):
        from handoff.sim import LiberoAdapter
        self.root = Path(DATA).resolve()
        self.cfg = load_config(Path(__file__).resolve().parents[1] / "configs/pilot.json")
        self.rows = load_manifest(self.root)
        self.adapter = LiberoAdapter(self.cfg)

    def tearDown(self):
        self.adapter.close()

    def test_model_and_state_restore_both_cameras_exactly(self):
        episode = self.root / self.rows[0]["path"]
        self.adapter.load_model((episode / "model.xml").read_text())
        state = np.load(episode / "states.npy")[0]
        for _ in range(2):
            self.adapter.reset(state)
            obs = self.adapter.observe()
            for key in ("global_rgb", "wrist_rgb", "proprio"):
                np.testing.assert_array_equal(obs[key], np.load(episode / f"{key}.npy", mmap_mode="r")[0])

    def test_numerical_ik_preserves_scene_and_meets_pose_tolerance(self):
        episode = self.root / self.rows[0]["path"]
        self.adapter.load_model((episode / "model.xml").read_text())
        state = np.load(episode / "scene_state.npy")
        self.adapter.reset(state, zero_velocity=True)
        target = self.adapter.scene()[self.cfg["target_object"]]
        position, rotation = requested_pose(target["position"], target["rotation"], [.06, 0], .12, self.adapter.rim_offset(), 0)
        valid, reason, details = self.adapter.initialize(state, position, rotation, 42)
        self.assertTrue(valid, (reason, details))
        self.assertLess(details["position_error"], self.cfg["ik_position_tolerance"])
        self.assertEqual(len(self.adapter.features()), 110)

    def test_real_collection_snapshots_and_resume_with_noop_test_policy(self):
        from handoff.trials import collect, load_trials, plan_starts
        class NoopTestPolicy:
            policy_id = "integration-test-noop-not-a-vla"
            def reset(self, identifier): pass
            def predict(self, observation):
                actions = np.zeros((8, 7))
                actions[:, -1] = -1
                return actions
        cfg = {**self.cfg, "max_steps": 2, "counts": {s: 1 for s in self.cfg["counts"]}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_starts(self.root, root / "plan.jsonl", cfg)
            kwargs = {"plan": root / "plan.jsonl", "output": root / "run", "ipc": None,
                      "policy": NoopTestPolicy(), "adapter_factory": lambda cfg: self.adapter}
            # Keep the shared adapter alive until tearDown while collect owns the interface.
            actual_close = self.adapter.close
            self.adapter.close = lambda: None
            try:
                collect(**kwargs, max_trials=2)
                collect(**kwargs, max_trials=1)
            finally:
                self.adapter.close = actual_close
            rows = load_trials(root / "run" / "trials.jsonl", independent=False)
            self.assertEqual(len(rows), 3)
            self.assertTrue(any(r["valid"] for r in rows))
            for row in rows:
                with np.load(root / "run" / row["state_path"], allow_pickle=False) as state:
                    self.assertIn("model_xml", state)
                    self.assertEqual(state["initial_state"].ndim, 1)


if __name__ == "__main__":
    unittest.main()
