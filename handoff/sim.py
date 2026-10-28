"""Lazy LIBERO adapter, including numerical IK against the actual MuJoCo model."""
from __future__ import annotations

from pathlib import Path
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .common import fingerprint
from .geometry import capability_features, pose_error


class LiberoAdapter:
    def __init__(self, cfg):
        from .bootstrap import configure_libero
        configure_libero()
        os.environ.setdefault("MUJOCO_GL", "glx" if os.environ.get("DISPLAY") else "egl")
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise RuntimeError(f"LIBERO import failed: {exc}. See README.md for dependencies and MUJOCO_GL.") from exc
        self.cfg = cfg
        suite = benchmark.get_benchmark_dict()[cfg["suite"]]()
        matches = [suite.get_task(i) for i in range(suite.n_tasks) if suite.get_task(i).name == cfg["task"]]
        if len(matches) != 1:
            names = [suite.get_task(i).name for i in range(suite.n_tasks)]
            raise ValueError(f"Task not found exactly: {cfg['task']}. Available tasks: {names}")
        self.task = matches[0]
        bddl = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=cfg["image_size"],
                                       camera_widths=cfg["image_size"], hard_reset=False)
        self.env.seed(cfg["seed"])
        self.env.reset()
        base = self.env
        for _ in range(10):
            if hasattr(base, "robots") and hasattr(base, "_get_observations"):
                break
            base = base.env
        self.base = base
        self._loaded_model = None
        self.model_xml = self.base.sim.model.get_xml()
        self._bind_references()

    def _bind_references(self):
        base = self.base
        self.sim = base.sim
        self.robot = base.robots[0]
        self.joint_qpos = np.asarray(self.robot._ref_joint_pos_indexes, dtype=int)
        self.joint_dof = np.asarray(self.robot._ref_joint_vel_indexes, dtype=int)
        if len(self.joint_qpos) != 7:
            raise ValueError("This adapter requires the seven-joint Panda used by LIBERO")
        self.site_id = int(self.robot.eef_site_id)
        self.site_name = self.sim.model.site_id2name(self.site_id)
        self.object_ids = dict(self.base.obj_body_id)
        for name in (self.cfg["target_object"], self.cfg["goal_object"]):
            if name not in self.object_ids:
                raise ValueError(f"Unknown object {name}; available: {list(self.object_ids)}")
        self.gripper_qpos = np.array([self.sim.model.get_joint_qpos_addr(n) for n in self.robot.gripper.joints])
        joint_ids = [self.sim.model.joint_name2id(n) for n in self.robot.robot_model.joints]
        self.joint_limits = np.asarray(self.sim.model.jnt_range[joint_ids])
        self._object_geoms = {name: self._descendant_geoms(body) for name, body in self.object_ids.items()}
        self._moving_robot_geoms = self._get_moving_robot_geoms()

    def load_model(self, xml):
        """Recorded XML carries fixed fixtures, camera parameters and physical properties."""
        key = fingerprint(xml)
        if key == self._loaded_model:
            return
        import robosuite
        from libero.libero import get_libero_path

        root = ET.fromstring(xml)
        for element in root.findall("./asset/*"):
            original = element.get("file")
            if original is None:
                continue
            parts = Path(original).parts
            if "robosuite" in parts:
                index = max(i for i, p in enumerate(parts) if p == "robosuite")
                path = Path(robosuite.__file__).resolve().parent.joinpath(*parts[index + 1:])
            elif "assets" in parts:
                index = max(i for i, p in enumerate(parts) if p == "assets")
                path = Path(get_libero_path("assets")).joinpath(*parts[index + 1:])
            else:
                path = Path(original)
            if not path.is_file():
                raise ValueError(f"Missing recorded-model asset: {original} -> {path}")
            element.set("file", str(path.resolve()))
        relocated = ET.tostring(root, encoding="unicode")
        self.env.reset_from_xml_string(relocated)
        self._bind_references()
        self.model_xml = relocated
        self._loaded_model = key

    def _descendant_geoms(self, body):
        model = self.sim.model
        descendants = {int(body)}
        for _ in range(model.nbody):
            previous = len(descendants)
            descendants.update(i for i in range(model.nbody) if int(model.body_parentid[i]) in descendants)
            if len(descendants) == previous:
                break
        return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants]

    def _get_moving_robot_geoms(self):
        model = self.sim.model
        prefixes = (self.robot.robot_model.naming_prefix, self.robot.gripper.naming_prefix)
        result = set()
        for g in range(model.ngeom):
            name = model.geom_id2name(g) or ""
            if not name.startswith(prefixes):
                continue
            body = int(model.geom_bodyid[g])
            while body:
                if model.body_jntnum[body] > 0:
                    result.add(g)
                    break
                body = int(model.body_parentid[body])
        return result

    def close(self):
        self.env.close()

    def state(self):
        return np.asarray(self.sim.get_state().flatten(), dtype=np.float64).copy()

    def set_state(self, state, zero_velocity=False):
        state = np.asarray(state)
        if state.shape != self.state().shape:
            raise ValueError(f"MuJoCo state shape mismatch: dataset {state.shape}, environment {self.state().shape}")
        self.sim.set_state_from_flattened(state)
        if zero_velocity:
            self.sim.data.qvel[:] = 0
        self.sim.forward()

    def reset(self, state, zero_velocity=False):
        # Fixed fixtures are model.body_pos/quat, not part of flattened MjSimState.
        # Deterministic reset preserves the recorded XML instead of randomizing them.
        self.base.deterministic_reset = True
        try:
            self.env.reset()
        finally:
            self.base.deterministic_reset = False
        self._bind_references()
        self.set_state(state, zero_velocity)
        self._reset_controller()
        return self.observe()

    def _reset_controller(self):
        controller = self.robot.controller
        controller.update(force=True)
        controller.reset_goal()

    def observe(self):
        obs = self.base._get_observations(force_update=True)
        # Same 180-degree image convention as OFT's LIBERO get_libero_image helper.
        global_rgb = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1, :3], dtype=np.uint8)
        wrist_rgb = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1, :3], dtype=np.uint8)
        axis_angle = Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec()
        proprio = np.r_[obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]].astype(np.float32)
        if proprio.shape != (8,):
            raise ValueError(f"Expected LIBERO proprio dimension 8, got {proprio.shape}")
        return {"global_rgb": global_rgb, "wrist_rgb": wrist_rgb, "proprio": proprio,
                "instruction": self.task.language}

    def eef_pose(self):
        return (self.sim.data.site_xpos[self.site_id].copy(),
                self.sim.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def _object(self, name):
        body = self.object_ids[name]
        geoms = self._object_geoms[name]
        position = self.sim.data.body_xpos[body].copy()
        rotation = self.sim.data.body_xmat[body].reshape(3, 3).copy()
        # Conservative local bounds; geom_rbound supports meshes without guessing mesh extents.
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
        for g in geoms:
            center = rotation.T @ (self.sim.data.geom_xpos[g] - position)
            radius = self.sim.model.geom_rbound[g]
            lower, upper = np.minimum(lower, center - radius), np.maximum(upper, center + radius)
        if not geoms or not np.isfinite(upper - lower).all():
            raise ValueError(f"Cannot determine object bounds: {name}")
        return {"name": name, "position": position, "rotation": rotation, "size": upper - lower}

    def scene(self):
        return {name: self._object(name) for name in sorted(self.object_ids)}

    def scene_id(self):
        # Initial object layout defines the scene, independent of demonstration motion or robot state.
        scene = self.scene()
        signature = {name: {k: np.round(item[k], 4).tolist() for k in ("position", "rotation")}
                     for name, item in scene.items()}
        return fingerprint([self.cfg["task"], signature])

    def rim_offset(self):
        explicit = self.cfg.get("target_rim_offset")
        if explicit is not None:
            return float(explicit)
        objects = getattr(self.base, "objects_dict", {})
        values = list(objects.values()) if isinstance(objects, dict) else list(objects)
        extra = getattr(self.base, "objects", [])
        values += list(extra.values()) if isinstance(extra, dict) else list(extra)
        for obj in values:
            if getattr(obj, "name", None) == self.cfg["target_object"] and hasattr(obj, "top_offset"):
                return float(np.asarray(obj.top_offset)[2])
        raise ValueError("Cannot resolve bowl rim from object.top_offset. Run inspect and set target_rim_offset explicitly.")

    def width(self):
        return float(np.sum(np.abs(self.sim.data.qpos[self.gripper_qpos])))

    def has_target_contact(self):
        target_geoms = set(self._object_geoms[self.cfg["target_object"]])
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if (contact.geom1 in target_geoms and contact.geom2 in self._moving_robot_geoms or
                    contact.geom2 in target_geoms and contact.geom1 in self._moving_robot_geoms):
                return True
        return False

    def collisions(self):
        result = []
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if contact.dist < -1e-5 and (contact.geom1 in self._moving_robot_geoms or contact.geom2 in self._moving_robot_geoms):
                result.append([self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)])
        return result

    def features(self):
        scene = self.scene()
        target = scene.pop(self.cfg["target_object"])
        goal = scene.pop(self.cfg["goal_object"])
        position, rotation = self.eef_pose()
        return capability_features(position, rotation, self.sim.data.qpos[self.joint_qpos], self.width(),
                                   target, goal, list(scene.values()), self.cfg["max_neighbors"])

    def success(self):
        return bool(self.base._check_success())

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.0001:
            raise ValueError("LIBERO action must be finite, seven-dimensional, and within [-1, 1]")
        _, _, done, _ = self.env.step(action.tolist())
        return self.success(), bool(done)

    def initialize(self, scene_state, position, rotation, seed):
        """Direct-state initialization measures conditional skill ability, not approach reachability."""
        self.reset(scene_state, zero_velocity=True)
        original_scene = self.scene()
        reference = self.sim.data.qpos[self.joint_qpos].copy()
        rng = np.random.default_rng(seed)
        solutions = []
        for restart in range(4):
            q = reference.copy() if restart == 0 else reference + rng.normal(0, 0.25, 7)
            q = np.clip(q, self.joint_limits[:, 0] + 1e-5, self.joint_limits[:, 1] - 1e-5)
            for _ in range(180):
                self.sim.data.qpos[self.joint_qpos] = q
                self.sim.forward()
                actual_position, actual_rotation = self.eef_pose()
                error = np.r_[position - actual_position, Rotation.from_matrix(rotation @ actual_rotation.T).as_rotvec()]
                if np.linalg.norm(error[:3]) < self.cfg["ik_position_tolerance"] and np.linalg.norm(error[3:]) < self.cfg["ik_rotation_tolerance"]:
                    solutions.append(q.copy())
                    break
                jp = self.sim.data.get_site_jacp(self.site_name).reshape(3, -1)[:, self.joint_dof]
                jr = self.sim.data.get_site_jacr(self.site_name).reshape(3, -1)[:, self.joint_dof]
                jacobian = np.vstack([jp, jr])
                inverse = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3 * np.eye(6), np.eye(6))
                delta = inverse @ error + 0.02 * (np.eye(7) - inverse @ jacobian) @ (reference - q)
                q = np.clip(q + np.clip(delta, -0.12, 0.12), self.joint_limits[:, 0] + 1e-5, self.joint_limits[:, 1] - 1e-5)
        if not solutions:
            return False, "ik_failure", {}
        for q in sorted(solutions, key=lambda item: np.linalg.norm(item - reference)):
            self.set_state(scene_state, zero_velocity=True)
            self.sim.data.qpos[self.joint_qpos] = q
            for address, name, initial in zip(self.gripper_qpos, self.robot.gripper.joints, self.robot.gripper.init_qpos):
                lo, hi = self.sim.model.jnt_range[self.sim.model.joint_name2id(name)]
                self.sim.data.qpos[address] = hi if initial >= 0 else lo
            self.sim.forward()
            if self.collisions():
                continue
            self._reset_controller()
            for _ in range(5):
                self.step([0, 0, 0, 0, 0, 0, -1])
            actual_position, actual_rotation = self.eef_pose()
            pe, re = pose_error(actual_position, actual_rotation, position, rotation)
            scene_now = self.scene()
            drift = max(np.linalg.norm(scene_now[n]["position"] - item["position"]) for n, item in original_scene.items())
            angular_drift = max(np.linalg.norm(Rotation.from_matrix(scene_now[n]["rotation"] @ item["rotation"].T).as_rotvec())
                                for n, item in original_scene.items())
            details = {"position_error": pe, "rotation_error": re, "object_drift": drift,
                       "object_angular_drift": angular_drift, "collisions": self.collisions()}
            if drift > self.cfg["object_drift_tolerance"] or angular_drift > 0.03:
                return False, "object_drift", details
            if pe > self.cfg["ik_position_tolerance"] or re > self.cfg["ik_rotation_tolerance"]:
                return False, "tracking_error", details
            if details["collisions"]:
                return False, "initial_collision", details
            if self.width() < self.cfg["open_width"] or self.success():
                return False, "invalid_gripper_or_already_successful", details
            self.sim.data.qvel[:] = 0
            self.sim.forward()
            self._reset_controller()
            return True, "initialized", details
        return False, "initial_collision", {}
