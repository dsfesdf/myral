from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation6d(matrix):
    """Two rotation-matrix columns, column-major (not the first six flattened values)."""
    return np.asarray(matrix)[:, :2].T.reshape(6)


def target_frame(matrix):
    """Use target yaw with world-up; bowl roll/pitch do not tilt height bins."""
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return Rotation.from_euler("z", yaw).as_matrix()


def sample_region(rng, ring: int, sector: int, cfg):
    lo, hi = cfg["radius_edges"][ring:ring + 2]
    radius = np.sqrt(rng.uniform(lo * lo, hi * hi))
    angle = (sector + rng.random()) * 2 * np.pi / cfg["sectors"]
    return [float(radius * np.cos(angle)), float(radius * np.sin(angle))]


def region_id(ring: int, sector: int) -> str:
    return f"r{ring}_s{sector}"


def requested_pose(target_pos, target_rotation, xy, height, rim_offset, yaw_degrees):
    frame = target_frame(target_rotation)
    position = np.asarray(target_pos) + frame @ np.array([*xy, rim_offset + height])
    # Panda's nominal top-down orientation: local z points toward the table.
    rotation = frame @ Rotation.from_euler("z", yaw_degrees, degrees=True).as_matrix() @ np.diag([1, -1, -1])
    return position, rotation


def pose_error(actual_position, actual_rotation, desired_position, desired_rotation):
    return (float(np.linalg.norm(np.asarray(actual_position) - desired_position)),
            float(np.linalg.norm(Rotation.from_matrix(desired_rotation @ actual_rotation.T).as_rotvec())))


def feature_names(max_neighbors=6):
    names = [f"eef_relative_{i}" for i in range(9)]
    names += [f"joint_{i}" for i in range(7)] + ["gripper_width"]
    names += [f"goal_relative_{i}" for i in range(9)]
    names += [f"target_size_{i}" for i in range(3)] + [f"goal_size_{i}" for i in range(3)]
    for n in range(max_neighbors):
        names += [f"neighbor_{n}_{i}" for i in range(12)] + [f"neighbor_{n}_present"]
    return names


def capability_features(eef_pos, eef_rotation, joints, width, target, goal, neighbors, max_neighbors=6):
    origin = np.asarray(target["position"])
    frame = target_frame(np.asarray(target["rotation"]))

    def relative(position, rotation):
        return np.r_[frame.T @ (np.asarray(position) - origin), rotation6d(frame.T @ np.asarray(rotation))]

    values = [relative(eef_pos, eef_rotation), np.asarray(joints), [width],
              relative(goal["position"], goal["rotation"]), target["size"], goal["size"]]
    ordered = sorted(neighbors, key=lambda item: (np.linalg.norm(np.asarray(item["position"]) - origin), item["name"]))
    if len(ordered) > max_neighbors:
        raise ValueError("Increase max_neighbors; silently truncating scene geometry is not supported")
    for index in range(max_neighbors):
        if index < len(ordered):
            item = ordered[index]
            values.append(np.r_[relative(item["position"], item["rotation"]), item["size"], 1.0])
        else:
            values.append(np.zeros(13))
    result = np.concatenate(values).astype(np.float32)
    if len(joints) != 7 or result.shape != (len(feature_names(max_neighbors)),) or not np.isfinite(result).all():
        raise ValueError("Invalid capability feature vector")
    return result
