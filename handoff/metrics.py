from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def wilson(successes, count, z=1.959963984540054):
    if count == 0:
        return None, None
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return max(0.0, center - radius), min(1.0, center + radius)


def reliability(y, probabilities, bins=10):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    result = []
    for index in range(bins):
        selected = indices == index
        result.append({"lower": index / bins, "upper": (index + 1) / bins, "n": int(selected.sum()),
                       "predicted": float(probabilities[selected].mean()) if selected.any() else None,
                       "observed": float(y[selected].mean()) if selected.any() else None})
    return result


def probability_metrics(y, probabilities):
    y, probabilities = np.asarray(y, dtype=float), np.asarray(probabilities, dtype=float)
    if y.shape != probabilities.shape or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Metrics require equal, nonempty binary labels and probability arrays")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Invalid probabilities")
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    table = reliability(y, probabilities)
    return {"n": len(y), "success_rate": float(y.mean()), "brier": float(np.mean((probabilities - y)**2)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
            "ece": float(sum(r["n"] / len(y) * abs(r["predicted"] - r["observed"]) for r in table if r["n"])),
            "reliability": table}


def fit_baselines(rows):
    rows = [r for r in rows if r["valid"] and r["split"] == "train"]
    if not rows:
        raise ValueError("No training outcomes for baselines")
    global_rate = float(np.mean([r["success"] for r in rows]))
    groups = defaultdict(list)
    for row in rows:
        groups[row["region"]].append(row["success"])
    return {"global": global_rate, "region": {key: (sum(values) + 1) / (len(values) + 2) for key, values in groups.items()}}


def baseline_predictions(rows, model):
    return {"constant": np.full(len(rows), model["global"]),
            "region_table": np.array([model["region"].get(r["region"], model["global"]) for r in rows])}
