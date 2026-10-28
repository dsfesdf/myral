from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .common import atomic_json
from .geometry import region_id
from .metrics import baseline_predictions, fit_baselines, probability_metrics, wilson
from .trials import load_trials


def region_summary(raw, independent, cfg, split="test", min_samples=20, predictions=None):
    result = []
    for ring in range(len(cfg["radius_edges"]) - 1):
        for sector in range(cfg["sectors"]):
            region = region_id(ring, sector)
            attempted = [r for r in raw if r["split"] == split and r["region"] == region]
            valid = [r for r in independent if r["split"] == split and r["region"] == region and r["valid"]]
            count, successes = len(valid), sum(r["success"] for r in valid)
            low, high = wilson(successes, count)
            reasons = Counter(r["termination"] for r in attempted if not r["valid"])
            status = "unobserved" if not attempted else "no_valid_starts" if not count else "insufficient" if count < min_samples else "measured"
            predicted = float(np.mean([predictions[r["candidate_id"]] for r in valid])) if valid and predictions else None
            result.append({"region": region, "ring": ring, "sector": sector, "attempts": len(attempted),
                           "valid_attempts": sum(r["valid"] for r in attempted), "independent_n": count,
                           "invalid_attempts": sum(not r["valid"] for r in attempted),
                           "valid_fraction": sum(r["valid"] for r in attempted) / len(attempted) if attempted else None,
                           "successes": successes, "success_rate": successes / count if count else None,
                           "wilson_low": low, "wilson_high": high, "status": status,
                           "predicted": predicted, "prediction_error": predicted - successes / count if predicted is not None else None,
                           "invalid_reasons": dict(reasons)})
    return result


def posture_summary(rows, cfg, split):
    groups = defaultdict(dict)
    for row in rows:
        if row["split"] == split and row["valid"] and row["phase"] == "posture":
            groups[row["pair_id"]][(row["height"], row["yaw_degrees"])] = row
    conditions = [(h, y) for h in cfg["heights"] for y in cfg["yaw_degrees"]]
    complete = [group for group in groups.values() if all(c in group for c in conditions)]
    baseline = (cfg["height"], 0)
    results = []
    for height, yaw in conditions:
        outcomes = [group[(height, yaw)]["success"] for group in complete]
        difference = [group[(height, yaw)]["success"] - group[baseline]["success"] for group in complete] if baseline in conditions else []
        results.append({"height": height, "yaw_degrees": yaw, "complete_pairs": len(complete),
                        "success_rate": float(np.mean(outcomes)) if outcomes else None,
                        "paired_difference_from_nominal": float(np.mean(difference)) if difference else None})
    return {"complete_pairs": len(complete), "incomplete_pairs": len(groups) - len(complete), "conditions": results}


def plot_regions(summary, cfg, output, has_predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [("success_rate", "Observed success", 0, 1)]
    if has_predictions:
        fields += [("predicted", "Predicted success", 0, 1), ("prediction_error", "Predicted - observed", -1, 1)]
    figure, axes = plt.subplots(1, len(fields), subplot_kw={"projection": "polar"}, figsize=(5 * len(fields), 5), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, cfg["sectors"] + 1)
    radii = cfg["radius_edges"]
    for axis, (field, title, vmin, vmax) in zip(axes[0], fields):
        values = np.full((len(radii) - 1, cfg["sectors"]), np.nan)
        for row in summary:
            if row[field] is not None:
                values[row["ring"], row["sector"]] = row[field]
        cmap = plt.get_cmap("coolwarm" if field == "prediction_error" else "viridis").copy()
        cmap.set_bad("lightgray")
        axis.grid(False)
        mesh = axis.pcolormesh(theta, radii, np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        for row in summary:
            center = (row["sector"] + 0.5) * 2 * np.pi / cfg["sectors"]
            inner, outer = radii[row["ring"]:row["ring"] + 2]
            if row["status"] == "insufficient":
                axis.bar(center, outer - inner, bottom=inner, width=2 * np.pi / cfg["sectors"],
                         fill=False, hatch="///", linewidth=0.2, edgecolor="gray")
        axis.set_ylim(0, radii[-1])
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis, shrink=0.7)
    figure.suptitle("Gray: no valid observation; hatched: fewer than required independent starts")
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(Path(output) / f"regions.{extension}", dpi=180)
    plt.close(figure)


def plot_reliability(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray")
    for label, model in metrics.items():
        bins = [b for b in model["reliability"] if b["n"]]
        axis.plot([b["predicted"] for b in bins], [b["observed"] for b in bins], "o-", label=label)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed success frequency")
    axis.legend()
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(Path(output) / f"reliability.{extension}", dpi=180)
    plt.close(figure)


def plot_posture(summary, cfg, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.full((len(cfg["heights"]), len(cfg["yaw_degrees"])), np.nan)
    for row in summary["conditions"]:
        if row["success_rate"] is not None:
            values[cfg["heights"].index(row["height"]), cfg["yaw_degrees"].index(row["yaw_degrees"])] = row["success_rate"]
    figure, axis = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    image = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap)
    axis.set_xticks(range(len(cfg["yaw_degrees"])), labels=cfg["yaw_degrees"])
    axis.set_yticks(range(len(cfg["heights"])), labels=[f"{h * 100:g}" for h in cfg["heights"]])
    axis.set(xlabel="Yaw offset (degrees)", ylabel="Height above bowl rim (cm)", title=f"Complete matched pairs: {summary['complete_pairs']}")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(Path(output) / f"posture.{extension}", dpi=180)
    plt.close(figure)


def report(trials, output, model=None, split="test", min_samples=20):
    raw = load_trials(trials, independent=False)
    rows = load_trials(trials)
    schema = json.loads(Path(trials).with_name("run.json").read_text())
    cfg = schema["config"]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r["valid"] and r["split"] == split]
    if not valid:
        raise ValueError(f"No valid independent trials in {split}; no success-rate estimate can be produced")
    prediction_map, metrics = {}, {}
    predictor = None
    if model:
        from .capability import CapabilityPredictor
        predictor = CapabilityPredictor(model)
        predictor.validate_run(schema, valid)
    baselines = predictor.bundle["baselines"] if predictor else fit_baselines(rows)
    predictions = baseline_predictions(valid, baselines)
    if predictor:
        predictions["mlp"], uncertainty = predictor.predict([r["features"] for r in valid])
        prediction_map = {r["candidate_id"]: float(p) for r, p in zip(valid, predictions["mlp"])}
        for row, std in zip(valid, uncertainty):
            row["ensemble_std"] = float(std)
    for name, values in predictions.items():
        metrics[name] = probability_metrics([r["success"] for r in valid], values)
    result = {"split": split, "policy_id": schema["policy_id"], "raw_attempts": len([r for r in raw if r["split"] == split]),
              "independent_valid_trials": len(valid), "scene_count": len({r["scene_id"] for r in valid}), "metrics": metrics,
              "interval_note": "Wilson intervals describe sampled starts; repeated starts are removed. They do not establish across-scene or across-training-seed significance."}
    atomic_json(destination / "metrics.json", result)
    summary = region_summary(raw, rows, cfg, split, min_samples, prediction_map)
    atomic_json(destination / "regions.json", summary)
    with (destination / "regions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows({**r, "invalid_reasons": json.dumps(r["invalid_reasons"])} for r in summary)
    atomic_json(destination / "predictions.json", [{"candidate_id": r["candidate_id"], "scene_id": r["scene_id"],
                "success": r["success"], **{name: float(values[index]) for name, values in predictions.items()},
                "ensemble_std": r.get("ensemble_std")} for index, r in enumerate(valid)])
    plot_regions(summary, cfg, destination, bool(model))
    plot_reliability(metrics, destination)
    if any(r["phase"] == "posture" for r in valid):
        posture = posture_summary(rows, cfg, split)
        atomic_json(destination / "posture.json", posture)
        plot_posture(posture, cfg, destination)
    return result
