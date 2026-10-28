from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from torch import nn

from .common import atomic_json, file_sha256, fingerprint
from .metrics import fit_baselines, probability_metrics
from .trials import load_trials


class CapabilityMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def fit(trials, output, epochs=100, batch_size=128, learning_rate=1e-3, patience=15, seed=42, device="cpu"):
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, patience and learning_rate must be positive")
    root = Path(output)
    if (root / "model.pt").exists():
        raise ValueError("Choose a fresh model output directory")
    root.mkdir(parents=True, exist_ok=True)
    rows = load_trials(trials)
    train = [r for r in rows if r["valid"] and r["split"] == "train"]
    calibration = [r for r in rows if r["valid"] and r["split"] == "calibration"]
    if len(train) < 10 or not calibration or len({r["success"] for r in train}) < 2:
        raise ValueError("Need at least ten training trials, both outcome classes, and separate calibration scenes")
    schema = json.loads(Path(trials).with_name("run.json").read_text())
    names = schema["feature_names"]
    x = np.asarray([r["features"] for r in train], dtype=np.float32)
    xc = np.asarray([r["features"] for r in calibration], dtype=np.float32)
    if x.shape[1] != len(names) or xc.shape[1] != len(names):
        raise ValueError("Feature vector does not match run.json schema")
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    xt = torch.tensor((x - mean) / std, device=device)
    yt = torch.tensor([r["success"] for r in train], dtype=torch.float32, device=device)
    xct = torch.tensor((xc - mean) / std, device=device)
    yct = torch.tensor([r["success"] for r in calibration], dtype=torch.float32, device=device)
    states, calibration_logits, history = [], [], []
    for member in range(3):
        torch.manual_seed(seed + member)
        model = CapabilityMLP(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        best, best_state, stale = float("inf"), None, 0
        log = []
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(xt), device=device)
            for start in range(0, len(xt), batch_size):
                selected = order[start:start + batch_size]
                loss = nn.functional.binary_cross_entropy_with_logits(model(xt[selected]), yt[selected])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                validation = float(nn.functional.binary_cross_entropy_with_logits(model(xct), yct).item())
            log.append({"epoch": epoch + 1, "calibration_bce": validation})
            if validation < best - 1e-5:
                best, stale = validation, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
            if stale >= patience:
                break
        model.load_state_dict(best_state)
        with torch.inference_mode():
            calibration_logits.append(model(xct).cpu().numpy())
        states.append(best_state)
        history.append(log)
        print(f"capability member {member + 1}: epochs={len(log)}, calibration BCE={best:.4f}", flush=True)
    logits = np.asarray(calibration_logits)
    yc = yct.cpu().numpy()

    def objective(log_temperature):
        predictions = expit(logits / np.exp(log_temperature)).mean(0)
        return probability_metrics(yc, predictions)["log_loss"]

    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    temperature = float(np.exp(result.x))
    bundle = {"format_version": 1, "dimension": len(names), "feature_names": names, "mean": mean.tolist(),
              "std": std.tolist(), "states": states, "temperature": temperature, "policy_id": schema["policy_id"],
              "baselines": fit_baselines(rows), "trials_sha256": file_sha256(Path(trials)),
              "train_scene_ids": sorted({r["scene_id"] for r in train}),
              "calibration_scene_ids": sorted({r["scene_id"] for r in calibration})}
    torch.save(bundle, root / "model.pt")
    metadata = {k: v for k, v in bundle.items() if k != "states"}
    metadata["calibration_metrics"] = probability_metrics(yc, expit(logits / temperature).mean(0))
    metadata["history"] = history
    atomic_json(root / "training.json", metadata)
    return {"model": str(root / "model.pt"), "train_n": len(train), "calibration_n": len(calibration), "temperature": temperature}


class CapabilityPredictor:
    def __init__(self, path, device="cpu"):
        self.bundle = torch.load(path, map_location="cpu", weights_only=False)
        self.device = device
        self.models = []
        for state in self.bundle["states"]:
            model = CapabilityMLP(self.bundle["dimension"]).to(device)
            model.load_state_dict(state)
            self.models.append(model.eval())

    def predict(self, features):
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.bundle["dimension"] or not np.isfinite(values).all():
            raise ValueError("Feature schema mismatch")
        values = (values - np.asarray(self.bundle["mean"])) / np.asarray(self.bundle["std"])
        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits = np.array([model(tensor).cpu().numpy() for model in self.models])
        probabilities = expit(logits / self.bundle["temperature"])
        return probabilities.mean(0), probabilities.std(0)

    def validate_run(self, schema, rows):
        if schema["policy_id"] != self.bundle["policy_id"] or schema["feature_names"] != self.bundle["feature_names"]:
            raise ValueError("Capability model belongs to a different policy or feature schema")
        seen = set(self.bundle["train_scene_ids"]) | set(self.bundle["calibration_scene_ids"])
        if any(r["split"] == "test" and r["scene_id"] in seen for r in rows):
            raise ValueError("Model training/calibration scene occurs in sealed test set")
