from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .common import atomic_json, json_default, load_config


def doctor():
    modules = ("numpy", "scipy", "h5py", "PIL", "matplotlib", "torch", "libero", "robosuite", "transformers", "prismatic")
    result = {"python": sys.version, "executable": sys.executable,
              "modules": {name: importlib.util.find_spec(name) is not None for name in modules}}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        result["gpus"] = gpu.stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        result["gpus"] = []
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="LIBERO local VLA and takeover capability experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report installed dependencies and GPU")
    init = commands.add_parser("init-libero", help="Create noninteractive project-local LIBERO paths")
    init.add_argument("--repo", default="third_party/LIBERO")
    init.add_argument("--config-dir")
    download = commands.add_parser("download", help="Download only the selected task from pinned public data")
    download.add_argument("--config", default="configs/pilot.json")
    download.add_argument("--output", default="data/raw")
    inspect = commands.add_parser("inspect", help="Load the real task, inspect object names, save both camera images")
    inspect.add_argument("--config", default="configs/pilot.json")
    inspect.add_argument("--output", default="runs/inspection")
    prepare = commands.add_parser("prepare", help="Replay raw LIBERO HDF5 and crop successful local demonstrations")
    prepare.add_argument("--config", default="configs/pilot.json")
    prepare.add_argument("--raw", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=int)
    replay = commands.add_parser("replay", help="Verify prepared camera/proprio alignment and execute expert actions")
    replay.add_argument("--data", required=True)
    replay.add_argument("--count", type=int, default=10)
    replay.add_argument("--output", default="runs/replay.json")
    train = commands.add_parser("train-oft", help="Run upstream OFT with local cropped data")
    train.add_argument("--repo", default="third_party/openvla-oft")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--devices", type=int, default=1)
    train.add_argument("--micro-batch", type=int, default=4)
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--base", default="openvla/openvla-7b")
    train.add_argument("--dry-run", action="store_true")
    check = commands.add_parser("check-oft", help="Run the real OFT tokenizer, image transform and collator without 7B weights")
    check.add_argument("--repo", default="third_party/openvla-oft")
    check.add_argument("--data", required=True)
    check.add_argument("--processor", default="openvla/openvla-7b")
    serve = commands.add_parser("serve-oft", help="Load a frozen checkpoint and expose atomic-file inference")
    serve.add_argument("--repo", default="third_party/openvla-oft")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--ipc", required=True)
    starts = commands.add_parser("plan-starts", help="Create an immutable, grouped spatial or paired-posture plan")
    starts.add_argument("--config", default="configs/pilot.json")
    starts.add_argument("--data", required=True)
    starts.add_argument("--output", required=True)
    starts.add_argument("--phase", choices=("position", "posture"), default="position")
    collect = commands.add_parser("collect", help="Initialize candidates and evaluate a frozen policy; resumable")
    collect.add_argument("--plan", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--ipc", required=True)
    collect.add_argument("--max-trials", type=int)
    collect.add_argument("--timeout", type=float, default=120)
    fit = commands.add_parser("train-capability", help="Fit three state MLPs and calibrate on separate scenes")
    fit.add_argument("--trials", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--batch-size", type=int, default=128)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--patience", type=int, default=15)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", default="cpu")
    report = commands.add_parser("report", help="Export baselines, calibration, Wilson intervals and standalone figures")
    report.add_argument("--trials", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--model")
    report.add_argument("--split", choices=("train", "calibration", "test"), default="test")
    report.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "init-libero":
            from .bootstrap import configure_libero
            result = configure_libero(args.repo, args.config_dir)
        elif args.command == "download":
            from .bootstrap import download_task
            result = download_task(load_config(args.config), args.output)
        elif args.command == "inspect":
            from PIL import Image
            from .sim import LiberoAdapter
            adapter = LiberoAdapter(load_config(args.config))
            try:
                directory = Path(args.output)
                directory.mkdir(parents=True, exist_ok=True)
                obs = adapter.observe()
                for name in ("global_rgb", "wrist_rgb"):
                    Image.fromarray(obs[name]).save(directory / f"{name}.png")
                result = {"instruction": adapter.task.language, "objects": adapter.scene(),
                          "rim_offset": adapter.rim_offset(), "proprio": obs["proprio"],
                          "feature_dimension": len(adapter.features())}
                atomic_json(directory / "inspection.json", result)
            finally:
                adapter.close()
        elif args.command == "prepare":
            from .data import prepare
            result = prepare(args.raw, args.output, load_config(args.config), args.limit)
        elif args.command == "replay":
            from .data import replay_check
            result = replay_check(args.data, args.count)
            atomic_json(args.output, result)
            if not result["passed"]:
                print(json.dumps(result, indent=2, default=json_default))
                parser.exit(1, "Replay mismatch: inspect the report before training.\n")
        elif args.command == "train-oft":
            from .oft import launch_training
            launch_training(args.repo, args.data, args.output, args.devices, args.micro_batch, args.steps, args.base, args.dry_run)
            return
        elif args.command == "check-oft":
            from .oft import check_batch
            result = check_batch(args.repo, args.data, args.processor)
        elif args.command == "serve-oft":
            from .oft import OFTPolicy
            from .ipc import serve
            serve(OFTPolicy(args.repo, args.checkpoint), args.ipc)
            return
        elif args.command == "plan-starts":
            from .trials import plan_starts
            result = plan_starts(args.data, args.output, load_config(args.config), args.phase)
        elif args.command == "collect":
            from .trials import collect
            result = collect(args.plan, args.output, args.ipc, args.max_trials, args.timeout)
        elif args.command == "train-capability":
            from .capability import fit
            result = fit(args.trials, args.output, args.epochs, args.batch_size, args.learning_rate, args.patience, args.seed, args.device)
        else:
            from .report import report
            result = report(args.trials, args.output, args.model, args.split, args.min_samples)
        print(json.dumps(result, indent=2, default=json_default))
    except (ValueError, RuntimeError, ImportError, FileNotFoundError, TimeoutError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
