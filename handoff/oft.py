"""Launch upstream OFT with a local-data bridge; serve through its official helpers."""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from .common import checkpoint_id
from .data import DATASET_NAME, oft_to_libero_actions


def configure_repo(repo):
    repo = Path(repo).resolve()
    if not (repo / "vla-scripts" / "finetune.py").is_file():
        raise ValueError("--repo must point to an OpenVLA-OFT checkout")
    sys.path.insert(0, str(repo))
    return repo


def verify_constants():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # TF is only preprocessing; reserve CUDA memory for PyTorch.
    from prismatic.vla import constants

    expected = {"ACTION_DIM": 7, "PROPRIO_DIM": 8, "NUM_ACTIONS_CHUNK": 8}
    for name, value in expected.items():
        if getattr(constants, name, None) != value:
            raise RuntimeError(f"OFT {name} must be {value}; select its LIBERO configuration explicitly")
    if constants.ACTION_PROPRIO_NORMALIZATION_TYPE != "bounds_q99":
        raise RuntimeError("Local dataset statistics require OFT bounds_q99 normalization")


def check_batch(repo, data, processor_path="openvla/openvla-7b"):
    configure_repo(repo)
    verify_constants()
    import torch
    from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform
    from .oft_dataset import LocalOFTDataset

    processor = PrismaticProcessor.from_pretrained(processor_path, trust_remote_code=True)
    transform = RLDSBatchTransform(ActionTokenizer(processor.tokenizer), processor.tokenizer,
                                    image_transform=processor.image_processor.apply_transform,
                                    prompt_builder_fn=PurePromptBuilder, use_wrist_image=True, use_proprio=True)
    dataset = LocalOFTDataset(data, DATASET_NAME, transform, resize_resolution=(224, 224), train=False)
    iterator = iter(dataset)
    samples = [next(iterator), next(iterator)]
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                                processor.tokenizer.pad_token_id, padding_side="right")
    batch = collator(samples)
    if tuple(batch["actions"].shape) != (2, 8, 7) or tuple(batch["proprio"].shape) != (2, 8):
        raise RuntimeError(f"Unexpected collated action/proprio shapes: {batch['actions'].shape}, {batch['proprio'].shape}")
    processor.save_pretrained(Path(data) / "processor")
    return {"batch_shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
            "processor_saved": str(Path(data) / "processor"), "model_weights_loaded": False}


def training_command(repo, data, output, devices=1, micro_batch=4, steps=5000, base="openvla/openvla-7b"):
    if micro_batch < 2:
        raise ValueError("Upstream OFT collator squeezes away the batch dimension at batch_size=1; use micro_batch >= 2")
    if devices < 1 or 32 % (devices * micro_batch):
        raise ValueError("devices * micro_batch must divide effective batch size 32")
    repo = Path(repo).resolve()
    source = repo / "vla-scripts" / "finetune.py"
    if not source.is_file():
        raise ValueError("Missing upstream vla-scripts/finetune.py")
    options = {
        "vla_path": base, "data_root_dir": str(Path(data).resolve()), "dataset_name": DATASET_NAME,
        "run_root_dir": str(Path(output).resolve()), "use_l1_regression": "True", "use_diffusion": "False",
        "use_film": "False", "num_images_in_input": 2, "use_proprio": "True", "lora_rank": 32,
        "batch_size": micro_batch, "grad_accumulation_steps": 32 // (devices * micro_batch),
        "max_steps": steps, "learning_rate": 1e-4, "image_aug": "False", "save_freq": min(1000, steps),
        "use_val_set": "True", "val_freq": min(1000, steps), "wandb_project": "libero-handoff",
    }
    tree = ast.parse(source.read_text())
    fields = {node.target.id for cls in tree.body if isinstance(cls, ast.ClassDef)
              for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    missing = set(options) - fields
    if missing:
        raise RuntimeError(f"OFT API mismatch: missing configuration fields {sorted(missing)}. Pin a compatible checkout.")
    seed = json.loads((Path(data) / "preparation.json").read_text())["config"]["seed"]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={devices}",
               "-m", "handoff.oft", "--repo", str(repo), "--seed", str(seed), "--"]
    for key, value in options.items():
        command += [f"--{key}", str(value)]
    return command


def launch_training(repo, data, output, devices=1, micro_batch=4, steps=5000, base="openvla/openvla-7b", dry_run=False):
    from .data import load_manifest
    rows = load_manifest(data)
    if not any(r["split"] == "calibration" for r in rows):
        raise ValueError("Need a separate calibration scene group for model selection")
    command = training_command(repo, data, output, devices, micro_batch, steps, base)
    print(json.dumps({"command": command, "effective_batch": 32, "freeze_vision": True}, indent=2))
    if not dry_run:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("WANDB_MODE", "offline")
        subprocess.run(command, env=env, check=True)


class OFTPolicy:
    def __init__(self, repo, checkpoint, unnorm_key=DATASET_NAME):
        configure_repo(repo)
        from .bootstrap import configure_libero
        configure_libero()
        # Upstream chooses constants from argv; this explicit marker selects LIBERO.
        if not any("libero" in item.lower() for item in sys.argv):
            sys.argv.append("libero")
        verify_constants()
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action

        self.cfg = SimpleNamespace(
            model_family="openvla", pretrained_checkpoint=str(Path(checkpoint).resolve()),
            use_l1_regression=True, use_diffusion=False, use_film=False,
            num_images_in_input=2, use_proprio=True, center_crop=False, lora_rank=32,
            load_in_8bit=False, load_in_4bit=False, unnorm_key=unnorm_key,
            num_diffusion_steps_train=50, num_diffusion_steps_inference=50,
            num_open_loop_steps=2, task_suite_name="libero_spatial",
        )
        self.model = get_vla(self.cfg)
        self.policy_id = checkpoint_id(Path(checkpoint))
        if unnorm_key not in self.model.norm_stats:
            raise ValueError(f"Checkpoint has no normalization statistics for {unnorm_key}")
        self.processor = get_processor(self.cfg)
        self.proprio_projector = get_proprio_projector(self.cfg, self.model.llm_dim, proprio_dim=8)
        self.action_head = get_action_head(self.cfg, self.model.llm_dim)
        self._predict = get_vla_action
        for model in (self.model, self.proprio_projector, self.action_head):
            model.eval()

    def reset(self, episode_id):
        # The OFT helper used here is stateless; action queues belong to the rollout client.
        self.episode_id = episode_id

    def predict(self, obs):
        import torch

        observation = {"full_image": obs["global_rgb"], "wrist_image": obs["wrist_rgb"], "state": obs["proprio"]}
        with torch.inference_mode():
            actions = self._predict(self.cfg, self.model, self.processor, observation, obs["instruction"],
                                    action_head=self.action_head, proprio_projector=self.proprio_projector,
                                    use_film=False)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (8, 7):
            raise ValueError(f"OFT returned {actions.shape}, expected one [8,7] chunk")
        return oft_to_libero_actions(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    repo = configure_repo(args.repo)
    sys.argv = [str(repo / "vla-scripts" / "finetune.py"), *remainder]
    verify_constants()
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from .oft_dataset import LocalOFTDataset
    import prismatic.vla.datasets as datasets
    import peft

    datasets.RLDSDataset = LocalOFTDataset
    original_get_peft_model = peft.get_peft_model

    def frozen_vision_peft(*args, **kwargs):
        model = original_get_peft_model(*args, **kwargs)
        visual = 0
        for name, parameter in model.named_parameters():
            if "vision_backbone" in name:
                parameter.requires_grad_(False)
                visual += 1
        if not visual:
            raise RuntimeError("Could not identify OFT vision_backbone to freeze")
        return model

    peft.get_peft_model = frozen_vision_peft
    os.chdir(repo)
    runpy.run_path(str(repo / "vla-scripts" / "finetune.py"), run_name="__main__")


if __name__ == "__main__":
    main()
