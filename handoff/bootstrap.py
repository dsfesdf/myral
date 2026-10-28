from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from .common import atomic_json, file_sha256

LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OFT_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
DATA_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
DATA_REPOSITORY = "yifengzhu-hf/LIBERO-datasets"


def configure_libero(repo=None, config_dir=None):
    """Avoid LIBERO's first-import prompt and keep configuration project-local."""
    project = Path(__file__).resolve().parents[1]
    local_repo = Path(repo).resolve() if repo else project / "third_party" / "LIBERO"
    if (local_repo / "libero" / "libero" / "__init__.py").exists() and str(local_repo) not in sys.path:
        # LIBERO's legacy setup.py omits its namespace parent with modern editable installs.
        sys.path.insert(0, str(local_repo))
    if repo is None:
        spec = importlib.util.find_spec("libero")
        if spec is None:
            raise RuntimeError("LIBERO is missing; install with scripts/install.sh sim")
        matches = [Path(p) / "libero" for p in spec.submodule_search_locations or [] if (Path(p) / "libero" / "assets").exists()]
        if not matches:
            raise RuntimeError("Cannot resolve LIBERO assets; use init-libero --repo /path/to/LIBERO")
        benchmark_root = matches[0].resolve()
    else:
        benchmark_root = Path(repo).resolve() / "libero" / "libero"
    config_dir = Path(config_dir or os.environ.get("LIBERO_CONFIG_PATH", project / "runs" / "libero-config")).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.yaml"
    if not path.exists():
        data_path = project / "data" / "raw"
        data_path.mkdir(parents=True, exist_ok=True)
        config = {"benchmark_root": str(benchmark_root), "bddl_files": str(benchmark_root / "bddl_files"),
                  "init_states": str(benchmark_root / "init_files"), "assets": str(benchmark_root / "assets"), "datasets": str(data_path)}
        atomic_json(path, config)  # JSON is valid YAML, without a bootstrap PyYAML import.
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return {"LIBERO_CONFIG_PATH": str(config_dir), "config_file": str(path)}


def download_task(cfg, output, revision=DATA_REVISION):
    from huggingface_hub import hf_hub_download

    filename = f"{cfg['suite']}/{cfg['task']}_demo.hdf5"
    path = Path(hf_hub_download(repo_id=DATA_REPOSITORY, repo_type="dataset", filename=filename,
                              revision=revision, local_dir=output))
    result = {"repo_id": DATA_REPOSITORY, "revision": revision, "filename": filename,
              "local_path": str(path.resolve()), "sha256": file_sha256(path)}
    atomic_json(path.with_suffix(".download.json"), result)
    return result
