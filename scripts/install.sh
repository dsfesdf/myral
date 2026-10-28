#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
handoff_mode="${1:-sim}"
if [[ "$handoff_mode" != sim && "$handoff_mode" != oft ]]; then
    echo "Usage: bash scripts/install.sh sim|oft (in its own Python 3.10 environment)" >&2
    exit 2
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Activate a Python 3.10 environment first"'
mkdir -p third_party
python - <<'PY'
import json, pathlib, subprocess
lock = json.loads(pathlib.Path('configs/upstream-lock.json').read_text())
for name in ('LIBERO', 'openvla-oft'):
    path = pathlib.Path('third_party') / name
    entry = lock[name]
    created = not path.exists()
    if created:
        subprocess.run(['git', 'clone', '--depth', '1', entry['url'], str(path)], check=True)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if created and actual != entry['revision']:
        subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', entry['revision']], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', entry['revision']], check=True)
        actual = entry['revision']
    if actual != entry['revision']:
        raise SystemExit(f'{path} is at {actual}; check out locked revision {entry["revision"]} explicitly before installation')
PY
if [[ "$handoff_mode" == sim ]]; then
    python -m pip install -e '.[test]' -r configs/requirements-sim.txt
    python -m pip install 'torch==2.2.0' --index-url https://download.pytorch.org/whl/cpu
else
    python -m pip install -e '.[test]' -e third_party/openvla-oft -r configs/requirements-sim.txt -c configs/constraints-oft.txt
    python - <<'PY'
import importlib.metadata as metadata, json, pathlib, subprocess, sys
lock = json.loads(pathlib.Path('configs/upstream-lock.json').read_text())
for name in ('transformers', 'dlimp'):
    entry = lock[name]
    actual = json.loads(metadata.distribution(name).read_text('direct_url.json') or '{}')
    if actual.get('vcs_info', {}).get('commit_id') != entry['revision']:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        f'{name} @ git+{entry["url"]}@{entry["revision"]}'], check=True)
import platform, torch
if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not str(torch.version.cuda).startswith('12.'):
    raise SystemExit('Install FlashAttention 2.5.5 for this platform manually; bundled wheel targets Linux x86_64 / CUDA 12 / torch 2.2 / Python 3.10')
wheel = 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
subprocess.run([sys.executable, '-m', 'pip', 'install', wheel], check=True)
PY
fi
python -m pip install --no-deps --config-settings editable_mode=compat -e third_party/LIBERO
python -m handoff init-libero
python -m handoff doctor
echo "Installed handoff ${handoff_mode} environment. See README.md for the data and training commands."
