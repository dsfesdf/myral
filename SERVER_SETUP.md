# 在 GPU 服务器上恢复训练项目

## 完整项目快照

`transfer/full-project.tar.gz` 包含本机项目的代码、原始及处理后数据、
运行日志、第三方代码和缓存。仅排除 Git 内部元数据与 `transfer/`
目录（避免重复打包已有传输包）。项目内所有 `.gitignore` 已按要求删除。
全量快照是打包时刻的文件副本，不包含后来新增的文件。

克隆并执行 `git lfs pull` 后，可在仓库根目录恢复：

```bash
sha256sum -c transfer/full-project.tar.gz.sha256
tar -xzf transfer/full-project.tar.gz
```

此操作会恢复快照中的同名文件，建议在刚克隆的目录中执行。
Python 环境、系统驱动及项目目录外的模型缓存仍需在服务器另行准备。

代码依赖的上游仓库由 `scripts/install.sh` 按
`configs/upstream-lock.json` 中的版本自动下载，无需复制本机 Python 环境。

## 数据

训练数据包为 `transfer/local-data.tar.gz`，对应校验文件为
`transfer/local-data.tar.gz.sha256`。数据包通过 Git LFS 保存。
服务器需要先安装 Git LFS（Ubuntu 可使用 `sudo apt-get install git-lfs`），再运行：

```bash
git lfs install
git clone https://github.com/dsfesdf/myral.git
cd myral
git lfs pull
```

取得完整数据包后，在仓库根目录执行：

```bash
sha256sum -c transfer/local-data.tar.gz.sha256
mkdir -p data
tar -xzf transfer/local-data.tar.gz -C data
```

训练读取 `data/local/` 内的图像、本体状态、动作及统计文件，无需重新
渲染原始示范。基础模型权重不在此数据包中，需要服务器能够下载或另行提供。

## 环境与训练

```bash
conda create -y -n handoff-oft python=3.10 pip
conda activate handoff-oft
bash scripts/install.sh oft
handoff check-oft --data data/local
handoff train-oft --data data/local --output runs/oft --dry-run
handoff train-oft --data data/local --output runs/oft-smoke --devices 1 --micro-batch 2 --steps 20
```

确认短程训练、checkpoint 保存和显存正常后，再执行正式训练：

```bash
handoff train-oft --data data/local --output runs/oft --devices 1 --micro-batch 2 --steps 5000
```

本项目要求 micro batch 至少为 2；有效 batch 固定为 32，由启动器计算
梯度累积。48GB GPU 的实际显存占用仍需通过短程运行确认。
