# 在 GPU 服务器上恢复训练项目

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
