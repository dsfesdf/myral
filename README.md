# LIBERO 局部 VLA 与接管能力验证

单任务实现：Panda 接近黑碗、抓取并放到盘子上。复用 LIBERO 公开示范，微调 OpenVLA-OFT；冻结策略后，测量目标附近 24 个区域的成功率，训练三成员接管能力 MLP，再进行位置配对的高度/yaw 实验。

这是实验代码。`tests/` 中的合成样本只验证软件，不代表 VLA 或仿真任务成功率。实际执行、安装情况见 [VALIDATION.md](VALIDATION.md)。

后续实验的目录、命名、数据生命周期、权重/rollout/评测保存规则见 [docs/REPOSITORY_SPEC.md](docs/REPOSITORY_SPEC.md)；Codex 自动遵守的工作区规则见 [AGENTS.md](AGENTS.md)。

从当前单任务基线扩展到标准数据目录、主动采集、多任务和 UR5 的阶段路线见 [docs/EXTENSION_ROADMAP.md](docs/EXTENSION_ROADMAP.md)。

## 安装

使用两个 Python 3.10 环境，避免仿真与 OFT 的 TensorFlow / Transformers 依赖互相影响：

```bash
conda create -y -n handoff-sim python=3.10 pip
conda activate handoff-sim
bash scripts/install.sh sim

conda create -y -n handoff-oft python=3.10 pip
conda activate handoff-oft
bash scripts/install.sh oft
```

`sim` 环境安装 CPU PyTorch，足以训练小型能力网络。`oft` 环境安装 CUDA PyTorch 2.2.0、修改版 Transformers、TensorFlow 与 OFT，以及 Linux x86_64 / CUDA 12 的 FlashAttention 2.5.5 预编译 wheel；其他平台需自行匹配。复现版本记录在 `configs/upstream-lock.json`，安装后的完整依赖可保存为 `python -m pip freeze > runs/environment.txt`。

项目安装后可以使用 `handoff` 或 `python -m handoff`。在仓库根目录运行下面命令。LIBERO 的初始配置自动写入 `runs/libero-config/`，不会弹出路径确认提示。已有 `LIBERO_CONFIG_PATH` 时尊重该配置。

渲染默认在有桌面 `DISPLAY` 时使用 `MUJOCO_GL=glx`，无桌面时使用 `egl`。当前机器已实测 GLX；EGL 设备初始化失败时可在有桌面会话中显式设置 `export MUJOCO_GL=glx`。

## 1. 获取数据并检查接口

```bash
conda activate handoff-sim
handoff doctor
handoff init-libero
handoff inspect --output runs/inspection
handoff download --output data/raw
```

下载器只获取所选任务的 HDF5，固定数据集 revision，记录 SHA256。默认文件为：

```text
data/raw/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```

该文件包含 50 条公开示范；检查器保存全局、腕部两张 RGB 和物体几何信息。任务名、目标物体、控制预算、空间范围均在 [configs/pilot.json](configs/pilot.json) 中。

先用十条数据检查，再处理全量：

```bash
handoff prepare \
  --raw data/raw/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
  --output data/check10 --limit 10
handoff replay --data data/check10 --count 10 --output runs/replay10.json

handoff prepare \
  --raw data/raw/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
  --output data/local
```

准备器恢复每条示范的原始模型 XML 和 MuJoCo 状态，找到首次距离碗中心不超过 12cm、夹爪张开且未接触目标的位置，往前保留五步，截取到原始任务结束。固定柜子/炉灶的位置属于模型参数，只有状态向量无法恢复它们；每条示范及每次试验快照均保留 XML，重置时禁止重新随机化固定物体。没有满足条件或状态回放未达到环境成功判据的示范会被拒绝并记录原因。

全局与腕部图像都由同一个仿真接口重新渲染，并按 OFT 的 LIBERO 约定旋转 180°，避免依赖原始 HDF5 图像的方向。八维本体输入为 `xyz + axis-angle + 两个夹爪关节位置`。夹爪动作由 LIBERO 的 `-1=开、+1=闭` 转为 OFT 的 `1=开、0=闭`；推理时逆转换。

每条轨迹输出为可 mmap 的 NPY 数组及 JSON 元数据。按初始物体布局分组，哈希划分约 70% 训练、15% 校准、15% 测试；统计量只从训练组计算。十条检查数据可能缺少某个分组，正式训练须使用全量并检查各组场景数。

`replay` 检查首帧图像、本体状态和动作转换，并实际执行保存的专家动作。失败会返回非零退出码，先修复回放再训练。

## 2. 微调并冻结 OpenVLA-OFT

```bash
conda activate handoff-oft
handoff check-oft --data data/local
handoff train-oft --data data/local --output runs/oft --dry-run
handoff train-oft --data data/local --output runs/oft --devices 1 --micro-batch 4 --steps 5000
```

从 `openvla/openvla-7b` 底座开始，LoRA rank 32，冻结视觉编码器，训练语言侧适配器、动作头与本体投影。有效 batch 固定 32，学习率 1e-4；先训练 5k 步，校准组用于验证与是否延长至 10k 的选择。测试组示范不进入训练。

启动器检查上游配置字段及 `7 维动作 / 8 维本体 / 8 步动作块`。通过运行时替换上游 `RLDSDataset`，将本地 NPY 数据传给 **原版 `RLDSBatchTransform`**，继续复用 OFT 的分词、prompt、图像处理、collator 和损失。没有修改第三方训练源码。为保持训练/推理一致，首版禁用图像增强及推理中心裁剪。

`check-oft` 只下载处理器/分词器，使用实际 OFT 变换和 collator 检查一批本地数据，不加载 7B 权重。`micro-batch` 至少为 2，以避免上游 collator 在 batch=1 时压掉本体状态的批次维度。TensorFlow 只做预处理，启动器禁止其占用 GPU 显存。

训练参数由设备数、micro batch 自动换算梯度累积步数；checkpoint 由上游保存（含合并权重、动作头、本体投影与 `dataset_statistics.json`）。选择完整的已合并 checkpoint 启动服务：

```bash
handoff serve-oft --checkpoint /path/to/merged_checkpoint --ipc runs/ipc
```

服务使用上游 `get_vla_action` 输出八步动作。仿真端每次只执行两步就重新观测。两端通过原子 NPZ 文件交换观测和动作，可以在不同环境或同一共享文件系统的不同机器上运行。服务启动时计算 checkpoint 内容指纹；中途更换策略或服务会中止当前试验。

本机 12GB 显存不足以承诺运行标准 BF16 7B 配置。正式 VLA 微调和推理应在文档中计划的 48/80GB GPU 上进行；不自动改成量化模型或换用已在目标任务训练过的 checkpoint。

## 3. 位置分区与接管标签采集

另开仿真环境的终端，保持推理服务运行：

```bash
conda activate handoff-sim
handoff plan-starts --data data/local --output runs/position-plan.jsonl
handoff collect --plan runs/position-plan.jsonl --output runs/position --ipc runs/ipc --max-trials 10
# 检查前十次初始化记录，再去掉 --max-trials 继续；同一命令支持续跑。
handoff collect --plan runs/position-plan.jsonl --output runs/position --ipc runs/ipc
```

目标坐标系采用碗的 yaw 与世界竖直方向。八个 45° 扇区 × 三个半径层（4–8、8–12、12–16cm）构成 24 区域；高度为碗口上方 12cm，夹爪朝下且张开。碗口高度由对象 `top_offset` 读取，无法读取时明确报错，也可在配置中指定 `target_rim_offset`。

区域内按面积均匀采样。数值 IK 使用实际 MuJoCo 雅可比，优先接近统一参考关节构型的解；检查整个机器人碰撞、初始化误差、夹爪状态和场景物体漂移。初始化后五步保持，再将初始速度置零。

**这里直接设置关节状态，测的是有效起点下的局部能力；没有验证从准备位置到该起点的无碰撞路径。** IK/碰撞失败不能解释为规划不可达的证明。

每区域目标为训练 40、校准 10、测试 30 次有效试验。位置采集最多预排三倍尝试，到达各自有效配额后停止；不足额如实保留。有效起点执行最多 300 控制步，环境成功则提前结束。无效起点的 `success=null`，不混入局部失败。

每次记录完整初始与源场景状态、请求/实际位姿、关节构型、区域、策略指纹、初始化错误、操作标签和终止原因。成功率模型的特征在执行动作**之前**提取。

## 4. 训练成功率网络与出图

```bash
handoff train-capability --trials runs/position/trials.jsonl --output runs/capability
handoff report --trials runs/position/trials.jsonl \
  --model runs/capability/model.pt --output runs/position-report
```

首版为仿真状态 MLP：末端相对目标位姿（位置 + 6D 旋转）、七个关节角、夹爪宽度、盘子相对位姿、物体尺寸与六个带掩码的场景对象槽，共 110 维。碰撞使用真实几何；网络的尺寸特征使用 MuJoCo 几何包围球得到的保守包围盒。网络不读取后续轨迹、成功标志或测试集统计。

三个独立初始化的 `256 → 256 → 1` MLP 使用 BCE、AdamW；校准组早停并拟合共享温度。输出三个成员概率的均值和标准差。更新 VLA 后必须重新采集当前策略标签，旧能力模型不能直接复用。

报告默认只读取封存测试组的评测指标，并比较训练组拟合的全局常数概率、区域频率表与 MLP。输出：

- `regions.csv/json`：各区域尝试数、有效比例、去重后的样本量、真实成功率与 95% Wilson 区间。
- `metrics.json`、`predictions.json`：Brier、log loss、ECE、逐起点预测及模型间差异。
- `regions.png/svg`：真实/预测/误差空间地图；灰色表示没有有效观测，斜线表示样本量不足。
- `reliability.png/svg`：预测概率与真实频率。

日志拒绝场景/初始状态/配对组跨分组、不同策略版本混合和无效起点带失败标签；相同物理起点的确定性重复会去重。Wilson 区间描述采样起点，不作为跨场景或跨训练种子的显著性结论。

## 5. 高度与姿态配对实验

```bash
handoff plan-starts --data data/local --phase posture --output runs/posture-plan.jsonl
handoff collect --plan runs/posture-plan.jsonl --output runs/posture --ipc runs/ipc
handoff report --trials runs/posture/trials.jsonl \
  --model runs/capability/model.pt --output runs/posture-report
```

固定同一场景与同一 x/y，交叉测试碗口上方 8/12/16cm 和 yaw −15°/0°/+15°，保存实际关节解。首轮每区域训练/校准/测试分别取 3/2/3 个配对位置。额外输出 `posture.json` 与 `posture.png/svg`；姿态差异只使用九个条件全部有效的完整配对组，并报告缺失配对数量。

## 代码入口与测试

`handoff/sim.py`：真实仿真与 IK；`data.py` / `oft_dataset.py`：裁剪、归一化与 OFT 数据桥；`oft.py` / `ipc.py`：训练与策略服务；`trials.py`：不可变采样计划及断点续跑；`capability.py` / `report.py`：能力训练和统计图表。

```bash
conda activate handoff-sim
python -m pytest -q
HANDOFF_SIM_DATA=data/local python -m pytest -q tests/test_real_sim.py
```

核心测试覆盖裁剪条件、夹爪方向、未来动作块、本体状态维度、训练统计隔离、目标坐标不变性、区域采样、跨组泄漏、重复起点、动作重规划频率、IPC、网络训练和报告生成。真实环境、原始示范回放与 OFT 大模型需分别验证，测试通过不等于研究假设成立。
