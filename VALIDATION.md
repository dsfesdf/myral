# 实现与本机验证记录

日期：2026-10-28。项目目录：`/home/robot/test`。

## 已安装

| 环境 | 用途 | 核心版本 |
|---|---|---|
| `handoff-sim` | LIBERO 仿真、示范处理、区域评测、能力网络与绘图 | Python 3.10.22、MuJoCo 2.3.7、robosuite 1.4.1、CPU PyTorch 2.2.0 |
| `handoff-oft` | OFT 数据处理、微调和策略服务 | Python 3.10.22、CUDA PyTorch 2.2.0、FlashAttention 2.5.5、修改版 Transformers 4.40.1、TensorFlow 2.15.0 |

两个环境均已 editable 安装本项目，`pip check` 均通过。完整依赖快照位于 `runs/validation/sim-environment.txt` 与 `runs/validation/oft-environment.txt`。第三方仓库保持原始源码，固定版本见 `configs/upstream-lock.json`。

当前机器 GPU 为 **RTX 3060 12GB**。本机 EGL 设备显示初始化失败，使用已有桌面会话的 GLX 已成功渲染。没有改动系统驱动。

## 已完成的真实数据检查

- 从固定 Hugging Face revision 下载目标任务，原始 HDF5 包含 **50 条示范**，约 486MB。路径为 `data/raw/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5`。
- 原始文件 SHA256：`ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`。
- 按每条示范的模型 XML 与状态回放，生成 `data/local/`：**49 条局部示范、3,558 帧**。`demo_44` 未通过实际环境成功判据，拒绝原因保存在 `data/local/preparation.json`；没有把原始 HDF5 中统一写入的成功标签直接当作验证结果。
- 按场景分组后，训练/校准/测试分别为 **36 / 8 / 5 个场景**，对应 **2,598 / 612 / 348 帧**。动作和本体归一化统计仅使用训练组。
- 首十条示范的裁剪后专家动作回放全部成功；全局 RGB、腕部 RGB 和本体输入均能精确恢复。回放结果：`runs/replay10.json`。
- 实际测试了数值 IK、碰撞/漂移检查、完整状态快照与断点续跑。柜子等固定物体的模型参数与状态一起恢复，避免重置时出现视觉场景漂移。

## 已完成的软件与 OFT 接口验证

- `HANDOFF_SIM_DATA=data/local python -m pytest -q`：**20 passed**。包括三项真实渲染/物理测试；其余覆盖数据裁剪、坐标系、动作约定、统计隔离、IPC、锁、场景文件校验、能力网络训练和报告输出。完整日志：`runs/validation/tests.log`。
- `handoff check-oft --data data/local`：实际运行 OFT 分词器、图像变换、`RLDSBatchTransform` 和 collator。得到图像 `[2,12,224,224]`、本体 `[2,8]`、动作 `[2,8,7]`。处理器保存在 `data/local/processor/`；结果保存在 `runs/validation/oft-batch.json`。
- 实际导入上游 `finetune.py`，确认其引用的 `RLDSDataset` 已接到本地数据实现。训练命令的配置字段检查、有效 batch=32 和 dry run 均通过。
- CUDA FlashAttention BF16 小规模运行通过，输出形状 `[2,16,2,32]`。这验证了已安装二进制与当前 GPU 的基本兼容性。
- 已生成位置计划 `runs/position-plan.jsonl`（最多 5,760 次初始化尝试，目标 1,920 次有效执行）和姿态计划 `runs/posture-plan.jsonl`（1,728 次尝试）。它们绑定源场景状态/模型的文件哈希，并按场景隔离数据组。

## 尚未运行的研究实验

**尚未进行 7B 微调，也没有训练后的 VLA checkpoint、真实 VLA 接管标签或其区域成功率结果。** 当前 12GB GPU 无法容纳标准 BF16 7B 配置；没有自动改用量化模型或任务专项预训练 checkpoint。

能力 MLP 已在合成测试数据上完成训练、概率校准和出图验证。真实仿真测试中的临时 no-op 策略仅用于检查采集器和续跑，不是 VLA，不作为能力实验结果；其临时输出已经由测试清理。

下一步在具备足够显存的 GPU 上，按 README 先微调 `data/local/`，启动冻结策略服务，再用已生成计划采集真实标签，最后训练能力网络并输出地图。本项目中的空间初始化用于测量局部能力，不代表已经验证接近路径可达性。
