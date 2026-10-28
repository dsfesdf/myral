# 实验仓库与产物规范

本规范适用于后续所有局部 VLA、接管成功率、空间分区、rollout、消融和真机迁移实验。目标是让任何 Codex 或研究者都能从一个 run 目录恢复“使用了什么输入、什么代码、什么权重、如何得到结果”。

## 目录边界

```text
repo/
├── AGENTS.md                 # Codex 必须遵守的工作规则
├── docs/                     # 研究协议、数据格式、运行与结果说明
├── configs/                  # 可复用、版本控制的实验配置
├── handoff/                  # 产品代码与可复用实验组件
├── tests/                    # 单元、集成和显式 opt-in 的仿真测试
├── scripts/                  # 安装、下载、迁移、批量提交脚本
├── third_party/              # 固定 commit 的外部仓库，只读
├── data/                     # 本地数据，默认不进 Git
├── runs/                     # 所有实验产物，默认不进 Git
└── transfer/                 # 需要跨机器传输的压缩包及校验和
```

源码增删只发生在 `handoff/`、`tests/`、`scripts/`、`configs/`、`docs/`。`third_party/` 只能通过 `configs/upstream-lock.json` 和 `scripts/install.sh` 更新。`data/` 与 `runs/` 的产物不直接提交；跨机器时放入 `transfer/`，同时提供 `.sha256`。

## 数据生命周期

### 原始数据

```text
data/raw/<dataset_name>/<task>_demo.hdf5
data/raw/<dataset_name>/<task>_demo.download.json
```

原始文件下载后立即记录来源仓库、revision、文件名、下载时间和 SHA256。原始数据只读，不在原地裁剪或重写。

### 处理后数据集

```text
data/processed/<dataset_name>/<task>/<dataset_id>/
├── manifest.jsonl
├── preparation.json
├── dataset_statistics.json
└── episodes/<episode_id>/
    ├── metadata.json
    ├── model.xml
    ├── scene_state.npy
    ├── states.npy
    ├── global_rgb.npy
    ├── wrist_rgb.npy
    ├── proprio.npy
    ├── actions.npy
    └── native_actions.npy
```

当前兼容的 `data/local/` 是早期目录名；后续新数据统一使用上述 `data/processed/...` 路径。`dataset_id` 取处理配置、原始数据 fingerprint、上游代码 commit 的短哈希。已有数据不移动、不覆盖；需要迁移时创建新 dataset_id。

`manifest.jsonl` 每行一个 episode，至少包含 `episode_id`、`scene_id`、`split`、`path`、`length`、`source_sha256`、`source_demo` 和 `instruction`。split 按 scene 分组，禁止按视频帧随机切分。`preparation.json` 写入数据 schema 版本、裁剪规则、相机方向、本体/动作约定和拒绝原因。

### 起点计划与标签

```text
runs/<run_id>/plans/position.jsonl
runs/<run_id>/plans/posture.jsonl
runs/<run_id>/plans/*.meta.json
runs/<run_id>/trials/trials.jsonl
runs/<run_id>/trials/states/<candidate_id>.npz
```

计划是不可变输入，包含源 episode、源 `model.xml` 与 `scene_state.npy` 的 SHA256。改变区域、采样数量、姿态或随机种子必须生成新计划文件。`trials.jsonl` 记录请求位姿、实际位姿、关节状态、初始化有效性、失败原因、策略 fingerprint、成功标签和终止原因。

无效起点使用 `valid=false, success=null`；IK 失败、碰撞、物体漂移和追踪误差不能编码成局部策略失败。有效操作才有 `success∈{0,1}`。

## 训练权重、服务和 rollout

每个策略训练是一个独立 run：

```text
runs/<run_id>/
├── manifest.json
├── config.json
├── environment.txt
├── source.json
├── checkpoints/
│   ├── step_0005000/
│   ├── selected.json
│   └── sha256.json
├── logs/
├── validation/
└── status.json
```

`checkpoints/step_<整数>/` 只由训练过程写入，完成后不可修改。`selected.json` 只能指向一个完整 checkpoint，并记录选择所用 calibration split、指标和选择命令；禁止用 `latest` 作为评测输入。checkpoint 外部存储时，在 `source.json` 写入 URI、文件大小和 SHA256。

策略服务临时文件放在：

```text
runs/<run_id>/service/ipc/
├── ready.json
├── requests/
└── responses/
```

服务必须把 `policy_id`、`server_id`、动作维度、动作块长度写入 `ready.json`。rollout 不直接写训练目录，使用独立目录：

```text
runs/<run_id>/rollouts/<split>/<episode_or_candidate>/
├── rollout.json
├── observations.npz
├── actions.npy
├── events.jsonl
└── video.mp4                 # 可选，体积大时只保留外部 URI 和 SHA256
```

`rollout.json` 至少记录策略 fingerprint、服务 fingerprint、仿真/真机环境、scene、起点、控制频率、动作块执行步数、成功、终止原因和时延统计。一次 rollout 失败后保留原目录；重跑使用新 `attempt_<整数>`，不覆盖。

## 评测、能力模型和图表

```text
runs/<run_id>/eval/<eval_id>/
├── manifest.json
├── metrics.json
├── predictions.json
├── regions.json
├── regions.csv
├── posture.json                 # 需要时
├── reliability.png/.svg
├── regions.png/.svg
└── posture.png/.svg             # 需要时
```

`eval/<eval_id>/manifest.json` 固定测试集 fingerprint、策略 checkpoint、能力模型 fingerprint、评测配置和命令。测试集一旦开始评测不得参与训练、温度校准、阈值选择或采集计划。

能力模型单独存放在训练 run：

```text
runs/<run_id>/capability/
├── model.pt
├── training.json
├── schema.json
└── calibration.json
```

它必须绑定一个冻结策略 fingerprint。策略更新后建立新 capability run，不能把旧标签直接当作新策略标签。模型比较至少包括全局常数、区域频率表和 MLP；报告必须标出无效区域、样本不足区域和 Wilson 区间。

## 命名、配置和复现

- run：`YYYYMMDD_HHMM_<task>_<purpose>_<seed>`。
- checkpoint：`step_<整数>`；不使用 `best`、`final`、`latest` 作为唯一标识。
- episode：源数据的稳定 `episode_id`；candidate：计划中的稳定 `candidate_id`；rollout：`<candidate_id>/attempt_<整数>`。
- 配置从 `configs/` 复制到 run 后再运行。运行期间修改配置必须新建 run。
- 每个 run 记录 Git commit、脏工作区状态、Python executable、`pip freeze`、GPU、上游仓库 commit、输入 fingerprint、随机种子和父 run。
- 随机数至少固定 Python、NumPy、PyTorch、CUDA 和仿真环境 seed；并把 seed 写入每条 plan/rollout 记录。

## 增删实验代码的规则

新增实验：

1. 在 `configs/` 建配置或新增明确字段；
2. 在 `handoff/` 增加可复用实现和 CLI 子命令；
3. 在 `tests/` 加最小确定性测试；
4. 在 `docs/` 说明输入、输出、失败标签和验收指标；
5. 用新 run 验证，不修改旧 run。

删除实验：先标记 deprecated，保留一个迁移说明和最后兼容版本；确认没有 manifest、脚本或 README 引用后才能删除。数据格式、动作坐标系、图像约定和成功判据的改变必须递增 schema 版本，禁止复用旧版本号。

## Codex 的默认操作顺序

```text
读取 AGENTS.md 与本规范
→ git status
→ 检查输入 manifest / checkpoint / environment
→ 新建 run_id 和 manifest
→ 运行 dry-run / check-oft / 小规模 rollout
→ 再进行完整训练或标签采集
→ 独立 eval，不回写训练数据
→ 保存 metrics、图表、命令和 status
→ 更新文档与测试
```

如果路径、schema、策略 fingerprint 或测试 split 不一致，Codex 应停止该实验并报告冲突；不能通过删除文件、覆盖旧目录或默默重采样来继续。
