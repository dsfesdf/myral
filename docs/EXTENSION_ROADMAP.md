# 后续扩展路线

扩展按“先保持可复现，再增加能力”的顺序推进。每一阶段完成验收后，才允许进入下一阶段；所有阶段都使用新的 run 和新的 schema 版本。

## Phase 0：固定当前基线

目标：得到可复现的单任务局部 VLA 基线。

- 使用 `data/local/` 作为兼容输入，完成 OpenVLA-OFT 微调。
- 选择一个合并后的 checkpoint，建立 `runs/<run_id>/checkpoints/` 和 `selected.json`。
- 运行不少于一个 calibration、一个 sealed test 和一次冻结策略 rollout。
- 通过 `handoff check-oft`、仿真回放和完整评测；保存环境、输入和权重 fingerprint。

验收：策略版本可加载；同一 candidate 重跑时输入、动作接口和成功判定一致；测试集不参与选择。

## Phase 1：把现有路径迁移到标准数据目录

目标：消除 `data/local/` 与散落的 `runs/*.jsonl`。

- 新增 `data/processed/libero_spatial/<task>/<dataset_id>/` 输出路径。
- 保留 `data/local/` 只读兼容，不复制修改其内容。
- `prepare` 增加 `--dataset-id` 或自动 fingerprint；输出新 manifest、schema 版本和源文件校验。
- `plan-starts`、`collect`、`report` 全部以 `runs/<run_id>/plans|trials|eval` 为输出。

验收：从原始 HDF5、固定 commit 和配置可以生成同一个 dataset fingerprint；旧数据仍可被读取，但新实验不再向 `data/local/` 写入。

## Phase 2：策略训练与 rollout 生命周期

目标：将训练、服务和 rollout 绑定到同一个策略身份。

- 增加 run 初始化命令，生成 `manifest.json`、`config.json`、`source.json`、`status.json`。
- 训练结束自动计算每个 checkpoint 的 SHA256，并写入 `checkpoints/sha256.json`。
- 增加 `select-checkpoint`，只允许依据 calibration 指标写入 `selected.json`。
- `serve-oft` 从 `selected.json` 加载，禁止传入无 manifest 的裸目录。
- rollout 结果移到 `rollouts/<split>/<candidate>/attempt_<n>/`，视频和大文件写 external URI。

验收：删除或替换任一 checkpoint、服务或输入数据后，评测 manifest 能检测 fingerprint 不一致。

## Phase 3：能力模型和空间实验扩展

目标：从单个 MLP 扩展到可比较的能力估计实验。

- 增加区域表、全局常数、MLP、集成模型和校准模型的统一接口。
- 对位置、末端高度、yaw、关节构型和邻近障碍分别建立 plan 类型，不能把不同条件混成一个 region ID。
- 增加 bootstrap/分层置信区间，并区分起点采样误差、场景差异和策略随机性。
- 增加能力模型版本；策略 checkpoint 改变时强制新标签或明确的版本迁移。

验收：报告能复现区域成功率图、可靠性图、姿态配对结果和失败原因表；无效起点不进入操作成功率分母。

## Phase 4：主动采集闭环

目标：比较随机、均匀、边界和面向完整任务收益的采集方法。

- 新增 `handoff propose-collection` 和 `handoff shadow-update` 命令。
- 每轮生成独立 `plans/round_<n>/`，记录试采、影子策略、训练步数和候选分配。
- 影子策略从同一个父 checkpoint 开始；每个提案记录试探成本、GPU 时间和自动执行次数。
- 采集后的能力模型必须允许重新选择交接目标，然后在未见开发场景上估计整体增益。
- 最终 test 只在所有采集决策完成后运行一次。

验收：能比较单位人工时间的完整任务成功率曲线，并同时报告试采开销、机器人执行时间和 GPU 时间。

## Phase 5：多任务与视觉能力模型

目标：验证方法是否超出单任务仿真状态 MLP。

- 每个任务独立保存 dataset、policy、capability 和 eval；共享底座时在 manifest 中记录父 checkpoint。
- 增加视觉能力模型作为独立对照，输入只能使用交接前可获得的信息。
- 不把不同任务的语言泛化、相机布局泛化和策略迁移混入同一个结果；每种泛化都建立独立 split 和 eval ID。

验收：报告任务内、未见布局和未见目标条件的结果，并保留单任务结果作为可诊断基线。

## Phase 6：UR5 迁移与真机

目标：把已验证的实验协议迁移到 UR5，而不是直接声称 sim-to-real。

- 新增 `handoff/robots/ur5/`，不修改 Panda 适配器。
- 记录 UR5 型号、夹爪、控制器、TCP 标定、相机标定、动作频率和安全限位。
- 真机示范放在独立 dataset namespace，例如 `data/processed/ur5/<task>/<dataset_id>/`。
- 真机 rollout 使用 `runs/<run_id>/rollouts/real/`，保存原始视频、控制日志、复位次数和人工干预事件。
- 先做冻结策略交接实验，再做主动采集；真机与仿真不得合并成同一个成功率分母。

验收：独立相机或治具成功判定可复现；规划失败、接管失败和人工急停有不同 termination；报告真实人工分钟数和自动执行成本。

## 每阶段禁止的捷径

- 不覆盖已有 run、checkpoint、plan 或评测目录。
- 不把失败/无效样本删除后重新统计。
- 不让测试结果参与 checkpoint、温度、阈值或采集提案选择。
- 不把模型大小、数据来源、规划预算和主动采集规则同时改变后解释为单一因素收益。
- 不把仿真专家查询次数直接称为人工示范分钟数；两种成本分开记录。
