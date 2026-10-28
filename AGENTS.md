# Codex 工作区规范

本文件是 Codex 在本仓库继续开发时必须遵守的仓库级规则。更具体的模块规则放在对应子目录的 `AGENTS.md`；子目录规则可以补充本文件，但不能放宽数据、权重和结果的可追溯要求。

## 开始工作前

1. 读取本文件和 [`docs/REPOSITORY_SPEC.md`](docs/REPOSITORY_SPEC.md)。
2. 运行 `git status --short`，保留用户已有修改。
3. 查看当前实验的 `manifest.json`、`config.json`、`environment.txt` 和 `README.md`，确认输入数据、代码版本、环境和 checkpoint。
4. 不把 `data/`、`runs/`、`third_party/` 中的生成物当作源码修改；

## 修改源码

- 核心实现放在 `handoff/`；测试放在 `tests/`；可执行脚本放在 `scripts/`；实验配置放在 `configs/`。
- 研究实验的新行为先新增模块或新命令，再删除旧路径。删除、重命名或改变已有数据格式前，先搜索调用方，并提供迁移或兼容检查。
- 不修改 `third_party/` 中的上游源码。需要适配时在 `handoff/` 建桥接层，并记录上游仓库和 commit。
- 不把真实实验逻辑写进测试专用代码；合成测试只能验证软件性质，不能伪造机器人结果。
- 每次改变数据格式、实验协议、checkpoint 加载方式或 CLI 参数，都要同步更新 `docs/REPOSITORY_SPEC.md` 和 README 中对应段落。

## 运行实验

- 每个实验使用全新的 `runs/<run_id>/`；禁止覆盖已有 run。`run_id` 使用 `YYYYMMDD_HHMM_<task>_<purpose>_<seed>`，例如 `20261028_2130_spatial_vla_s42`。
- 运行目录必须包含 `manifest.json`，记录输入数据 fingerprint、代码 commit、配置 fingerprint、环境文件、上游版本、随机种子和父 run。
- 训练、rollout 和评测必须引用 checkpoint fingerprint；不能只写 `latest`、`best` 或手工文件名。
- 评测结果只写入该 run 的 `eval/`，不能把测试集结果回写训练数据、区域计划或模型选择文件。
- 失败、无效起点、超时和初始化错误必须保留原始日志；不能删除失败样本来提高成功率。

## 验证与汇报

- 代码修改至少运行相关单元测试；涉及仿真时运行 `tests/test_real_sim.py`；涉及 OFT 数据桥时运行 `handoff check-oft`。
- 报告结果时同时说明使用的 run 路径、checkpoint、数据 manifest、测试 split 和命令。
- 如果依赖、显卡、外网、模型下载或仿真渲染阻塞实验，记录在 `runs/<run_id>/status.json` 和最终说明中，不用替代模型或删除约束默默绕过。
