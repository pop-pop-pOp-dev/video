# NC-RTED / ReactVAU

当前唯一研究与实验主线是 [NC-RTED：正常参照校准的关系—时间证据蒸馏](docs/EXPERIMENT_SPEC.md)。该文件按用户本次附件原样写入，替代全部旧研究方案；本次变更完成文档采用与入口清理，尚不代表 NC-RTED 已实现、验收或开始正式训练。

研究复用现有 ReactVAU Stage1 Fast、最终 Stage2 Slow、LoRA 与 projector，正式矩阵为 R0 + A/U/S/F 三种子，共 12 次增量训练、13 个评测模型。绝对截止、资源上限、训练与评测协议以实验规格为准。此前 97,140/97,158 的训练覆盖说明和原复现未完成项继续保留。

- [实验规格](docs/EXPERIMENT_SPEC.md)：唯一活动方案。
- [复现实验说明](docs/NC_RTED_REPRODUCTION_RUNBOOK.md)：已验收入口、完整实验矩阵与产物交付要求。
- [执行规则](AGENTS.md)：操作、数据权限、审查与资源边界。
- [项目状态](PROJECT_STATE.md)与[ReactVAU 状态](docs/REACTVAU_STATE.md)：读取文末最新追加记录；较早的“当前/最新”标题只代表历史时点。
- [旧方案归档](archive/research_plans/superseded_by_nc_rted_20261009/README.md)：旧文件已移出活动目录，原路径、归档位置和 SHA-256 见[清单](archive/research_plans/superseded_by_nc_rted_20261009/manifest.json)。
- [文献综述](docs/LITERATURE_2025_2026_CROSSDOMAIN_ROOT.md)：保留已核查文献依据；其中的旧候选路线排序不构成当前执行指令。

现有模型和数据位于 `models/reactvau/`、`artifacts/reactvau/`、`data/reactvau/`；历史报告、代码、预测、权重和失败证据继续保留。新增实现及其冻结产物的位置按实验规格登记，不能将推荐目录当作已完成产物。

ReactVAU 使用 `.venv-reactvau/`。下载、安装、构建及运行前加载 `configs/reactvau/download_environment.sh`，缓存与临时文件放在项目数据卷，至少保留 20 GiB 空闲。额外 H100 和存储的实际就绪状态须核验。

每项操作写入追加日志 `logs/OPERATIONS.jsonl`。命令通过 `python3 scripts/run_logged.py --name DESCRIPTION -- COMMAND ARG...` 执行；其他操作使用 `scripts/log_operation.py`。日志规则见 [docs/LOGGING.md](docs/LOGGING.md)。官方测试标签不得用于训练、教师生成、阈值调整或模型选择。
