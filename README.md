# NC-RTED / ReactVAU

正常参照校准的关系—时间证据蒸馏。唯一研究规格见 [EXPERIMENT_SPEC](docs/EXPERIMENT_SPEC.md)，模块和接口见 [代码设计](docs/NC_RTED_CODE_DESIGN.md)。

这是正在开发的实验代码候选。已有模型、教师、任务输入、增量训练、恢复和统计实现；CPU 测试与局部静态审查不能替代真实长输入和完整系统验收。正式训练与官方评测尚未启动，不提供效果或显著性结论。

复用原 ReactVAU Stage1 Fast、最终 Stage2 Slow、LoRA 和 projector。新增过程编码器与证据 token 学习正常参照监督；原权重不重新训练两阶段。正式设计为 R0 + A/U/S/F × 17/42/2026，12 次增量训练、13 个评测模型。旧训练覆盖为 97,140/97,158。

## 安装与 CPU 检查

先安装适配设备/驱动的 PyTorch，再在 Python 3.10+ 环境执行：

```bash
pip install -e '.[test]'
python -m pytest -q tests/test_nc_rted*.py
```

真实 ReactVAU 运行还需其完整依赖和本地模型；参考 `external/ReactVAU-paper/requirements.txt`。4090 上已使用的环境版本在 `configs/nc_rted/environment_4090_reference.json`，它是实测环境记录，不是对 PRO6000 的兼容保证。PRO6000 需独立验证驱动、CUDA、PyTorch 架构支持与最长输入容量。

## 代码入口

- `src/nc_rted/model.py`、`bridge.py`：过程编码、质量/位置预测和 Slow 视觉 token 接入。
- `teacher_pipeline.py`、`teacher_store.py`：来源互斥正常参照、教师校准与持久化。
- `media_observer.py`、`caption_provider.py`、`detection_provider.py`：因果观测、完整描述范围与原任务输入。
- `train_worker.py`、`training.py`、`recovery.py`：配对种子、FP32 主参数、更新和完整恢复。
- `statistics.py`：来源配对 bootstrap 与六比较 Holm 校正。
- `scripts/nc_rted_train.py --help`：绑定原权重、Fast、媒体、教师和源码的训练入口；详见 [运行契约](docs/NC_RTED_PRODUCTION_RUNTIME_CONTRACT.md)。
- `scripts/nc_rted_prepare_observations.py --help`：绑定来源和模型的可恢复训练观测提取；部分完成不会发布完整教师输入。
- `scripts/nc_rted_export_frozen_vision.py --help`：从最终 Stage2 严格导出继承视觉张量并核验来源；原权重保持不变。
- `scripts/nc_rted_build_teacher.py --help`、`scripts/nc_rted_export_fast_snapshot.py --help`：数据准备接口。
- `scripts/nc_rted_queue.py --help`：事务任务队列；初始化只登记任务，不证明正式运行已获验收。

媒体/教师/Fast 快照与原始权重必须按具体路径和哈希绑定。缺失资产显式报错，不用虚假检测或替代视频。十项工程验收全部通过且代码/配置/数据冻结后才允许正式运行。

## 运行入口与当前缺口

准备真实资产并生成运行清单后，先核对清单文件的 SHA-256：

```bash
python scripts/nc_rted_train.py --config /absolute/path/runtime.json \
  --config-sha256 MANIFEST_SHA256 --mode diagnostic --dry-run
```

`--dry-run` 仅检查文件绑定，成功状态为 `FILE_BINDINGS_PASS_SEMANTIC_NOT_RUN`，不表示真实模型或媒体流程通过。正式模式还必须提供独立的 `--admission` 与 `--admission-sha256`，且全部工程验收通过。诊断结果不能作为正式模型结果。

固定训练清单为 6,000 个检测前缀和 2,000 条原始描述。Fast 快照与观察媒体必须匹配内容哈希、帧率、帧数和尺寸；原 ReactVAU 源码也须绑定完整运行依赖。固定 RT-DETR 资产已取得并完成真实观测测试；2,413 个训练媒体的全帧时间戳已核验。本地任务进程恢复已通过专项测试和独立静态审查，整套 CPU 回归 346 项通过。全量教师覆盖、派生媒体、最长输入 GPU 验收、完整盲预测入口及正式资源准入仍需完成。

### 视觉权重与数值验收状态

真实 Slow 集成发现，最终 Stage2 内嵌视觉权重与单独下载的原始 SigLIP 不同。已精确导出全部 421 个内嵌张量；来源绑定、原模型内视觉张量核验及生产加载代码已通过独立静态审查；运行时不得重新加载原始 SigLIP 覆盖继承权重。早期原始 SigLIP 观测已保留为诊断材料并排除出正式教师。用正确权重重新测试的 12 个预定训练窗口，冷缓存、热缓存及重叠窗口的关系特征完全一致；这不构成 Slow 输出或完整系统验收。

另外，同一输入的原有 BF16 记忆合并在默认 GPU 运算下不完全可重复；确定性运算下的局部重建已一致，完整数值设置仍需统一验收。当前发布版是开发快照，不能据 CPU 测试直接启动正式训练。

已加入通过独立审查的教师计算缓存及确定性运算设置工具。教师缓存保留原算法输出，实际全量提速尚未测量；数值设置已接入训练和观测提取运行时；提取配置 v2 强制绑定最终 Stage2 视觉来源，拒绝旧原始 SigLIP 配置。完整盲预测入口仍需完成。真实短训练前缀现已直接使用 Slow 内的继承视觉塔，通过来源适配器核验、禁用路径等价、冷／热缓存概率一致性及前向／反向诊断：392 个 LoRA 张量、38 个新模块张量均有有限非零梯度，峰值约 22.0 GiB；该测试使用确定性运算并在反向前将冻结视觉模型移至 CPU，未包含优化器更新、完整恢复、最长输入或正式效果评测。另以 1024 token 生成上限核验同一检测前缀的原路径／禁用分支和冷／热缓存 greedy 输出，token 完全一致；这不是完整描述任务验收。

观测提取 v2 的完整 CPU 回归和静态审查已通过。原权重来源核验单次主机内存需求下界约 30.72 GiB；真实有界提取已完成 16 个窗口，恢复时复用这 16 个并仅新增 1 个；全量 6,000 窗口及教师覆盖仍待完成，不能从模块参数量推断完整系统资源需求。

## 数据与依赖

本仓库包含代码、配置、规格、接口说明和测试。模型、训练/测试媒体、访问凭据和机器日志不随源码发布。训练不得读取官方测试标签或指标；官方评测保留完整分母。

`external/ReactVAU-paper` 包含运行所需的固定上游源码及本地适配，来源与文件哈希见 `THIRD_PARTY.md` 和 `CODE_SNAPSHOT.json`。该目录遵循其原始 LICENSE，限非商业科研用途。
