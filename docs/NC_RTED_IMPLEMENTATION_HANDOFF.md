# NC-RTED 实现接续说明

唯一实验规格：`docs/EXPERIMENT_SPEC.md`。当前是实现候选，formal_execution_allowed=false。复用原 Stage1/Stage2，不开始新的两阶段训练。PRO6000镜像准备不构成显存/挂载/驱动/容量已验收。

## 模块装配

1. `TrainingCatalog.load` 核验来源划分、6000检测前缀和2000原caption指令；`TeacherIndex.load` 核验教师及全部拒绝行。
2. `load_inherited_slow` 严格加载原export四文件和完整旧LoRA/projector；`build_training_bridge` 再按seed初始化唯一新分支。
3. 检测provider需要 `StreamingDetectionReader` 的冻结Fast快照、原协议、实际因果关系observer；描述provider需要原Stage2 dataset、原tower和caption observer，内部捕获原sampling审计。
4. `TrainingWorker` 负责原任务tokenizer/CE、新分支辅助损失、FP32主参数更新、恢复及进度。正式模式需要十项验收与精确run身份；诊断输出不能改名充当formal final。
5. 检测/描述完整盲预测完成后才能解锁官方指标，统计使用预注册的来源配对bootstrap和六比较Holm。

## 接下来必须完成

- 获取并绑定固定RT-DETR-R50 snapshot；生产observer的RGB/时长来自相同已核验媒体，完整caption块含尾段。
- 实际 Fast 候选快照已导出：2,413 个媒体、567,696 个查询，覆盖全部 6,000 个检测前缀。仍需当前媒体完整哈希/CFR/PTS及原分数数值等价验收。
- 实际媒体准备：固定caption的clip/event及官方VAU派生媒体仍需原resolver生成、逐项校验。不得用整视频假替派生片段。
- 生产训练入口为 `scripts/nc_rted_train.py`，运行资产契约见 `docs/NC_RTED_PRODUCTION_RUNTIME_CONTRACT.md`；完成真实装配、完整盲预测命令、队列作业语义验收与进程恢复，得到最长真实输入和吞吐证据后冻结配置。
- PRO6000环境登记精确设备、显存、CUDA/驱动、镜像digest、挂载、20GiB空闲及租约；按实测重新排完整矩阵。

## 验证边界

原7B权重精确重载、短合成输入forward/backward和FP32优化器更新已有证据。原MemoryManager对照覆盖增强/RT开关四组合与17个连续查询，但使用合成patch和小projector。CPU接口测试、静态审查和真实GPU最长输入验收分别记录，不能互相替代。

## 仓库发布

上传目标为用户指定的 `pop-pop-pOp-dev/video`。发布采用代码/配置/文档/测试的显式清单，保留远端历史；本地历史包含运行数据，不能直接整体push。权重、视频、缓存、日志和凭据不进入发布。代码仍缺生产装配时不称“完整可复现实验已发布”；发布成功必须记录远端commit。
