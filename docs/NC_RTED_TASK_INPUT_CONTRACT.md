# NC-RTED 原任务与教师接入契约（候选，尚未冻结）

`task_inputs.py` 只读取被 provenance 哈希绑定的增量训练清单及原训练指令；不打开 provenance 中提及的官方测试文件。固定 6,000 个检测前缀与 2,000 条描述指令，检测按数据集 × 正常/异常各 1,500 个。所有来源必须属于增量 train 分配。

描述问题和答案从原训练指令按 `(id, video)` 精确取回；不生成新语义真值。预处理直接调用原 `preprocess_multimodal` 与 `preprocess_qwen`，保留其特殊 token 与交叉熵标签掩码。只做原 Stage2 已有的 `<video>` 到 `<image>` 替换。原视频采样产生的 time message 原样传入。检测二值目标来自固定合法区间标签，答词为 Yes/No；问题必须由原检测运行器提供；rating/CoT 提示不能无审计地接二值目标。

原视觉 token 必须来自冻结原 projector/记忆路径，并由生产运行器提供实际帧时间、查询终点和图像元信息。此接口不重建或修改 Fast 触发、记忆、融合、平滑。原检测与 Stage2 描述路径的具体采样/时间位置差异需分别保留、验收，不能用同一个自写 memory reducer 代替两条路径。

描述观测块数必须等于 `ceil(observed_seconds/8)`；部分尾块保留。时间戳必须位于对应块内，检测只允许最近8秒；原记忆时间也不得晚于查询。生产 timestamp 采用 FP32，每关系四个实际观测时间。完整长输入不能通过丢块解决容量问题。

检测 sample/window ID 为 `detection:{dataset}:{key}:{query_index}`，描述 ID 为 `caption:{dataset}:{instruction_id}`。来源与关系 ID 只用于文件审计和对齐，不能作为学生特征。

教师行 `relation_ids` 每个真实候选一个ID（最多16），按 `mask/F_positions` 中的四格排列，末尾至64格只补零/false。接入时按观测候选顺序重排；核验教师支持是有效观测的子集，不因教师拒绝而删除学生输入。拒绝教师必须显式记录原因，并以 `eligible=false` 屏蔽辅助损失；合法空证据的 `quality=0, eligible=true` 与拒绝不同。S/F逐样本a相同；U复用相同位置分布，只替换质量。

待验收：生产媒体运行器、真实 prompt/缓存等价、最长分层样本、全教师覆盖、最终源代码独立复核。此契约及单元检查不能作为正式训练放行证据。

`inherited_memory.py` 直接调用原描述 projector forward 与原检测 MemoryManager.get_memory_tokens + projector.mlp；保持两者已有的位置编码差异。冻结图像输出保持原dtype，接训练前转为普通detached tensor，避免 inference tensor 不能参与 LoRA backward。PG 分数不允许缺省为零或漏对齐。

`train_worker.py` 连接固定catalog、teacher index、provider、原tokenizer、bridge、trainer与checkpoint。教师索引必须覆盖全部6,000窗口（包括拒绝），先检查U/F数据集内精确直方图和S/F逐样本a。provider只接样本ID，不接答案/教师。正式执行入口核验完整recipe、十项验收及源文件哈希；资源/租约核验仍由队列外层完成。诊断run_id必须以 `diagnostic:` 开头，禁止把其checkpoint直接作为正式起点。当前尚未提供完成验收的生产provider或formal admission，不能宣称正式训练已经可运行。
