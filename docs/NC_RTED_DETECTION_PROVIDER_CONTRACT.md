# 检测前缀输入契约

`StreamingDetectionReader` 读取显式 SHA256 绑定的 `nc_rted_frozen_fast/v1` snapshot。媒体行包含 dataset/media_key、media_path/media_sha256、fps/frame_count/height/width、target_fps=4、query_interval=4，以及完整的 queries(index,frame_indices,fast_score)。Fast implementation/checkpoint 身份必须与调用方提供的非空 fast_identity 相同。该文件不存 Slow 输出或标签；实际 snapshot 尚待从原缓存导出并复核。

原采样步长为 `max(1,int(fps/4))`。reader 验证实际选中前缀的 query/frame 顺序、媒体哈希和解码元数据，惰性按 query 解码，最多保留四帧 RGB，到目标后立即关闭 decoder。每 query 末帧单独调用原 SigLIP，目标的 RT 分支另按原四帧批处理，尾段仅重复已观察末帧。检测失败直接报错，不跳帧改变前缀标签对应关系。

`DetectionMemoryReplay` 直接构造原 MemoryManager，并采用原 SFTW/PEMF/Pool 参数。每步先更新原短期记忆，再在当前 Pool 插入前读取旧视觉 token、历史文本和时间提示；之后按照原 PG 分数阈值插入 Pool。原 evaluator 虽注释写 fused_score，实际执行传入 pg_score，因此回放无需 Slow 输出。检测原路径不添加 caption projector 的空间/时间位置编码，只调用原 projector.mlp。

训练固定清单包括未触发的前缀，训练 provider 对每个固定前缀构造任务目标；正式推理仍必须保留原 Fast 触发条件。二者不可混写成改变正式触发阈值。prompt style/template、RT 开关、memory enhancement、thresholds、time-message style 由继承的冻结协议显式传入，不能为不同组单独选择。

`FrozenDetectionProvider` 检查选中 query 实际末帧时刻与固定标签 endpoint 相同，仅读取这一时刻的 8 秒关系观察，提供固定 sample ID 对应的旧 token、新观察和原问题。新 token 从不回写旧记忆。失败后的部分回放必须重建；所有 reader 退出路径释放 decoder。

候选证据：原 MemoryManager 与精确原 time-message 方法，在17个连续 synthetic queries、四种 enhancement/RT组合中逐张量相等，涵盖短期窗口溢出和尾段；这不是实媒体/GPU最长输入验收。生产 Fast snapshot 与全链路 detector 绑定仍未放行。

## 媒体租约与Fast导出

reader在实际迭代时重新打开、加共享锁、对文件描述符哈希，并让OpenCV通过`/proc/self/fd/N`解码同一inode。路径被替换不能令冻结Fast分数搭配另一文件的像素；逐query读前/读后检查inode元数据，原地写入导致明确失败。关闭generator同时释放decoder及文件租约。

`nc_rted_export_fast_snapshot.py`接原PG JSON、媒体元数据及各自SHA，以及Fast checkpoint/implementation身份。元数据显式区分固定清单的media_key与原PG表的fast_cache_key；必须通过绑定的fps/frame_count证实原sample_interval与完整score数。导出不重新量化score，不运行模型或暗中补score。发布为原子create-if-absent，并检查20GiB余量。该工具不会独立证明已有PG缓存的数值精度；旧缓存若已经四舍五入，仍必须通过与正式Fast路径的等价性检查，不能据导出成功宣布满足1e-6误差门槛。
