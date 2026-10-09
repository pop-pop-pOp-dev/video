# NC-RTED Caption Provider Contract

`Stage2CaptionProvider` accepts one fixed `caption:{dataset}:{id}` sample ID.
It builds a single `(id, video)` lookup from the catalog and scans the inherited
dataset once. Each matching annotation must retain the exact original task,
type, conversations, and `_reactvau_relative_video` binding. This prevents an
instruction from being rebound to another media item.

For a requested caption, the provider calls the inherited dataset's `_get_item`
directly. It deliberately does not call `__getitem__`, whose released retry
behavior can substitute a later sample after a decoding error. The existing
Stage2 dataset mixin remains responsible for resolving
`request_index_for(annotation)` and leasing the exact cache object through
`acquire(relative_path, request_index)`. Existing image preprocessing, caption
prompt preprocessing, frame selection, and per-frame PG score alignment remain
inside the inherited `_get_item` path.

The provider accepts only the original preprocessed video tensor and its exact
aligned PG scores. It sends frozen `[frames,729,1152]` inherited vision patches
to `caption_memory_from_patches`, which invokes the original frozen projector
with `local_num_frames=1`; it does not create a replacement memory algorithm.

Original caption sampler times, original observed duration, and its original
time message are carried separately from relation-observation times. A causal
observer must expose an accepted `observe_causal_window` result with a nonempty
detector identity. The provider rejects the call before media access when that
observer is not ready, including the current state where no accepted RT-DETR
snapshot exists. It never fabricates detections.

For caption media, relation observation must have one block for every original
8-second partition endpoint, including a partial tail. The provider validates
that block count before returning `SampleMaterial`; the worker still applies
the shared causal scope checks before training.

`OriginalSamplingAuditReader` is the concrete Stage2 sampling adapter. During
one direct `_get_item(index)` it wraps the instance's existing `process_video`,
delegates with the identical video path, annotation, and data arguments, and
records only its returned frame indices, fps, and time message. It restores the
original method in `finally`, requires exactly one decode call, and reads the
aligned PG scores from that same inherited item. The returned sampler times are
therefore `frame_index / fps`, not the new detector's 2FPS timestamps. Calling
the reader separately from a provider causes a second decode/cache access; this
is an explicit measurable boundary, not an implicit cache or sampling change.

## 集成修正

原 instruction ID 允许实际使用的整数。原 dataset 将 `video` 加上 data_root，故使用 `(id,_reactvau_relative_video)` 绑定固定 instruction 的相对 `video`；加载后的绝对媒体路径另行锁定并在每次调用检查。原 task/type/conversations 必须一致。

provider 内直接通过 `OriginalSamplingAuditReader.read` 获取原 sample 和同一次 `process_video` 的采样审计，observer 接收这个 audit，不能重造原时刻/time_message。正常训练不再为了审计重复调用 `_get_item`。原帧数量和审计数量必须相同。原媒体完整时长仍须由被绑定的媒体元数据交给 observer，不能用末个低频采样帧时刻替代完整视频终点。

解码出的 CPU float32 pixels 在 vision forward 前按继承 trainer 的行为转到原 projector device/dtype；SigLIP 返回与输入相同 dtype。provider 使用 no_grad，输出必须是可由新增模块和 LoRA backward 保存的普通 frozen tensor，不能传播 inference tensor。

继承 `encode_image_video_memory_batch` 明确使用 `vision_model(...,chunk_size=32)`；provider沿用同一32帧分块，保留全部帧，不能改为长视频一次forward或随意缩减输入。

## 独立审查修复边界

原instruction的start/end/fps/reader-type及被固定的data_args解码字段必须一致；运行中改变这些字段在decode前拒绝。关系observer只接显式媒体/解码字段及原sampling审计，不接conversations、答案或任务语义字段。observer返回的features/mask/time必须冻结，并在inference_mode(False)中clone为可反向传播使用的普通张量。
