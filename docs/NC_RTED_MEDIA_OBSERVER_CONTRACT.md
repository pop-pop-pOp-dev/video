# NC-RTED Media Observer Contract

`CausalMediaObserver` is the concrete frozen relation-observation adapter for
detection prefixes and caption media. It receives only a `BoundMedia` registry,
an optional lease callback, frozen detector/SigLIP bindings, a bounded frozen
frame cache, and a decoder factory. It receives no answer, label, teacher data,
or Slow output.

Every media row binds dataset/key, leased path, SHA-256, FPS, frame count,
height, and width. Before decode, the observer verifies the SHA-256 of the file
returned by the lease callback and checks decoder metadata against the bound
row. A Stage2 cache can be passed as
`lambda media: cache.acquire(media.media_key, media.request_index)`. The
callback owns the original resolver lease. The observer additionally pins and
hashes the returned inode before decoding; pathname replacement cannot switch
the decoded bytes after verification. It never invokes a splitter.

`lease_verified_media(media)` is the reusable default for an already immutable
bound file. It takes a shared lock, hashes the open file descriptor, passes
`/proc/self/fd/<fd>` to the reader so decode uses that same inode, and checks
the inode/size/time signature before release. It can be shared by the original
caption decode and the relation observer. `direct_file_lease` remains an alias
for early adapters.

PTS are constant-frame-rate `index / fps`. The legal full-media end is
`frame_count / fps`; it is metadata-derived and is never replaced with the last
caption sampler timestamp. In an interval `(start, end]`, absolute global 2FPS
ticks are `floor(start*2)+1` through `floor(end*2)`. Each tick selects the
floor-indexed frame at or before the tick. The observer supplies only those
decoded frames to `observe_causal_window`, which enforces the same causal
window. It retains RGB only for the current block and reuses frozen patches and
detections through the bounded cache.

Detection observes `(max(0, query-8), query]` and refuses a query beyond the
bound media duration. Caption uses all left-aligned 8-second endpoints and its
partial tail. A block with no qualifying half-second tick returns an explicit
empty relation mask; decoder, detector, SigLIP, cache, or feature exceptions
are technical failures and propagate instead of being treated as empty.

`ready` is false until a fully accepted RT-DETR, inherited SigLIP, cache, and
decoder factory are supplied. Calls then fail before opening a lease. This is
the current production state while an accepted RT-DETR snapshot is unavailable.

### Inherited SigLIP Binding

`InheritedSigLipAdapter` accepts only the loaded original
`llava.model.multimodal_encoder.siglip_encoder.SigLipVisionTower`. Its declared
`vision_tower_name` must resolve to the supplied local snapshot, it must be
loaded, frozen, and in eval mode, and its processor must be the original
384-pixel bicubic SigLip processor. The adapter verifies the original encoder
source hash and checks every retained `vision_model.*` safetensors tensor
against the loaded tower, after excluding exactly the deleted final encoder
layer and replaced pooling head. This prevents a caller from pairing an
arbitrary tower with a valid-looking snapshot.

The source model/config hashes and verified dtype, device, preprocessing, and
encoding implementation become part of the cache identity. Weight-file bytes
are hashed once at construction; inference uses file metadata and tensor storage
version signatures to reject snapshot or loaded-tower mutation without hashing
multi-GB weights per observation block. The adapter restores eval status on
every call and rejects any gradient-enabled tower parameters.

Each cached patch tensor owns only one frame of storage. Batch views are cloned
before persistence, preventing one frame entry from serializing the whole batch.

Before calling the detector or writing cache entries, each block verifies that
the inode signature stayed unchanged across all RGB reads. Verified earlier
blocks can remain cached if the media later changes; corrupted reads cannot
publish entries under the original identity. Zero detections produce a bypass
mask, while decoding or invalid tracking errors remain technical failures.

This adapter supports materialized full-media clips using the decord caption
reader. Explicit start/end or other-reader overrides fail before decode rather
than silently exposing frames beyond the allowed task range. The fixed 2,000
training captions all use the supported full-media form (metadata audit).

Lease integrity is also checked while unwinding decode/inference exceptions;
the original exception is retained and annotated with a media-integrity failure.
The policy conservatively rejects inode ctime changes (including unlink on
some filesystems); pinning guarantees original bytes or rejection, never a
silent switch to replacement bytes.

Cache schema v2 invalidates the older storage representation. Each payload has
a content digest; damaged entries are recorded and recomputed. Unique temporary
files, process-wide I/O locking, bounded population-lock stripes and coordinated
eviction permit multiple workers to share the cache. Disk reserve remains 20GiB.
