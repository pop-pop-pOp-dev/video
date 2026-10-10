# NC-RTED Production Runtime Contract

`scripts/nc_rted_train.py` accepts only a JSON manifest with schema
`nc_rted_production_runtime/v1` and a separate SHA-256 passed through
`--config-sha256`. The manifest cannot self-attest its own digest.

The manifest binds absolute paths and SHA-256 values for the config inputs,
train annotations/provenance, an explicit ReactVAU source-file manifest,
base/tokenizer trees, exactly `config.json`, `adapter_config.json`,
`adapter_model.safetensors`, and `non_lora_trainables.bin` from the export,
approved Stage2 resolver/config, Fast snapshot, media catalog, RT-DETR/SigLIP
snapshots, and teacher store. Directory values use the stable tree digest:
sorted relative filename, NUL, file SHA-256, newline. Symlinks are rejected.
The detector object carries both `siglip_snapshot` with its SHA-256 and
`final_stage2_siglip_snapshot` with its SHA-256. The first is the original
raw SigLIP tree: its `config.json` SHA-256 is the expected raw-configuration
binding for the derived asset. The second is the final-Stage2 derived vision
tree used by the runtime. Its provenance must name the exact
`non_lora_trainables.bin` at
`inherited.export_directory/non_lora_trainables.bin`, whose SHA-256 is already
bound in `inherited.export_hashes`; it must also prove the complete retained
vision tensor map against that parent export. A derived path alone is not an
identity claim and cannot replace either the raw-config or parent-export
bindings.
The file-binding preflight requires the source manifest to cover every Python source file below
the inherited `llava/`, `eval_utils/`, and top-level `vad/` trees, including
Qwen loading, the multimodal memory manager, VAD `detect_utils`, and
`vad.get_prompt`. A placeholder source-manifest entry is insufficient. Before
the evaluator imports its early top-level `detect_utils`, the runtime exposes
only the bound `eval_utils/vad` directory. Preloaded modules named `llava`,
`eval_utils`, `vad`, or `detect_utils` must resolve under that bound
root; their loaded file must be present in the manifest with its current hash,
and package namespace paths must also remain under the bound root. Another
checkout already present in `sys.modules` is rejected. These source hashes are
checked before inherited imports and then retained as the assembly freeze;
per-sample operation checks only module path membership and does not rehash the
inherited source tree.

The `sampling` object is exact: `local_num_frames=1`, `frames_upbound=64`,
`frames_lowbound=4`, `sample_type=dynamic_fps1`, `time_msg=short_online_v2`,
`model_max_length=8192`, `vision_chunk_size=32`, and `projector=original`.
The runtime rejects drift before importing ReactVAU or model libraries.

After preflight succeeds, assembly calls
`configure_deterministic_algorithms()` before importing inherited ReactVAU
modules, accessing CUDA, or constructing models. The returned policy identity
is passed to the frozen detector and SigLIP adapter. The checkpoint store
retains its established nine-field identity; the policy implementation is bound
by `runtime_sha256`, and the final-Stage2 SigLIP snapshot/configuration is
bound by the hash-bound runtime manifest whose digest is `config_sha256`.
Preflight's derived-asset provenance check is CPU-only; it is not an
inherited-module or CUDA load.

Before the first train-path forward, assembly configures the inherited raw Slow
model with `gradient_checkpointing_enable(use_reentrant=False)` and
`config.use_cache=False`. The fixed
`nc_rted_training_memory_mode/v1` identity is retained on the live raw config;
its implementation is bound by `runtime_sha256`. Capacity evidence using a
different activation-memory mode is not production-equivalent.

The catalog binds the original full training annotations for the fixed catalog,
plus a separately hash-bound JSON containing exactly its 2,000 caption rows,
its YAML loader declaration, and the original PG-score JSON. The YAML must name
the subset and PG files. This lets the original `LazySupervisedDataset` process
only the selected caption rows while retaining the full original annotation
identity for catalog construction.

The runtime temporarily sets `REACTVAU_STAGE2_CACHE_CONFIG` to an already
hash-bound resolver config or materialized-media catalog while the original
loader performs its first-three-file check, then restores the prior process
environment. It never relies on ambient environment state to bypass that check.

Stage2 resolver mode requires `accepted_status=APPROVED_FOR_EXECUTION` in both
the manifest and hash-bound cache config. Materialized mode instead maps the
original relative media path to immutable catalog media and leases the
hash-verified file descriptor. Assembly constructs a local fail-closed Stage2
subclass without changing the imported `LazySupervisedDataset`, so caption
access always uses the original `_get_item`, original preprocessing/tokenizer/PG
scores, and the immutable Stage2 cache resolver.

For a committed teacher store, `teacher.artifact` names its directory and
`teacher.sha256` is the digest of its committed `index.json`; the store reader
then verifies every indexed chunk. A teacher pipeline JSON instead binds its
own file digest directly.

The media catalog maps detection keys and caption aliases to the same bound
media hash. It feeds `CausalMediaObserver`; labels, answers, and teacher data
are absent from this boundary. Detection uses only the supplied frozen Fast
snapshot and per-dataset protocol. Teacher records are validated during assembly before model loading. They
remain outside the public-media provider inputs.

`media.caption_observation_cache` is an explicit absolute-root, 20-GiB-reserve,
and at-most-23-GiB store for compact frozen caption relation observations. Its
identity includes the fixed caption request and original sampling audit, bound
media metadata, bounded-resolver source/range/config identity, detector and
SigLIP/numerical identities, and the exact feature/mask/time layout. It stores
only detached CPU observation features, masks, times, and audit metadata; it
does not retain video bytes, RGB frames, patch features, Slow state, old memory,
or trainable tensors. The production layout fixes cached features to BF16;
corrupt or nonmatching entries are deleted and rebuilt.
`scripts/nc_rted_prepare_caption_observations.py` invokes the ordinary caption
provider in deterministic catalog order; completed entries are reusable on a
later invocation, while all inherited decode/sampling/PG and old-memory paths
remain unchanged.

Before model construction, every selected detection task joins its
`(dataset, media_key)` to both the Fast snapshot and observer media catalog.
The media SHA-256, FPS, frame count, dimensions, selected query index, and its
final-frame endpoint must agree. This prevents frozen Fast memory and relation
evidence from being composed from different videos with the same key.

`--dry-run` validates file bindings and manifest shape only. It does not load a
model, allocate a GPU, decode media, parse the full catalog, write checkpoints,
or generate any cache. Assembly validates catalog, teacher coverage, the exact
caption subset, and formal-admission/source cross-links before model loading.

Diagnostic manifests require a `diagnostic:` run ID, isolated checkpoint root,
and positive `diagnostic_updates`. Formal runs require a non-diagnostic run ID
and `--admission` plus `--admission-sha256`; the admission is intentionally not
named inside the hash-bound runtime config. It contains `PASS`, execution
authorization, and exactly engineering checks `1` through `10` all set to
`PASS`, and can bind the final config digest in its run identity without a
self-referential two-file hash cycle. At execution,
`TrainingWorker` independently verifies that admission against the full fixed
recipe, checkpoint identity, complete task count, and source-file hashes.

The Slow loader has already applied the final-Stage2 export and loaded its
retained vision tower before the adapter is constructed. Assembly obtains that
existing tower from Slow, requires it to be loaded, moves it to the original
projector parameter dtype, freezes it, and puts it in eval mode. It must not
reload a standalone tower or rewrite the tower's recorded raw
`vision_tower_name`. `InheritedSigLipAdapter` receives the derived snapshot
only to validate the parent export, raw configuration, and every live retained
tensor against the already-loaded tower. The sample provider restores eval mode
after the incremental trainer calls `train()` on its bridge.

The inherited tokenizer is loaded from the separately bound tokenizer tree with
`local_files_only=True` and model length 8192, then configured through the
runtime's inherited-tokenizer helper. `DataArguments` retains the bound dataset
YAML and exact sampling fields; only after the reused tower is available does
assembly assign its image processor, set multimodal mode, synchronize the
inherited model configuration, and apply the inherited data-argument helper.
The original `LazySupervisedDataset` remains the implementation boundary; the
runtime installs only its fail-closed Stage2 subclass around that original
behavior.

Derived parent verification is a material CPU-memory operation. One real
parent verification reached approximately 33 GiB RSS; the static lower bound is
about 30.72 GiB from the simultaneous parent-export bytes, deserialized parent
tensor map, and derived vision bytes. This is an execution capacity constraint,
not a performance or completion claim. The bounded v24 observation attempt was
terminated before its first observation, and formal full-freeze/admission has
not passed; neither all-6,000 observation completion nor formal execution is
claimed here.
