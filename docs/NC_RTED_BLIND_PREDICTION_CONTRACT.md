# NC-RTED Blind Prediction Contract

`nc_rted_predict.py` consumes a hash-bound identity manifest and binds the
runtime, Fast snapshot, inherited source manifest, tokenizer, decoder,
embedded-vision binding report, and implementation-source manifest by SHA-256.
The implementation manifest lists the exact prediction entrypoint and all
runtime source modules needed to execute it; the default factory rehashes that
set and the CLI requires its own path to be inside that exact checkout before
importing inherited evaluator or model code. Formal admission binds
the complete bindings, protocol, identity-manifest digest, and ordered identity
mapping. It has no input
surface for labels, answers, references, teacher rows, or metric files.  Its
formal denominator is exactly 251 UCF media, 800 XD media, and 3339 ordered
VAU instructions.  The manifest records these numbers and rejects an identity
set with any missing or duplicate row before a model is loaded.

The matrix has 13 tasks: R0 once and A/U/S/F for seeds 17, 42, and 2026. R0
is the inherited Slow model with evidence disabled. Each A/U/S/F task binds one
completed checkpoint directory by its immutable manifest and `state.pt` hashes,
plus a separate PASS final-checkpoint attestation binding those hashes to a
final 1,000-update checkpoint and its accepted run/code/config/data/teacher/
inherited-weight/runtime identities. The executed final Stage2 export tree hash
must equal the attested inherited-weight identity.
The global prediction manifest registers only the 13 task identities,
so R0 can start before future training checkpoints exist. One worker selects
one task, validates only that task's complete manifest, loads one independent
Slow instance, and writes only under that task's output root. Slow state is
never shared between task workers.

The default production factory is `nc_rted.prediction_runtime:default_factory`;
`--adapter-factory module:callable` is an explicit override for a separately
bound implementation.  The default factory preserves the bound ReactVAU
VAD query/memory/trigger/fusion/causal-smoothing route and the inherited HIVAU
4 FPS, four-frame query/decode/generation route. R0 loads the original final
Stage2 export with evidence disabled and bypasses the new detector and
observation/evidence branch. A/U/S/F each reload that same export and
then restore only the exact selected committed `state.pt["trainable"]` mapping
after checking checkpoint hash, group, seed, keys, shapes, dtypes and finite
values; prediction never accepts per-group synthetic full-export directories.
The worker passes the
official VAU question verbatim and pins Fast prompt context to `none`; it does
not reuse Stage2's training `cap64` path or add Fast scores to the question.
VAU results require full text and token IDs. VAD results require every query
with finite probabilities in [0,1], plus finite causal scores whose length
equals the original decoded frame count.
The factory builds a new Slow bridge and model-local Fast/memory route for one
selected task. It passes the final Stage2 embedded tower directly to the
derived-vision adapter and configures the deterministic numerical policy before
any inherited import that can initialize CUDA.

`preflight` and `run` fail closed unless the hash-bound formal admission report
attests the exact hash of the embedded vision binding. This is currently the
expected state until the final Stage2 embedded vision asset has been bound and
checked against the observer.

Records are atomically written and fsynced before a durable index points to
them. A task-local advisory file lease covers each complete worker run and all
record/index updates. Every read verifies the record identity, task, model
binding, manifest binding, and content digest. A record has a terminal success
or technical failure state, provenance, and digest. Only transient local I/O
failures with explicit timeout/interruption/network errnos receive automatic
retry: at most three retries, persisted with fixed 5, 20, and 60 minute
`retry_not_before` deadlines. Permission, read-only filesystem, capacity,
integrity, protocol, schema, and numerical failures are non-retryable. Success records are immutable
and a partial temporary file is never indexed as success.
