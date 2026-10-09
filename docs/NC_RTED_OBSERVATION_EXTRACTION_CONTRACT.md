# Training observation preparation

`nc_rted_prepare_observations.py` extracts only the fixed 6,000 detection
prefixes. It binds the training provenance, all selected source allocations,
media hashes, and the full decoded relative-PTS audit before loading models.
It does not open official test identity inputs, answers or metrics. Observer
calls receive only dataset, media key and query time. Normal-reference
permission enters the teacher record after observation.
JSON inputs and PTS rows are parsed from the exact bytes whose hashes were
checked, so a path replacement cannot substitute unbound records.

The original frozen SigLIP BF16 tower and pinned RT-DETR snapshot supply the
accepted singleton/contiguous-layout observer. Exact Python source closures,
model snapshots, runtime identities and CPU thread count bind the output run.
No Slow model or trainable module is loaded. Frame caching is bounded and the
20 GiB free-space floor remains mandatory.
Constructed adapter identities must also match the detector provenance/files
and SigLIP configuration/weights verified before loading, before any window
can be committed. RT-DETR loading uses a private, hash-verified staged snapshot
and checks the actual model state, configuration and processor against it.
The inherited source closure is checked across import/model
construction, including its file identities and the constructed encoder source hash.

Each completed window is its own immutable teacher store (one compressed NPZ
or an explicit rejection). A single nonblocking writer lock protects the run.
Insertion and sealing require that validated writer context and its unchanged
run binding; direct finalization cannot assign existing windows a new identity.
Resume validates each committed window before reusing it, preserving partial
staging directories after process death. Technical failures stop preparation;
this validation also applies to reused records and final sealing. Missing run
metadata with existing output artifacts is rejected, preserving the orphaned
output for inspection instead of assigning it a new configuration.
absence of relations or reliable background is an explicit auxiliary rejection,
never an invented normal label. Every selected window remains in the denominator.

Only after all exact IDs exist does a root teacher-store index atomically seal
those existing payloads, without copying their data. The index binds the run
configuration and exact selected-ID digest. `--max-new-windows` permits bounded
preparation diagnostics, but partial sets have no root commit and cannot be
loaded as a complete teacher input. This option changes neither the fixed set
nor teacher calibration. Resume continues the same config and output identity.

This prepares frozen observations only. Crossfit teacher construction, real
support coverage, U/S/F target checks, and formal code/data freeze remain later
requirements. A complete observation store alone is not formal admission.

Local observation and inherited modules are imported from captured verified Python
source bytes, bypassing timestamp bytecode caches. Their source identities are
validated before import and again before journal creation. Preloaded modules
from either protected namespace are rejected.


Allocation admission uses one Linux POSIX process record lock on a fixed
coordination file at the mounted destination filesystem root. Cooperating model
staging, separate frame caches, and journal payload/metadata writes hold this
lock from their free-space check through allocation. Per-process/thread locks
support reentrant calls; a forked child uses fresh process state and acquires its
own record lock. Coordinator file descriptors remain open for process lifetime
because closing another descriptor for the same inode would release that
process's record lock. There are no custom fork callbacks. The coordination file
is never unlinked. The worker must have write access to that file; on hosts where
workers cannot create it at the mount root, provision it for that worker user
before preparation. Unsupported access fails instead of using a private lock. Unrelated applications on the volume are outside this
cooperative protocol. Directory and lock-file creation preserve the reserve.
The output writer lock is acquired before model construction and held through
extraction; this prevents duplicate loading for the same run. A competing process
regression with space for one payload must produce one successful allocation and
one reserve rejection, without allocating below the floor.


The main thread defers SIGINT across one allocation context and delivers it only
after releasing the process record lock and thread exclusion. Ctrl-C can therefore
wait for the current allocation/model-staging operation to finish. Nested calls
forward deferred signals to the outer handler; a forked child does not replay a
parent's pending signal. Ordinary process termination still lets the OS release
its record locks. Non-main threads cannot run Python SIGINT handlers. This is a
cooperative local preparation protocol, not a general asynchronous-exception API.
