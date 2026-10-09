# NC-RTED Queue Contract

The SQLite queue is the single durable authority for local attempts. A worker
records a launch intent before starting a detached child, then atomically
replaces it with the PID, Linux starttime, host identity, and lease token before
monitoring. A replacement controller adopts a same-host live PID with the same
starttime and never starts a second child for that attempt. A crash during the
pre-PID launch window, a PID identity mismatch, or a remote process retains its
reservation for inspection instead of launching a duplicate.

Every attempt owns a separate run directory. The child receives its job key,
lease token, input hash, and producer-completion path through its environment.
It must publish a completion record containing those bindings and the exact
declared artifact checksums; existing artifact paths alone are never promoted.

GPU locks live under the payload's declared data volume, never `/tmp`, and are
nonblocking after a claim. Heartbeats prove controller ownership. Progress uses
real checkpoint/media transactions and has a separate timeout of
`max(1800, 5 * progress_p99_seconds)`.

Supervision is fenced by atomic lease-owner transfer. Worker mutations carry the
current owner, so a displaced controller loses authority. A progress timeout
creates a suspected-stall hold for long-input verification; it alone never
authorizes termination. Only typed deadline, budget, and disk hard-limit
outcomes enter process-group teardown.

A confirmed protective stop is stored as a dedicated attempt-bound code before
any signal is sent. It is independent of mutable diagnostic text; reconciliation
and expiry keep the reservation until process identity is verified gone, then
finish that attempt as protective failure rather than accepting any output.
Leaderless adoption requires descendant PID/starttime identities persisted by
the original controller. A numeric process group by itself is never continuity
evidence after restart.

The lock file descriptor is inherited by the detached child and the controller
does not explicitly unlock it while unwinding. A live process group therefore
keeps its device reservation through controller failure. Formal launch admission
also requires accepted, structured device qualification bound to the requested
physical GPU and its volume reserve; a descriptive device-binding string is not
admission evidence. This queue currently fails executable formal jobs closed:
hardware/lease-time attestation is not yet wired to a trusted resource service,
so a payload `resource_qualification` dictionary is intentionally insufficient.

Only deadline, budget, disk, integrity, leakage, and numerical hard limits may
tear down a child process group. Other launch or transient failures follow the
fixed 5/20/60-minute retry schedule. Formal jobs remain blocked until accepted
evidence supplies concrete commands, resources, and output contracts.

Artifact contracts validate more than JSON keys: formal training accepts the
actual `nc_rted_checkpoint_v2` final manifest with a matching `state.pt` byte
count and SHA-256, requires the exact run
identity and 1,000 completed updates; predictions require the exact official ID
set plus bound model and input hashes. The atomic
`result.tmp` includes the job key, lease token, and artifact records, so a
restart can commit an already-published result exactly once. Queue workers do
not read labels or metrics.

The accepted worker contract has one foreground process-group leader. It may
use synchronous helper children while the leader remains alive, but it may not
leave background descendants after leader exit. A child that deliberately calls
`setsid()` or outlives its leader escapes this local supervision model; that
launcher needs a cgroup-based supervisor before formal execution can be enabled.
