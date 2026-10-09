# NC-RTED Numerical Policy

`reports/nc_rted/inherited_memory_repeatability_v1.json` recorded that repeated
default BF16 inherited-memory scatter-add runs differed by as much as `13.4375`.
With PyTorch deterministic algorithms enabled, the same diagnostic repeats were
exact. This evidence is diagnostic only; it does not establish full-runtime
repeatability, throughput, teacher coverage, or formal acceptance.

`nc_rted.numerics.configure_deterministic_algorithms()` is the one reusable
process-wide policy helper. It must run before CUDA initialization. It requires
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, calls
`torch.use_deterministic_algorithms(True, warn_only=False)`, and verifies both
the enabled and non-warning state. A conflicting existing workspace setting, an
already initialized CUDA process, or an unverifiable PyTorch state fails closed.

The returned `NumericalPolicy.identity()` is a SHA-256 over the policy schema
and settings. Runtime/cache owners can bind this identity to their own records.
The helper does not change model formulas, dtype, input order, or batching.
PyTorch may select different kernels under deterministic mode, so all comparison
groups and caches must bind this policy identity.

This is not backward-compatible with processes that initialize CUDA before the
policy or intentionally use a different cuBLAS workspace setting: they fail
instead of silently claiming determinism. Deterministic-mode runtime overhead is
unmeasured and is not claimed here.
