# NC-RTED Reproduction Runbook

This runbook is the operational companion to `docs/EXPERIMENT_SPEC.md`, section 13. The specification remains the scientific contract. This document records accepted release provenance and the current execution boundary; it does not authorize a new run or claim a scientific outcome.

## Scope And Accepted Release

The accepted public baseline is GitHub `main` commit `52c39a2fbe7fe6f29d9e4232434491950143d8a3`, verified by publication snapshot `e35507ce78a9543f7baed822f0d1f1b9df513f64e0e12555a1484ba98756fba4`. It includes the accepted v39 NC-RTED code path. Public source still needs the corresponding approved artifact bundle for local models, sealed manifests, media, and run-specific evidence.

Use hash-bound accepted source roots and inputs. The source39/source40 roots are the formal-training path; the dedicated prediction source is v39 and must not be substituted with a training root. The canonical prediction implementation manifest is `artifacts/nc_rted/prediction_implementation_v39/implementation_manifest.json` with SHA-256 `06a8d1238225c6b90a4f2a7034511b3ff55a2eb7477cd2a931828ae5c061cd21`. Its inherited prediction source manifest is SHA-256 `7b36f3f377a4d14e8f33ac5850d19e7757e6b3e29fddc67b4c62ae54c215723a`.

The accepted evaluator is the v36 metric-locked path. It keeps official evaluation separate from blind prediction and does not permit labels, answers, metrics, or test outcomes to affect training, threshold selection, configuration, or seed selection.

## Fixed Experiment Contract

The model set is R0 plus A/U/S/F at seeds 17, 42, and 2026: 12 formal training runs and 13 evaluated models. R0 is the inherited checkpoint and receives no new training.

- Each formal member uses the fixed 8,000 samples and 1,000 optimizer updates with accumulation 8.
- The four training groups share architecture, initialization rules, data order, generation configuration, and the inherited ReactVAU base. A removes auxiliary distillation, U is the strength-matched control, S distills calibrated quality only, and F distills quality plus relation-time distribution.
- Complete evaluation requires UCF-Crime 251 videos, XD-Violence 800 videos, and HIVAU 3,339 questions, with the fixed generation cap of 512.
- Preserve the historical training-coverage disclosure: 97,140 / 97,158. Do not reinterpret it as new NC-RTED coverage.
- Preserve the teacher limitation: 265/6,000 auxiliary-valid rows, 12 positive F rows, and 15 rows where U and F differ. Rejections and missing auxiliary supervision are not positive evidence.

The architecture reuses the original ReactVAU Stage1 Fast, final Stage2 Slow, LoRA, and projector. It does not retrain the original two stages. The added process/evidence-token branch is not an outcome claim.

## Current Execution Boundary

Sealed preparation, long-input evidence, prediction implementation, and metric-lock code are accepted inputs for later stages. Reuse them by their accepted bindings; do not repeat preparation or passing tests merely to reconstruct a command.

Formal training remains `0/12` and full evaluation remains `0/13`. R0 blind prediction is in progress and is not a completed full evaluation. No completed or running state authorizes an unbound formal worker, changes the fixed matrix, or unlocks official metrics.

The source39 qualification is the present prerequisite for formal training. `reports/nc_rted/source39_source40_postqualification_runbook_v1.md` is the operational authority after its final qualification evidence exists. It defines the actual gate-evidence extraction, scope issuance, source39 seed-17 sequence, and source40 seed-42/2026 applicability and queue sequence. It is an execution handoff, not an authorization: a scope may be issued only after the required actual evidence, resource identity, lease, deadline, budget, and 20 GiB reserve are bound and accepted.

Final checkpoints are retained on the user-approved persistent remote data disk for later retrieval. Preserve checkpoints, blind outputs, failures, and provenance records; do not silently delete or relocate them.

## Reproduction Sequence

1. Obtain the accepted public source and approved artifact bundle. Verify only the source roots, manifests, and evidence bound by the selected accepted operator.
2. Reuse sealed preparation, catalog, teacher, coverage, long-input, and implementation evidence. Do not regenerate a sealed artifact without a concrete changed input or failure.
3. After source39 qualification completes with its required status, follow the accepted source39/source40 postqualification runbook exactly. Its resource scope and queue controls remain the authorization boundary.
4. Train only the formal member selected by an accepted scope. Retain recovery evidence and final checkpoints, and report only final checkpoints as formal outputs.
5. Generate blind outputs for each frozen completed checkpoint before reading official metrics. Preserve outputs and failures separately from metric inputs.
6. Run the v36 metric-locked evaluation and predeclared statistics only after the required full prediction denominators and frozen configurations are present.

## Completion And Reporting

`completion_report.json` must distinguish operational completion from scientific outcome.

- `complete` requires 12/12 formal training runs, 13/13 full evaluations, all fixed denominators, statistics, costs, and reproduction materials.
- `incomplete` lists every missing run, prediction, artifact, or failed requirement without converting absence into success.
- `scientific_outcome` may be `positive`, `mixed`, `negative`, or `uninformative` only after completed analysis.

The final package retains accepted source and operator hashes, frozen configuration and environment identity, final checkpoints, blind outputs, the metric implementation and statistics, coverage and failure strata, and end-to-end cost records. It must state the teacher limitation and historical coverage disclosure above.

Record operations through `scripts/run_logged.py` or `scripts/log_operation.py`. Do not place credentials, private endpoints, or secrets in documentation, source snapshots, commands, reports, or logs.
