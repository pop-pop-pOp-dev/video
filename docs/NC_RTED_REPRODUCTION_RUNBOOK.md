# NC-RTED Reproduction Runbook

This runbook is the operational companion to `docs/EXPERIMENT_SPEC.md`, section 13. The specification remains the scientific contract. It does not authorize a new run, replace accepted operators, or claim that the formal matrix or full evaluation has completed.

## Scope and Frozen Release

The published code baseline is GitHub `main` commit `208b213a`. The accepted publication delta is v34: source33 formal-input producer `280f500`, source34 applicability consumer/builder `4bb1029`, and source34 formal-input materializer `03a3bf0`, with their reviewed tests. Publication is source provenance, not evidence that a formal run is admitted or complete.

The public v34 code does not contain the full datasets, inherited weights, sealed run-specific manifests, or all local evidence reports. A first-time reproduction therefore needs the corresponding approved artifact bundle as well as the public source. The final result package is not complete until the formal runs, blind outputs, metric artifacts, and completion report exist.

Use only accepted frozen source roots and hash-bound inputs. Do not run a workspace `HEAD`, unreviewed helper, or a reconstructed command line. The accepted operator artifacts are the executable source of truth:

- `reports/nc_rted/source33_seed17_postqualification_queue_operator_v2.json`
- `reports/nc_rted/source33_seed17_engineering_gate_evidence_map_v1.json`
- `reports/nc_rted/source33_postqualification_resource_scope_template_v1.json`
- `reports/nc_rted/source34_seed_applicability_native_review_v1.json`
- `reports/nc_rted/source34_target_formal_input_producer_native_review_v1.json`

The source33 operator requires its frozen source-root working directory, the ReactVAU runtime environment, and `PYTHONPATH` set to that source root's `src` directory before producer or queue imports. Its documented producer and controller sequence captures immutable inputs before a queue worker changes into an attempt directory. Keep that ordering.

## Fixed Experiment Contract

The model set is R0 plus A/U/S/F at seeds 17, 42, and 2026: 12 formal training runs and 13 evaluated models. R0 is the inherited checkpoint and receives no new training.

- Each formal member uses the fixed 8,000 samples and 1,000 optimizer updates, with accumulation 8.
- The four training groups share architecture, initialization rules, data order, generation configuration, and the inherited ReactVAU base. A removes the auxiliary distillation, U uses the strength-matched control, S distills calibrated quality only, and F distills quality plus relation-time distribution.
- Evaluation is complete only at UCF-Crime 251 videos, XD-Violence 800 videos, and HIVAU 3,339 questions, with the fixed generation cap of 512.
- Preserve the historical training-coverage disclosure: 97,140 / 97,158. Do not reinterpret it as new NC-RTED coverage.
- Preserve the teacher limitation: 265 auxiliary-valid rows, 12 positive F rows, and 15 rows where U and F differ. Rejections and missing auxiliary supervision are not positive evidence.

The architecture reuses frozen Fast, SigLIP, inherited projector, and inherited memory behavior. The trainable addition is the process/evidence-token branch with existing Slow LoRA. This is a method synopsis only; it makes no novelty or outcome claim.

## What Is Sealed, and What Is Still Future Work

Sealed inputs include the accepted source33/source34 producer lineage, the published v34 source snapshot, the fixed matrix, the full training catalog contract, and accepted longest-input and coverage evidence. Existing preparation artifacts and accepted evidence may be reused by hash; do not rebuild them simply to reproduce a command.

The actual source33 qualification, resource authorization, materialization, enqueue, formal training, blind prediction, metrics, and scientific interpretation remain separate steps. An accepted producer, a profile, a review, or a prepared queue operator is not a completed formal run. The source33 scope must remain absent or non-authorizing until all ten required gates, current host/GPU/interpreter/environment, lease, budget, free-space reserve, and deadline are bound by actual evidence.

In particular, inherited gates 1/2/3/5/6/8/9 may cite their accepted evidence records, but gates 4, 7, and 10 require actual qualification/update/recovery evidence. No document should turn a pending artifact into `AUTHORIZED` by copying a template.

## Reproduction Sequence

1. For a first-time environment, start from published v34 source provenance, obtain the approved artifact bundle, and validate the selected source and report hashes. For this continuing workspace, reuse accepted unchanged hashes, evidence, and frozen source roots; do not rehash all artifacts or restart a live qualification worker merely to repeat setup.
2. Reuse sealed catalog, teacher, long-input, and coverage artifacts. Confirm the bindings required by the selected profile or operator document; do not generate new substitutes or repeat accepted preparation without a concrete changed input or failure.
3. Run the accepted source33 qualification sequence only when its resource and dependency conditions are met. It produces a non-admitted profile and then actual qualification evidence. Preserve raw, fault, resume, and final qualification records.
4. After real qualification, author the resource scope from `source33_postqualification_resource_scope_template_v1.json`. Bind the exact profile and qualification hashes, current host and GPU UUID, interpreter/environment, lease, deadline, budget, reserve, and all ten gate records. This is the point at which the scope may become authorized; it is not a prefilled file.
5. Use the exact postqualification command in the accepted source33 operator v2 to materialize and enqueue seed 17. The accepted source33 producer is `scripts/nc_rted_produce_source33_formal_bundle_inputs.py`; the queue controller is `scripts/nc_rted_queue.py`. The producer performs its own source and disposable-queue admission checks.
6. Build source34 seed applicability only from the accepted source33 qualification and profile using `scripts/nc_rted_build_source34_seed_applicability.py`. It is a conservative projection for seeds 42 and 2026, not a target timing measurement.
7. Materialize accepted target inputs only through `scripts/nc_rted_produce_source34_formal_bundle_inputs.py` after target scope authorization. Use the source34 queue consumer rather than manually invoking a formal runner.
8. Retain two rolling recovery states and the final checkpoint during training; report only final checkpoints as formal outputs. Retain final checkpoints on the approved persistent remote data disk for later retrieval rather than deleting or silently relocating them.
9. For each frozen completed checkpoint, generate blind outputs before reading official metrics. Keep prediction outputs and failure records separate from metric inputs. Do not use official test labels or answers, metrics, or test outcomes for training, threshold selection, configuration changes, or seed selection.
10. Unlock metric computation only when all frozen configurations are fixed and the required full prediction denominators are complete. Then run the predeclared statistics and report costs, coverage, failures, and all seed-level results.

## Completion and Reporting

`completion_report.json` must distinguish operational completion from scientific outcome.

- `complete` requires 12/12 formal training runs, 13/13 full evaluations, all fixed denominators, statistics, costs, and reproduction materials.
- `incomplete` must list every missing run, prediction, artifact, or failed requirement without converting absence into success.
- `scientific_outcome` may be `positive`, `mixed`, `negative`, or `uninformative` only after the completed analysis. It is independent of the operational completion field.

The final reproduction package retains the source snapshot, accepted operator/report hashes, frozen configuration and environment identity, final checkpoints, blind outputs, metric implementation, statistical outputs, coverage/failure strata, and end-to-end cost record. It must also state the teacher coverage limitation and the historical coverage disclosure above.

## Environment and Records

Before a first-time stage, confirm the selected source root, required artifact bindings, writable output location, free-space reserve, and resource identity. In a continuing workspace, use the accepted handoff and verify only the bound inputs or worker state relevant to the next action. Record each operation through `scripts/run_logged.py` or `scripts/log_operation.py`. Do not place credentials, private connection endpoints, or secrets in this runbook, reports, commands, source snapshots, or logs.
