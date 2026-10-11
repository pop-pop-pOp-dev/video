# NC-RTED / ReactVAU

Normally referenced relation-time evidence distillation. The scientific contract is [EXPERIMENT_SPEC](docs/EXPERIMENT_SPEC.md); implementation boundaries are described by the accepted source and artifact bindings, not by a workspace `HEAD`.

The accepted public baseline is GitHub `main` commit `52c39a2fbe7fe6f29d9e4232434491950143d8a3`, verified by snapshot `e35507ce78a9543f7baed822f0d1f1b9df513f64e0e12555a1484ba98756fba4`. It reuses the original ReactVAU Stage1 Fast, final Stage2 Slow, LoRA, and projector. NC-RTED does not retrain those two stages.

Formal training is currently `0/12`; full official evaluation is `0/13`. R0 blind prediction is running, which is neither a completed evaluation nor evidence of a result. This repository makes no effectiveness or significance claim.

## Fixed Contract

The formal matrix is R0 plus A/U/S/F at seeds 17, 42, and 2026. The 12 trained members use 8,000 fixed samples, 1,000 optimizer updates, and accumulation 8. A removes auxiliary distillation, U is the strength-matched control, S distills calibrated quality only, and F distills quality plus relation-time distribution.

Full evaluation is fixed at 251 UCF-Crime videos, 800 XD-Violence videos, and 3,339 HIVAU questions, with a generation cap of 512. The historical training coverage remains 97,140 / 97,158. Teacher coverage remains limited to 265/6,000 auxiliary-valid rows, 12 positive F rows, and 15 rows where U and F differ.

## Accepted Paths

Source39 and source40 are the accepted formal-training roots. Their actual postqualification sequence is governed by `reports/nc_rted/source39_source40_postqualification_runbook_v1.md`; a profile, review, or prepared wrapper does not authorize a formal run. The required qualification evidence, resource scope, GPU and environment identity, lease, budget, deadline, and 20 GiB reserve must be bound before execution.

Prediction uses the dedicated v39 prediction source, distinct from source39/source40. Its canonical implementation manifest is `artifacts/nc_rted/prediction_implementation_v39/implementation_manifest.json` with SHA-256 `06a8d1238225c6b90a4f2a7034511b3ff55a2eb7477cd2a931828ae5c061cd21`; the inherited prediction source manifest is SHA-256 `7b36f3f377a4d14e8f33ac5850d19e7757e6b3e29fddc67b4c62ae54c215723a`.

The accepted evaluator is the v36 metric-locked implementation. Blind prediction precedes official metrics, and official labels, answers, metrics, and test outcomes cannot enter training, teacher generation, threshold fitting, configuration changes, or seed selection.

## First-Time Local Setup

For a first-time source checkout, install a PyTorch build matched to the local device and driver, then use Python 3.10 or later:

```bash
pip install -e '.[test]'
python -m pytest -q tests/test_nc_rted*.py
```

These are local setup and CPU-check examples only. They do not establish admission, replace accepted evidence, or authorize a formal run. A real ReactVAU run also requires the complete local dependency set in `external/ReactVAU-paper/requirements.txt` and the bound local artifacts. `configs/nc_rted/environment_4090_reference.json` records a prior measured environment; it is not a compatibility guarantee for another device.

## Reproduction Boundary

The public repository contains source, configuration, specifications, interfaces, and tests. Local models, media, sealed run-specific manifests, credentials, and machine logs require their approved artifact bundle and are not public-source substitutes. Reuse sealed preparation, long-input evidence, and accepted implementation bindings; do not repeat preparation or passing tests without a changed input or a concrete failure.

Final checkpoints are retained on the user-approved persistent remote data disk for later retrieval. Preserve final checkpoints, blind outputs, failures, and provenance records.

`external/ReactVAU-paper` contains the fixed upstream source and local adaptation used by this project. Its source and file hashes are recorded in `THIRD_PARTY.md` and `CODE_SNAPSHOT.json`; the upstream directory remains subject to its original noncommercial research license.

For the complete operational sequence and completion criteria, use [NC_RTED_REPRODUCTION_RUNBOOK.md](docs/NC_RTED_REPRODUCTION_RUNBOOK.md). Do not put credentials, private endpoints, or secrets in repository documentation or logs.
