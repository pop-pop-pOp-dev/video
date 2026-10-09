"""Deterministic NC-RTED teacher construction, with no test-data dependency."""
from __future__ import annotations

import hashlib
import numpy as np


class TeacherError(ValueError): pass


def masked_softmax(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A finite distribution over valid relation-time cells, or all zero."""
    logits, valid = np.asarray(logits, dtype=np.float64), np.asarray(valid, dtype=bool)
    if logits.shape != valid.shape: raise TeacherError("logit and mask shapes differ")
    if not np.isfinite(logits[valid]).all(): raise TeacherError("valid logits must be finite")
    result = np.zeros_like(logits)
    if not valid.any(): return result
    selected = logits[valid]; selected -= selected.max()
    exp = np.exp(selected); result[valid] = exp / exp.sum()
    return result


def calibrated_teacher(reference_distances: np.ndarray, calibration_maxima: np.ndarray, valid: np.ndarray):
    """Return (quality, positions, 65-way target) using §4.2's fixed formula."""
    refs = np.asarray(reference_distances, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if refs.ndim != 2 or refs.shape[0] != 3 or refs.shape[1] != valid.size or valid.size > 64:
        raise TeacherError("three references and one distance per relation-time cell are required")
    if not valid.any(): raise TeacherError("all-invalid observation has no auxiliary supervision")
    if not np.isfinite(refs[:, valid]).all(): raise TeacherError("valid cells require finite reference distances")
    if (refs[:, valid] < 0).any(): raise TeacherError("distances must be nonnegative")
    distances = np.zeros(valid.size, dtype=np.float64)
    distances[valid] = np.median(refs[:, valid], axis=0)
    maximum = float(distances[valid].max()) if valid.any() else 0.0
    calibration = np.asarray(calibration_maxima, dtype=np.float64)
    if calibration.ndim != 1 or not 64 <= calibration.size <= 128 or not np.isfinite(calibration).all():
        raise TeacherError("calibration requires 64..128 finite compatible normal windows")
    if (calibration < 0).any(): raise TeacherError("calibration maxima must be nonnegative")
    p = (1 + np.count_nonzero(calibration >= maximum)) / (calibration.size + 1)
    quality = float(np.clip((.05 - p) / .05, 0, 1))
    positions = masked_softmax(distances, valid)
    joint = np.concatenate(([1 - quality], quality * positions))
    return quality, positions, joint


def validate_retrieval_support(reference_source_families: list[str], calibration_source_families: list[str], target_source_family: str) -> None:
    """Validate provenance constraints before calling pure teacher math.

    This function validates source-family counts only.  Matching, static-context
    retrieval, media overlap exclusion, and fold assignment remain manifest work.
    """
    if len(reference_source_families) != 3 or len(set(reference_source_families)) != 3:
        raise TeacherError("exactly three distinct normal reference source families are required")
    if target_source_family in reference_source_families:
        raise TeacherError("target source family cannot be a reference")
    if not 64 <= len(calibration_source_families) <= 128:
        raise TeacherError("calibration requires 64..128 windows")
    counts = {name: calibration_source_families.count(name) for name in set(calibration_source_families)}
    if len(counts) < 16 or max(counts.values()) > 4:
        raise TeacherError("calibration requires >=16 families and <=4 windows per family")
    if target_source_family in counts or counts.keys() & set(reference_source_families):
        raise TeacherError("Q, C, and R source families must be mutually exclusive")


def strength_matched_quality(f_quality: np.ndarray, raw_maxima: np.ndarray, stable_window_hashes: list[str]) -> np.ndarray:
    """The U control: rank F qualities by global M, breaking ties only by window hash."""
    quality, maxima = np.asarray(f_quality, dtype=np.float64), np.asarray(raw_maxima, dtype=np.float64)
    if quality.ndim != 1 or quality.shape != maxima.shape or len(stable_window_hashes) != quality.size:
        raise TeacherError("quality, maxima, and window hashes must align")
    if not np.isfinite(quality).all() or not np.isfinite(maxima).all() or (quality < 0).any() or (quality > 1).any() or (maxima < 0).any(): raise TeacherError("invalid teacher numerical domain")
    if len(set(stable_window_hashes)) != len(stable_window_hashes): raise TeacherError("window hashes must be unique")
    order = sorted(range(quality.size), key=lambda index: (maxima[index], stable_window_hashes[index]))
    result = np.empty_like(quality)
    result[order] = np.sort(quality, kind="stable")
    return result


def stable_window_hash(source_family: str, media_id: str, timestamp: float) -> str:
    """Teacher-only tie breaker; never pass source identity to a deployment student."""
    return hashlib.sha256(f"{source_family}\0{media_id}\0{timestamp:.6f}".encode()).hexdigest()
