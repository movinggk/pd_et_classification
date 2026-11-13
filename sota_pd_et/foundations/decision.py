"""Select a binary decision threshold using inner validation predictions only.

PD is 0 and ET is 1. Raw margins need not be centered at zero, even when
training uses balanced class weights. One common threshold maximizes the
mean of the inner folds' balanced accuracies. This is a selection statistic,
not an estimate of held-out performance. Outer labels never enter this API.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from sklearn.metrics import balanced_accuracy_score


DECISION_RULES = ("inner-balanced", "zero")
DEFAULT_DECISION_RULE = "inner-balanced"


def validation_fold_ids(length, inner_splits):
    """Assign exactly one inner-validation fold to each training participant."""
    result = np.full(length, -1, dtype=int)
    for fold, (_, valid) in enumerate(inner_splits):
        valid = np.asarray(valid)
        if (valid.ndim != 1 or not len(valid) or not np.issubdtype(valid.dtype, np.integer)
                or np.any(valid < 0) or np.any(valid >= length)
                or len(np.unique(valid)) != len(valid) or np.any(result[valid] != -1)):
            raise ValueError("Inner validation assignments must be nonempty and disjoint.")
        result[valid] = fold
    if np.any(result < 0):
        raise ValueError("Each participant needs exactly one inner-validation prediction.")
    return result


def select_threshold(y, raw_scores, fold_ids=None, *, rule=DEFAULT_DECISION_RULE):
    """Return a finite raw-score threshold and explicit inner selection metrics.

    Each score must come from a fit that excluded that participant. ``fold_ids``
    identifies the fit producing each score; omitting it treats the arrays as
    one validation set (used for inner early stopping). Each fold must contain
    both classes. A single threshold is applied to every fold. With rule
    ``zero`` the threshold remains zero.

    Candidates are midpoints between distinct scores, the all-PD/all-ET
    boundaries when finite, and zero. They cover every attainable strict
    ``raw_score > threshold`` partition. Score ties stay together. Equal
    optima within 1e-12 favor the candidate closest to zero, then the larger
    threshold. In particular, constant scores never force ET predictions.
    """
    y, scores = np.asarray(y), np.asarray(raw_scores, dtype=np.float64)
    if (y.ndim != 1 or scores.shape != y.shape or not np.isfinite(scores).all()
            or not np.array_equal(np.unique(y), [0, 1])):
        raise ValueError("Need aligned finite scores and both PD=0 and ET=1.")
    if rule not in DECISION_RULES:
        raise ValueError(f"Unknown decision rule: {rule}.")
    fold_ids = np.zeros(len(y), dtype=int) if fold_ids is None else np.asarray(fold_ids)
    if fold_ids.shape != y.shape:
        raise ValueError("Need one inner-validation fold ID per participant.")
    labels, inverse = np.unique(fold_ids, return_inverse=True)
    weights = np.empty(len(y), dtype=float)
    for index in range(len(labels)):
        mask = inverse == index
        if not np.array_equal(np.unique(y[mask]), [0, 1]):
            raise ValueError("Every inner-validation fold must contain both classes.")
        for target in (0, 1):
            members = mask & (y == target)
            weights[members] = 1. / (2 * len(labels) * np.sum(members))

    threshold, candidate_count = 0., 1
    if rule == "inner-balanced":
        order = np.argsort(scores, kind="stable")
        ordered = scores[order]
        unique = np.unique(ordered)
        # Halving first avoids overflow when two same-sign scores are large.
        midpoints = unique[:-1] / 2 + unique[1:] / 2
        # Adjacent representable floats can round their midpoint upwards.
        midpoints = np.maximum(unique[:-1], np.minimum(midpoints, np.nextafter(unique[1:], -np.inf)))
        with np.errstate(over="ignore"):
            below = np.nextafter(unique[0], -np.inf)
        candidates = np.r_[midpoints, unique[-1], 0.]
        if np.isfinite(below):
            candidates = np.r_[below, candidates]
        candidates = np.unique(candidates)
        steps = np.where(y[order] == 0, weights[order], -weights[order])
        # Start with all ET (mean BA=0.5), then move tied rows together to PD.
        prefix = np.r_[0., np.cumsum(steps)]
        values = .5 + prefix[np.searchsorted(ordered, candidates, side="right")]
        optimum = np.max(values)
        tied = candidates[np.abs(values - optimum) <= 1e-12]
        threshold = float(min(tied, key=lambda t: (abs(float(t)), -float(t))))
        candidate_count = len(candidates)

    at_threshold, at_zero = [], []
    for index in range(len(labels)):
        mask = inverse == index
        at_threshold.append(float(balanced_accuracy_score(y[mask], scores[mask] > threshold)))
        at_zero.append(float(balanced_accuracy_score(y[mask], scores[mask] > 0)))
    return {
        "decision_rule": rule,
        "raw_decision_threshold": threshold,
        "decision_threshold": 0.,
        "output_score": "decision_score_et = raw_score_et - raw_decision_threshold; ET iff decision_score_et > 0.",
        "selection_scope": "Inner validation predictions only; no outer-test labels or scores.",
        "selection_metric": "Mean inner-fold balanced accuracy; equal weight per fold and per class.",
        "tie_break": "Closest candidate to zero, then larger threshold, within 1e-12 of the optimum.",
        "threshold_candidates": candidate_count,
        "inner_participants": len(y), "inner_folds": len(labels),
        "inner_mean_balanced_accuracy": float(np.mean(at_threshold)),
        "inner_std_balanced_accuracy": float(np.std(at_threshold, ddof=1)) if len(labels) > 1 else 0.,
        "inner_pooled_balanced_accuracy": float(balanced_accuracy_score(y, scores > threshold)),
        "inner_fold_balanced_accuracy": at_threshold,
        "inner_mean_balanced_accuracy_zero": float(np.mean(at_zero)),
        "inner_pooled_balanced_accuracy_zero": float(balanced_accuracy_score(y, scores > 0)),
        "inner_predicted_et": int(np.sum(scores > threshold)),
        "inner_predicted_et_zero": int(np.sum(scores > 0)),
        "interpretation": "Threshold/model selection statistics are optimistic; use outer held-out metrics for evaluation.",
    }


def saved_raw_threshold(metadata):
    """Restore a threshold without silently discarding a tuned decision rule.

    Earlier adapter-3 artifacts did not record a decision rule or a raw
    threshold and used zero. That compatibility case remains explicit. An
    artifact declaring inner-balanced selection must record its raw threshold.
    """
    rule = metadata.get("decision_rule", "zero")
    if rule not in DECISION_RULES:
        raise ValueError(f"Unknown saved decision rule: {rule}.")
    if rule == "inner-balanced" and "raw_decision_threshold" not in metadata:
        raise ValueError("An inner-balanced model requires raw_decision_threshold; refusing to fall back to zero.")
    value = metadata.get("raw_decision_threshold", metadata.get("decision_threshold", 0.))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Saved raw decision threshold must be a finite real number.")
    try:
        threshold = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Saved raw decision threshold must be a finite real number.") from error
    if not np.isfinite(threshold):
        raise ValueError("Saved raw decision threshold must be a finite real number.")
    return threshold


def apply_threshold(raw_scores, threshold):
    """Return float64 margins so existing ``margin > 0`` consumers stay correct."""
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Expected finite one-dimensional raw scores and a finite threshold.")
    with np.errstate(over="ignore"):
        margins = scores - float(threshold)
    if not np.isfinite(margins).all():
        raise ValueError("Threshold adjustment produced nonfinite decision scores.")
    return margins
