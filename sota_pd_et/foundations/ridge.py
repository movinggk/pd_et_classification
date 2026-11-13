"""Shared linear algebra for a fold-local, class-balanced ridge alpha search.

For training rows X, labels t in {-1,+1}, and balanced sample weights s,
weighted centering yields A = sqrt(s) (X - mean_s(X)) and
b = sqrt(s) (t - mean_s(t)). For every positive alpha the solution is
    beta(alpha) = (A.T A + alpha I)^-1 A.T b,
    intercept(alpha) = mean_s(t) - mean_s(X) beta(alpha).

One eigendecomposition of the smaller primal/dual Gram matrix and one
validation projection serve every alpha in this fold. The StandardScaler is
still fitted separately inside each inner training fold. No fitted state is
shared across folds. Numerically delicate grids and classification ties use
the existing sklearn implementation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ..evaluate import choose_alpha, validate_split
from ..evaluate_transformer import scaled_features
from ..model import fit_ridge
from .decision import DEFAULT_DECISION_RULE, select_threshold, validation_fold_ids


def _reference_scores(training, y, validation, alphas):
    return np.stack([fit_ridge(training, y, alpha).decision_function(validation)
                     for alpha in alphas])


def ridge_grid_scores(training, y, validation, alphas):
    """Return shape (alphas, validation rows), matching balanced RidgeClassifier.

    Inputs are already scaled by a scaler fitted on the training rows only.
    The arrays are never modified. The default alpha grid is comfortably above
    the Gram matrix's numerical resolution. Extremely small alphas fall back
    to sklearn instead of silently changing the effective regularization.
    """
    training = np.asarray(training, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    y = np.asarray(y)
    alphas = np.asarray(alphas, dtype=np.float64)
    if (training.ndim != 2 or validation.ndim != 2 or not min(training.shape)
            or not len(validation) or training.shape[1] != validation.shape[1]
            or y.shape != (len(training),) or not np.array_equal(np.unique(y), [0, 1])):
        raise ValueError("Need aligned finite training/validation matrices and both PD=0, ET=1.")
    if (alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all()
            or np.any(alphas <= 0)):
        raise ValueError("Alpha candidates must be finite and strictly positive.")
    if not np.isfinite(training).all() or not np.isfinite(validation).all():
        raise ValueError("Ridge inputs must be finite.")

    weights = len(y) / (2 * np.bincount(y.astype(np.int64), minlength=2))
    sample_weights = weights[y.astype(np.int64)]
    targets = 2 * y.astype(np.float64) - 1
    x_mean = np.average(training, axis=0, weights=sample_weights)
    y_mean = np.average(targets, weights=sample_weights)
    roots = np.sqrt(sample_weights)
    A = (training - x_mean) * roots[:, None]
    b = (targets - y_mean) * roots
    validation_centered = validation - x_mean

    dual = A.shape[0] <= A.shape[1]
    gram = A @ A.T if dual else A.T @ A
    try:
        eigenvalues, vectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        return _reference_scores(training, y, validation, alphas)
    resolution = np.finfo(np.float64).eps * max(A.shape) * max(1., float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -resolution or np.min(alphas) <= resolution:
        return _reference_scores(training, y, validation, alphas)
    eigenvalues = np.maximum(eigenvalues, 0.)
    if dual:
        projected_targets = vectors.T @ b
        projected_validation = (validation_centered @ A.T) @ vectors
    else:
        projected_targets = vectors.T @ (A.T @ b)
        projected_validation = validation_centered @ vectors
    coefficients = projected_targets[:, None] / (eigenvalues[:, None] + alphas[None, :])
    scores = (projected_validation @ coefficients).T + y_mean
    if not np.isfinite(scores).all():
        return _reference_scores(training, y, validation, alphas)

    # A margin within numerical roundoff of zero can change the strict > 0
    # class decision. Recompute such candidates with the reference solver.
    for index, alpha in enumerate(alphas):
        tie_tolerance = 1e-10 * max(1., float(np.max(np.abs(scores[index]))))
        if np.any(np.abs(scores[index]) <= tie_tolerance):
            scores[index] = fit_ridge(training, y, alpha).decision_function(validation)
    return scores


def tune_alpha(Z, y, alphas, inner_splits, *, decision_rule=DEFAULT_DECISION_RULE):
    """Search alpha and, optionally, a common inner-OOF decision threshold.

    Returned OOF scores are raw. The selected search row records their raw
    threshold. The default tunes it by mean inner-fold balanced accuracy.
    Explicit ``zero`` retains the original reference-search outputs.
    """
    alphas = list(alphas)
    y = np.asarray(y)
    oof = np.full((len(alphas), len(y)), np.nan)
    fold_ba = np.empty((len(alphas), len(inner_splits)))
    assignments = []
    for fold, (train, valid) in enumerate(inner_splits):
        validate_split(train, valid, y)
        assignments.append((train, valid))
        _, training, validation = scaled_features(Z, train, valid)
        scores = ridge_grid_scores(training, y[train], validation, alphas)
        oof[:, valid] = scores
        for index in range(len(alphas)):
            fold_ba[index, fold] = balanced_accuracy_score(y[valid], scores[index] > 0)
    if not np.isfinite(oof).all():
        raise RuntimeError("Missing/nonfinite inner out-of-fold scores.")
    fold_ids = validation_fold_ids(len(y), assignments)
    search = [{
        "alpha": float(alpha),
        "inner_mean_balanced_accuracy": float(np.mean(fold_ba[index])),
        "inner_std_balanced_accuracy": float(np.std(fold_ba[index], ddof=1)),
        "inner_pooled_balanced_accuracy": float(balanced_accuracy_score(y, oof[index] > 0)),
        "inner_pooled_roc_auc": float(roc_auc_score(y, oof[index])),
    } for index, alpha in enumerate(alphas)]
    if decision_rule != "zero":
        for index, row in enumerate(search):
            selection = select_threshold(y, oof[index], fold_ids, rule=decision_rule)
            row.update({key: selection[key] for key in (
                "decision_rule", "raw_decision_threshold", "inner_mean_balanced_accuracy",
                "inner_std_balanced_accuracy", "inner_pooled_balanced_accuracy",
                "inner_mean_balanced_accuracy_zero", "inner_pooled_balanced_accuracy_zero",
            )})
    selected = choose_alpha(search)
    return selected, search, oof[alphas.index(selected)]
