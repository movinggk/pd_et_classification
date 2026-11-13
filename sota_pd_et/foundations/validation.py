"""Participant splits for validation of the actual fine-tuned checkpoint."""

import numpy as np
from sklearn.model_selection import train_test_split

from ..evaluate import derive_seed, make_inner_splits, validate_split


FINETUNE_PROTOCOL = "reserved_validation_checkpoint_v1"


def checkpoint_split(y, ids, pool, validation_fraction, inner_folds, seed):
    """Reserve validation before making inner folds on fitting participants.

    All returned indices address the original cohort, except inner positions,
    which address ``fit_indices``. No participant outside ``pool`` is inspected
    for stratification. In particular, outer-test labels cannot affect this
    split or the inner learning-rate search.
    """
    y, ids, pool = np.asarray(y), np.asarray(ids), np.asarray(pool, dtype=int)
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be finite and strictly between zero and one.")
    if (pool.ndim != 1 or not len(pool) or len(np.unique(pool)) != len(pool)
            or np.any(pool < 0) or np.any(pool >= len(y))):
        raise ValueError("Need unique participant indices inside the training pool.")
    try:
        fit, valid = train_test_split(pool, test_size=validation_fraction,
                                     random_state=derive_seed(seed, 8101), stratify=y[pool])
        fit, valid = np.sort(fit), np.sort(valid)
        validate_split(fit, valid, y)
        inner = make_inner_splits(y[fit], inner_folds, derive_seed(seed, 8102))
    except ValueError as error:
        raise ValueError("Cannot reserve stratified checkpoint validation and create inner folds. "
                         "Both held-out classes and at least --inner-folds fitting participants per class "
                         "are required. Adjust --validation-fraction or fold counts.") from error
    return {
        "training_protocol": FINETUNE_PROTOCOL,
        "split_seed": seed, "requested_validation_fraction": float(validation_fraction),
        "actual_validation_fraction": len(valid) / len(pool),
        "pool_indices": pool.tolist(), "pool_ids": ids[pool].tolist(),
        "fit_indices": fit.tolist(), "fit_ids": ids[fit].tolist(),
        "validation_indices": valid.tolist(), "validation_ids": ids[valid].tolist(),
        "inner_splits": [{
            "train_positions": a.tolist(), "validation_positions": b.tolist(),
            "train_ids": ids[fit[a]].tolist(), "validation_ids": ids[fit[b]].tolist(),
        } for a, b in inner],
    }


def make_checkpoint_plan(y, ids, outer_plan, validation_fraction, inner_folds):
    result = []
    for outer in outer_plan:
        split = checkpoint_split(y, ids, outer["train_indices"], validation_fraction,
                                 inner_folds, outer["model_seed"])
        split.update(repeat=outer["repeat"], fold=outer["fold"], model_seed=outer["model_seed"],
                     test_indices=outer["test_indices"], test_ids=outer["test_ids"])
        result.append(split)
    return result
