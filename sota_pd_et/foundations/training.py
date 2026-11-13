"""Supervised backbone fine-tuning with validation checkpoint restoration.

The common head is LayerNorm -> Dropout -> Linear, trained with class-balanced
binary cross entropy. Every fit creates a fresh pretrained backbone. Validation
predictions select epochs only inside a training pool. Retained models restore
their best validation checkpoint; no outer-test labels select training settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import ExitStack
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .encoders import EncoderOptions, load_encoder
from .decision import DECISION_RULES, DEFAULT_DECISION_RULE, select_threshold
from .progress import progress_bar, progress_message


@dataclass(frozen=True)
class TrainOptions:
    learning_rate: float = 1e-5
    head_learning_rate: float | None = None
    weight_decay: float = .01
    max_epochs: int = 10
    patience: int = 3
    batch_size: int = 2
    accumulation_steps: int = 4
    dropout: float = .1
    clip_grad_norm: float = 1.
    decision_rule: str = DEFAULT_DECISION_RULE

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.learning_rate, self.clip_grad_norm)):
            raise ValueError("Learning rates and gradient norm must be finite and positive.")
        if self.head_learning_rate is not None and (not math.isfinite(self.head_learning_rate)
                                                   or self.head_learning_rate <= 0):
            raise ValueError("The head learning rate must be finite and positive, or None for automatic scaling.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid weight decay or dropout.")
        if min(self.max_epochs, self.patience, self.batch_size, self.accumulation_steps) < 1:
            raise ValueError("Epoch, patience, batch, and accumulation counts must be positive.")
        if self.decision_rule not in DECISION_RULES:
            raise ValueError(f"Unknown decision rule: {self.decision_rule}.")


def resolve_head_learning_rate(feature_dim, requested=None):
    """Keep Adam's coherent first-update margin change from growing with width.

    For LayerNorm features z, sum(abs(z)) is at most D initially (ignoring
    dropout). Adam's first coordinate steps can align with a common z, giving
    a margin change of order lr*D even after global gradient clipping. Scale
    the automatic rate inversely with D above a reference width of 256.
    This is a deterministic stability default, not a fitted hyperparameter.
    Explicit rates retain their exact meaning for reproducing earlier runs.
    """
    if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim < 1:
        raise ValueError("Feature dimension must be a positive integer.")
    if requested is not None:
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("Head learning rate must be finite and positive.")
        return float(requested)
    return 1e-3 * min(1., 256. / feature_dim)


class FineTunedClassifier(nn.Module):
    def __init__(self, encoder, feature_dim, dropout=.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, 1))
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _tensor(rows, device):
    values = np.asarray(rows, dtype=np.float32)
    # Advanced NumPy indexing already creates an owned batch. Recopy only
    # inputs that torch cannot safely view (read-only or negative-strided).
    if not values.flags.writeable or any(stride < 0 for stride in values.strides):
        values = np.array(values, dtype=np.float32, copy=True)
    return torch.from_numpy(values).to(device)


def predict_scores(model, X, batch_size=2, *, progress=False, description="Prediction",
                   progress_position=0):
    model.eval()
    scores = []
    device = model.encoder.options.device
    with torch.inference_mode(), progress_bar(
        total=len(X), desc=description, unit="participant", enabled=progress, position=progress_position,
    ) as bar:
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            scores.append(model(_tensor(batch, device)).float())
            bar.update(len(batch))
    # All outputs together contain only one scalar per participant. One host
    # transfer avoids forcing the GPU queue to finish after every tiny batch.
    result = torch.cat(scores).cpu().numpy()
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError("Expected one finite classifier margin per participant.")
    return result


def _audit_backbone_gradients(parameters, finite_loss):
    """Check a complete update with one small device-to-host transfer.

    The maximum infinity norm is finite iff every dense gradient is finite;
    it is nonzero iff a backbone gradient has a nonzero element. Unlike a
    squared L2 norm this test cannot underflow tiny nonzero gradients to zero.
    """
    active = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not active:
        if not bool(finite_loss.item()):
            raise RuntimeError("Nonfinite fine-tuning loss.")
        raise RuntimeError("No backbone gradients: refusing to call this fine-tuning.")
    peak = torch.stack([gradient.detach().abs().amax() for gradient in active]).amax()
    peak_value, loss_valid = torch.stack((peak, finite_loss.to(peak.dtype))).cpu().tolist()
    if not loss_valid:
        raise RuntimeError("Nonfinite fine-tuning loss.")
    if not math.isfinite(peak_value):
        raise RuntimeError("Nonfinite backbone gradients.")
    return peak_value != 0.


def _progress_refresh_due(bar, increment):
    return (not bar.disable and
            bar.n + increment - bar.last_print_n >= bar.miniters and
            bar._time() - bar.last_print_t >= bar.mininterval)


def new_classifier(encoder_options, train_options, sample, cache_dir, loader=None):
    loader = loader or load_encoder
    encoder = loader(encoder_options, cache_dir)
    encoder.eval()
    dimension = getattr(encoder, "feature_dimension", None)
    if callable(dimension):
        feature_dim = int(dimension(sample.shape[1]))
    else:
        with torch.no_grad():
            feature_dim = encoder(_tensor(sample[:1], encoder_options.device)).shape[-1]
    encoder.requires_grad_(True)
    return FineTunedClassifier(encoder, feature_dim, train_options.dropout).to(encoder_options.device)


def fit_classifier(X, y, encoder_options: EncoderOptions, options: TrainOptions,
                   seed, cache_dir, *, validation=None, epochs=None, loader=None, verbose=False,
                   progress=False, description=None, retain_model=True):
    """Fit fresh weights and restore the best checkpoint when retaining a model.

    Inner searches use retain_model=False: only best-epoch predictions and
    selection metadata survive, avoiding large, unused checkpoint writes.
    Fixed-epoch fits without validation remain a low-level training utility;
    the comparison runner always validates its deployed neural checkpoint.
    """
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
    if X.ndim != 3 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("Training inputs must be finite participant/channel/time arrays.")
    if len(X) != len(y) or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Training requires aligned participant arrays and both PD=0 and ET=1.")
    if validation is None and (epochs is None or epochs < 1):
        raise ValueError("A fit without validation needs a fixed positive epoch count.")
    if validation is not None and epochs is not None:
        raise ValueError("Specify validation early stopping or fixed epochs, not both.")
    if validation is not None:
        X_valid, y_valid = validation
        X_valid, y_valid = np.asarray(X_valid, dtype=np.float32), np.asarray(y_valid)
        if (X_valid.ndim != X.ndim or X_valid.shape[1:] != X.shape[1:] or not len(X_valid)
                or y_valid.shape != (len(X_valid),) or not np.isfinite(X_valid).all()):
            raise ValueError("Validation needs aligned finite participant arrays with the training input shape.")
        if not np.array_equal(np.unique(y_valid), [0, 1]):
            raise ValueError("Early stopping requires both classes in validation.")
    seed_everything(seed)
    model = new_classifier(encoder_options, options, X, cache_dir, loader)
    head_learning_rate = resolve_head_learning_rate(model.feature_dim, options.head_learning_rate)
    resolved_options = replace(options, head_learning_rate=head_learning_rate)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": options.learning_rate},
        {"params": model.head.parameters(), "lr": head_learning_rate},
    ], weight_decay=options.weight_decay)
    backbone_parameters = tuple(model.encoder.parameters())
    model_parameters = tuple(model.parameters())
    counts = np.bincount(y, minlength=2)
    class_weights = torch.tensor(len(y) / (2 * counts), device=encoder_options.device, dtype=torch.float32)
    if verbose or progress:
        policy = "automatic" if options.head_learning_rate is None else "explicit"
        progress_message(f"    Head: {model.feature_dim} features, LR={head_learning_rate:.6g} ({policy}); "
                         f"training PD={counts[0]}, ET={counts[1]}", enabled=progress)
    generator = np.random.default_rng(seed)
    history, best_score, best_epoch, best_predictions = [], -math.inf, None, None
    best_selection = None
    limit = options.max_epochs if validation is not None else epochs
    backbone_gradient_seen = False
    with ExitStack() as resources:
        checkpoint_path = None
        if validation is not None and retain_model:
            directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="foundation_best_"))
            checkpoint_path = Path(directory) / "best.pt"
        epoch_bar = resources.enter_context(progress_bar(
            total=limit, desc=description or f"{encoder_options.model} fine-tuning",
            unit="epoch", enabled=progress, position=3,
        ))
        for epoch in range(1, limit + 1):
            model.train()
            order = generator.permutation(len(y))
            batches = [order[i:i + options.batch_size] for i in range(0, len(y), options.batch_size)]
            epoch_loss = torch.zeros((), device=encoder_options.device)
            epoch_loss_terms, seen = [], 0
            with progress_bar(
                total=len(y), desc=f"Epoch {epoch}/{limit}: training", unit="participant",
                enabled=progress, position=4,
            ) as batch_bar:
                for begin in range(0, len(batches), options.accumulation_steps):
                    accumulation = batches[begin:begin + options.accumulation_steps]
                    denominator = sum(len(indices) for indices in accumulation)
                    optimizer.zero_grad(set_to_none=True)
                    finite_loss = torch.ones((), device=encoder_options.device, dtype=torch.bool)
                    for indices in accumulation:
                        values = _tensor(X[indices], encoder_options.device)
                        target = torch.tensor(y[indices], device=encoder_options.device, dtype=torch.long)
                        logits = model(values)
                        losses = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
                        weighted = (losses * class_weights[target]).sum()
                        loss = weighted / denominator
                        finite_loss.logical_and_(torch.isfinite(loss.detach()))
                        loss.backward()
                        detached_loss = weighted.detach()
                        epoch_loss.add_(detached_loss)
                        epoch_loss_terms.append(detached_loss)
                        seen += len(indices)
                        if _progress_refresh_due(batch_bar, len(indices)):
                            batch_bar.set_postfix(loss=f"{float(epoch_loss) / seen:.5f}", refresh=False)
                        batch_bar.update(len(indices))
                    if backbone_gradient_seen:
                        # Once connectivity is verified, clipping below checks
                        # the finite norm of every parameter on every update.
                        # Avoid launching a second reduction for each backbone
                        # tensor merely to repeat the same nonzero audit.
                        if not bool(finite_loss.item()):
                            raise RuntimeError("Nonfinite fine-tuning loss.")
                        if not any(parameter.grad is not None for parameter in backbone_parameters):
                            raise RuntimeError("No backbone gradients: refusing to call this fine-tuning.")
                    else:
                        backbone_gradient_seen = _audit_backbone_gradients(backbone_parameters, finite_loss)
                        if backbone_gradient_seen and (verbose or progress):
                            progress_message("    Verified finite, nonzero backbone gradients.", enabled=progress)
                    nn.utils.clip_grad_norm_(model_parameters, options.clip_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
            # Transfer tiny loss scalars once per epoch and sum in Python's
            # float64 order, preserving the previous loss-history arithmetic.
            recorded_loss = sum(torch.stack(epoch_loss_terms).cpu().tolist())
            row = {"epoch": epoch, "training_weighted_bce": recorded_loss / len(y)}
            postfix = {"loss": f"{row['training_weighted_bce']:.5f}"}
            if validation is not None:
                predictions = predict_scores(model, X_valid, options.batch_size, progress=progress,
                                              description="Validation", progress_position=4)
                selection = select_threshold(y_valid, predictions, rule=options.decision_rule)
                score = selection["inner_mean_balanced_accuracy"]
                row["validation_balanced_accuracy"] = score
                row["validation_balanced_accuracy_zero"] = selection["inner_mean_balanced_accuracy_zero"]
                row["validation_raw_decision_threshold"] = selection["raw_decision_threshold"]
                row["validation_predicted_et"] = selection["inner_predicted_et"]
                row["validation_predicted_et_zero"] = selection["inner_predicted_et_zero"]
                row["validation_participants"] = len(predictions)
                row["validation_score_min"] = float(np.min(predictions))
                row["validation_score_max"] = float(np.max(predictions))
                if score > best_score + 1e-12:
                    best_score, best_epoch, best_predictions = score, epoch, predictions.copy()
                    best_selection = selection
                    if checkpoint_path is not None:
                        # Serialize immediately: state_dict() itself is a live
                        # view. Streaming storage avoids an extra full CPU clone.
                        torch.save(model.state_dict(), checkpoint_path)
                postfix.update(validation_BA=f"{score:.4f}", raw_cutoff=f"{selection['raw_decision_threshold']:.4g}", best_epoch=best_epoch,
                               predicted_ET=f"{row['validation_predicted_et']}/{len(predictions)}")
                history.append(row)
                if verbose and not progress:
                    print(f"      epoch {epoch}/{limit}: loss={row['training_weighted_bce']:.5f}, "
                          f"validation BA={score:.4f}, raw threshold={selection['raw_decision_threshold']:.4g}, "
                          f"predicted ET={row['validation_predicted_et']}/{len(predictions)}, "
                          f"best epoch={best_epoch}", flush=True)
            else:
                history.append(row)
                if verbose and not progress:
                    print(f"      epoch {epoch}/{limit}: loss={row['training_weighted_bce']:.5f}", flush=True)
            epoch_bar.set_postfix(postfix, refresh=False)
            epoch_bar.update(1)
            if validation is not None and epoch - best_epoch >= options.patience:
                if verbose and progress:
                    progress_message(f"    Early stopping at epoch {epoch}; best validation epoch={best_epoch}.", enabled=True)
                break
        model.zero_grad(set_to_none=True)
        if checkpoint_path is not None:
            # mmap reads checkpoint tensors on demand instead of allocating a
            # second full CPU model before copying back to the active device.
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
            model.load_state_dict(state, strict=True)
            del state
            model.eval()
    if not backbone_gradient_seen:
        raise RuntimeError("No nonzero backbone gradients during fine-tuning.")
    return model if retain_model else None, {
        "history": history, "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_score if validation is not None else None,
        "best_validation_scores": best_predictions,
        "best_threshold_selection": best_selection,
        "best_checkpoint_restored": validation is not None and retain_model,
        "backbone_gradient_verified": backbone_gradient_seen,
        "epochs_trained": len(history), "train_options": asdict(resolved_options),
        "requested_head_learning_rate": options.head_learning_rate,
        "head_learning_rate_policy": "auto_width_scaled" if options.head_learning_rate is None else "explicit",
        "feature_dimension": model.feature_dim,
        "training_class_counts": {"pd": int(counts[0]), "et": int(counts[1])},
        "training_class_weights": {"pd": float(len(y) / (2 * counts[0])),
                                   "et": float(len(y) / (2 * counts[1]))},
        "encoder_provenance": model.encoder.provenance,
    }
