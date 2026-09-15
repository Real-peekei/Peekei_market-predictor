"""
model_selection.py
Same idea as the football predictor's version, adapted for a binary
(up/down) target instead of a 3-class (H/D/A) one:

1. rolling_origin_cv() - evaluates each candidate model across several
   sequential chronological folds (expanding window: train on everything
   before a fold, test on that fold, slide forward) instead of a single
   train/test split. A single split is a noisy estimate - whichever model
   happens to win it can just be luck, and financial bar data is especially
   prone to this since bars aren't independent (today's price is mostly
   yesterday's price plus a small move) and markets are close to a random
   walk at most timeframes - exactly the regime where a flexible model is
   most likely to "win" a single split by fitting noise. Averaging log-loss
   across multiple folds gives a far more reliable read on which model (or
   the ensemble) actually performs better.

2. EnsembleClassifier - a thin wrapper that averages several fitted
   classifiers' predict_proba() output. Exposes the same predict_proba()
   interface as a normal sklearn classifier, so the rest of the codebase
   (model.py's walk_forward_backtest, server.py) doesn't need to know or
   care whether it's using a single model or an ensemble.

Whether the ensemble or a single candidate actually gets used is decided
by comparing CV scores honestly - never assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


class EnsembleClassifier:
    """
    Averages predict_proba() output from several already-fitted classifiers,
    equal-weighted. Picklable (plain Python object, no lambdas/closures) so
    it saves/loads with joblib like any other model.
    """

    def __init__(self, models: dict, classes: tuple = (0, 1)):
        self.models = models  # {name: fitted_model}
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        aligned = [_align_probs(model, X, self.classes_) for model in self.models.values()]
        return np.mean(aligned, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def _align_probs(model, X, all_classes: np.ndarray) -> np.ndarray:
    """
    model.predict_proba() only returns columns for classes actually seen
    during that model's training - if a small fold happened to contain only
    "up" bars, for instance, the output would have 1 column instead of 2,
    misaligned with a fixed label set. This pads/reorders to always match
    all_classes exactly, filling unseen classes with probability 0.
    """
    raw = model.predict_proba(X)
    aligned = np.zeros((len(X), len(all_classes)))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    for col, cls in enumerate(model.classes_):
        if cls in class_to_idx:
            aligned[:, class_to_idx[cls]] = raw[:, col]
    return aligned


def fit_named_model(model_name: str, candidate_models: dict, X, y, classes: tuple = (0, 1)):
    """
    Fits and returns either a single candidate (by name) or, if
    model_name == "ensemble", fits every candidate and wraps them in an
    EnsembleClassifier. Used both for the final live model and inside the
    walk-forward backtest's repeated retraining, so "ensemble" is a first
    -class option in both places, not just at initial training time.
    """
    if model_name == "ensemble":
        fitted = {name: make() for name, make in candidate_models.items()}
        for model in fitted.values():
            model.fit(X, y)
        return EnsembleClassifier(fitted, classes=classes)
    return candidate_models[model_name]().fit(X, y)


def rolling_origin_cv(X: pd.DataFrame, y_enc: pd.Series, candidate_models: dict,
                       n_splits: int = 5, min_train_frac: float = 0.4,
                       min_fold_rows: int = 10, classes: tuple = (0, 1)):
    """
    Expanding-window rolling CV. Returns None if there isn't enough data for
    at least one meaningful fold (caller should fall back to a simpler
    single-split evaluation in that case).

    Returns: {
      "n_folds": int,
      "candidates": {name: {"log_loss_mean": .., "log_loss_std": .., "accuracy_mean": ..}},
      "ensemble": {"log_loss_mean": .., "log_loss_std": .., "accuracy_mean": ..},
    }
    """
    n = len(X)
    all_classes = np.array(classes)

    min_train = max(30, int(n * min_train_frac))
    remaining = n - min_train
    if remaining < min_fold_rows:
        return None  # not enough data for even one honest fold

    n_splits = max(1, min(n_splits, remaining // min_fold_rows))
    fold_size = max(min_fold_rows, remaining // n_splits)

    candidate_folds = {name: {"log_loss": [], "accuracy": []} for name in candidate_models}
    ensemble_folds = {"log_loss": [], "accuracy": []}

    split_start = min_train
    folds_run = 0
    for i in range(n_splits):
        test_start = split_start
        test_end = n if i == n_splits - 1 else min(n, test_start + fold_size)
        if test_end - test_start < 1:
            break

        X_train, y_train = X.iloc[:test_start], y_enc.iloc[:test_start]
        X_test, y_test = X.iloc[test_start:test_end], y_enc.iloc[test_start:test_end]

        if y_train.nunique() < 2 or len(X_test) == 0:
            split_start = test_end
            continue

        fold_probs = []
        for name, make_model in candidate_models.items():
            model = make_model()
            model.fit(X_train, y_train)
            probs = _align_probs(model, X_test, all_classes)
            preds = all_classes[np.argmax(probs, axis=1)]
            candidate_folds[name]["log_loss"].append(log_loss(y_test, probs, labels=all_classes))
            candidate_folds[name]["accuracy"].append(accuracy_score(y_test, preds))
            fold_probs.append(probs)

        ensemble_probs = np.mean(fold_probs, axis=0)
        ensemble_preds = all_classes[np.argmax(ensemble_probs, axis=1)]
        ensemble_folds["log_loss"].append(log_loss(y_test, ensemble_probs, labels=all_classes))
        ensemble_folds["accuracy"].append(accuracy_score(y_test, ensemble_preds))

        folds_run += 1
        split_start = test_end

    if folds_run == 0:
        return None

    def summarize(fold_dict):
        return {
            "log_loss_mean": float(np.mean(fold_dict["log_loss"])),
            "log_loss_std": float(np.std(fold_dict["log_loss"])),
            "accuracy_mean": float(np.mean(fold_dict["accuracy"])),
        }

    return {
        "n_folds": folds_run,
        "candidates": {name: summarize(folds) for name, folds in candidate_folds.items()},
        "ensemble": summarize(ensemble_folds),
    }
