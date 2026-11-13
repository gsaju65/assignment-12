"""Split the crops into train and test by flight sequence, pick a model with cross-validation, and train it.

Crops from the same flight look alike (the same parked cars show up frame after frame), so every crop from a flight goes on the same side of the split. A
random split would put near-copies of test objects into training and make the score look better than it is.

Models:
    baseline             always guesses the most common class (the floor to beat)
    logistic_regression  linear model on standardized features
    random_forest        300 decision trees
    svm                  support vector machine with an RBF kernel, the classic pairing for HOG features

Output: the four trained models, the train/test split and the cross-validation scores, in models/.

Usage:
    python src/train.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset.npz"
MODELS = ROOT / "models"

SEED = 42
TEST_SHARE = 0.2  # share of flight sequences held out for the test set
N_FOLDS = 5


def main():
    data = np.load(DATASET)
    X, y, groups = data["X"], data["y"], data["groups"]

    # 1. Hold out 20% of the flight sequences as the test set
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    assert not set(groups[train_idx]) & set(groups[test_idx])  # no flight is on both sides
    print(f"Train: {len(train_idx)} crops from {len(set(groups[train_idx]))} sequences")
    print(f"Test:  {len(test_idx)} crops from {len(set(groups[test_idx]))} sequences")

    models = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    }

    # 2. Pick the model with 5-fold cross-validation on the training flights only (the test set isn't touched until evaluate.py)
    print(f"\n{N_FOLDS}-fold cross-validation accuracy (training set):")
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X[train_idx], y[train_idx], groups=groups[train_idx],
                                 cv=GroupKFold(n_splits=N_FOLDS), scoring="accuracy")
        cv_scores[name] = scores.mean()
        print(f"  {name:20} {scores.mean():.3f} (+/- {scores.std():.3f})")
    best = max(cv_scores, key=cv_scores.get)
    print(f"Chosen model: {best}")

    # 3. Refit every model on all the training flights and save it, with the split, for evaluate.py
    MODELS.mkdir(exist_ok=True)
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        joblib.dump(model, MODELS / f"{name}.joblib")
    np.savez(MODELS / "split.npz", train_idx=train_idx, test_idx=test_idx)
    joblib.dump({"best": best, "cv_accuracy": cv_scores}, MODELS / "selection.joblib")
    print(f"Saved {len(models)} models to {MODELS.name}/")


if __name__ == "__main__":
    main()