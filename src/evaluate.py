"""Score every trained model on the held-out test flights and make the confusion matrix figure.

Output:
    results/metrics.csv                cross-validation and test scores for every model
    results/classification_report.txt  per-class precision, recall and F1 for the chosen model
    figures/fig1_confusion_matrix.png  test set confusion matrix for the chosen model

Usage:
    python src/evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset.npz"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MODEL_LABELS = {"baseline": "Baseline", "logistic_regression": "Logistic regression",
                "random_forest": "Random forest", "svm": "SVM (RBF kernel)"}
CLASS_LABELS = {"car_van": "Car / van", "person": "Person", "truck_bus": "Truck / bus", "two_wheeler": "Two-wheeler"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, SURFACE = "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", [SURFACE, "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])  # same blue steps as Project 2

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def confusion_figure(y_test, pred, model_name, n_flights):
    """Figure 1: test set confusion matrix with the count in each cell."""
    classes = list(CLASS_LABELS)
    matrix = confusion_matrix(y_test, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matrix, cmap=BLUES)
    for i in range(len(classes)):
        for j in range(len(classes)):
            dark_cell = matrix[i, j] > matrix.max() / 2
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=12, color=SURFACE if dark_cell else INK)
    ax.set_xticks(range(len(classes)), [CLASS_LABELS[c] for c in classes])
    ax.set_yticks(range(len(classes)), [CLASS_LABELS[c] for c in classes])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Trucks and buses are most often mistaken for cars", pad=12)
    fig.text(0.01, -0.02, f"{model_name} on {len(y_test)} test crops from {n_flights} flights it never trained on.\n"
             "Rows are the true class, so the diagonal is the crops it got right.", fontsize=9, color=MUTED)
    fig.savefig(FIGURES / "fig1_confusion_matrix.png", facecolor=SURFACE)
    plt.close(fig)
    print("Saved figures/fig1_confusion_matrix.png")


def main():
    data = np.load(DATASET)
    test_idx = np.load(MODELS / "split.npz")["test_idx"]
    X_test, y_test, test_flights = data["X"][test_idx], data["y"][test_idx], data["groups"][test_idx]
    selection = joblib.load(MODELS / "selection.joblib")
    best = selection["best"]

    # 1. Score every model on the test flights
    rows, predictions = [], {}
    for name, cv_accuracy in selection["cv_accuracy"].items():
        model = joblib.load(MODELS / f"{name}.joblib")
        predictions[name] = model.predict(X_test)
        rows.append({"model": name,
                     "cv_accuracy": round(cv_accuracy, 3),
                     "test_accuracy": round(accuracy_score(y_test, predictions[name]), 3),
                     "test_macro_f1": round(f1_score(y_test, predictions[name], average="macro"), 3)})
    metrics = pd.DataFrame(rows)

    # 2. Save the scores, and the per-class report for the model chosen by cross-validation
    RESULTS.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    report = classification_report(y_test, predictions[best], digits=3)
    (RESULTS / "classification_report.txt").write_text(f"Model: {best}\n\n{report}")
    print(metrics.to_string(index=False))
    print(f"\n{best}\n{report}")

    # 3. Confusion matrix figure
    FIGURES.mkdir(exist_ok=True)
    confusion_figure(y_test, predictions[best], MODEL_LABELS[best], len(set(test_flights)))


if __name__ == "__main__":
    main()