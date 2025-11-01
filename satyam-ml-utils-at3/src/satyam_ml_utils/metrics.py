from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification(model, X_test, y_test):
    """
    Evaluate classification model and return metrics as dict.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        if hasattr(model, "predict_proba") else np.nan,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    """
    Plot confusion matrix as heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_regression(model, X_test, y_test):
    """
    Evaluate regression model and return metrics as dict.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    return metrics
