import joblib
import json
import matplotlib.pyplot as plt


def save_model(model, path: str):
    """Save model with joblib."""
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


def load_model(path: str):
    """Load model with joblib."""
    model = joblib.load(path)
    print(f"✅ Model loaded from {path}")
    return model


def save_metrics(metrics: dict, path: str):
    """Save metrics as JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")


def save_plot(fig, path: str):
    """Save matplotlib figure to disk."""
    fig.savefig(path, bbox_inches="tight")
    print(f"Plot saved to {path}")
