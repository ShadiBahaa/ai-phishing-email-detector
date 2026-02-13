"""
Model Evaluation Module
=======================
Evaluate the trained phishing detection model.

Usage:
    python src/models/evaluate.py
"""

import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = Path("src/models/phishing_model.joblib")
TEST_DATA_PATH = Path("data/processed/test_data.joblib")
REPORT_PATH = Path("docs/evaluation_report.txt")
PLOTS_DIR = Path("docs/plots")


def load_model_and_data():
    """Load trained model and test data."""
    logger.info("Loading model and test data...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python src/models/train.py' first."
        )
    
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found at {TEST_DATA_PATH}. "
            "Run 'python src/models/train.py' first."
        )
    
    model = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(TEST_DATA_PATH)
    
    logger.info(f"Test set size: {X_test.shape[0]} samples")
    
    return model, X_test, y_test


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    
    return metrics


def print_evaluation_report(y_true, y_pred, metrics):
    """Print comprehensive evaluation report."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    print("\n📊 CLASSIFICATION REPORT:")
    print("-" * 40)
    print(classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Phishing"]
    ))
    
    print("\n📈 SUMMARY METRICS:")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"  {name.upper():15s}: {value:.4f}")
    
    print("\n📋 CONFUSION MATRIX:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    print(f"  True Negatives  (TN): {cm[0][0]:5d}  |  False Positives (FP): {cm[0][1]:5d}")
    print(f"  False Negatives (FN): {cm[1][0]:5d}  |  True Positives  (TP): {cm[1][1]:5d}")
    
    print("\n🎯 INTERPRETATION:")
    print("-" * 40)
    print(f"  - {cm[1][1]} phishing emails correctly identified")
    print(f"  - {cm[0][0]} legitimate emails correctly identified")
    print(f"  - {cm[0][1]} legitimate emails incorrectly flagged as phishing (False Alarm)")
    print(f"  - {cm[1][0]} phishing emails missed (DANGEROUS!)")
    
    print("\n" + "=" * 60)


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Phishing Email Detection")
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to {save_path}")
    
    plt.close()


def plot_roc_curve(y_true, y_prob, metrics, save_path: Path = None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, 
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", 
             label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Phishing Email Detection")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curve plot to {save_path}")
    
    plt.close()


def save_report_to_file(y_true, y_pred, metrics, report_path: Path):
    """Save evaluation report to text file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PHISHING EMAIL DETECTOR - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(
            y_true, y_pred,
            target_names=["Legitimate", "Phishing"]
        ))
        
        f.write("\nSUMMARY METRICS:\n")
        f.write("-" * 40 + "\n")
        for name, value in metrics.items():
            f.write(f"  {name.upper():15s}: {value:.4f}\n")
        
        f.write("\nCONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"  [[TN={cm[0][0]:5d}, FP={cm[0][1]:5d}],\n")
        f.write(f"   [FN={cm[1][0]:5d}, TP={cm[1][1]:5d}]]\n")
    
    logger.info(f"Saved evaluation report to {report_path}")


def main():
    """Main evaluation pipeline."""
    logger.info("=" * 50)
    logger.info("Starting model evaluation")
    logger.info("=" * 50)
    
    try:
        # Load model and data
        model, X_test, y_test = load_model_and_data()
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        
        # Print report
        print_evaluation_report(y_test, y_pred, metrics)
        
        # Generate plots
        plot_confusion_matrix(
            y_test, y_pred, 
            save_path=PLOTS_DIR / "confusion_matrix.png"
        )
        plot_roc_curve(
            y_test, y_prob, metrics,
            save_path=PLOTS_DIR / "roc_curve.png"
        )
        
        # Save report to file
        save_report_to_file(y_test, y_pred, metrics, REPORT_PATH)
        
        logger.info("=" * 50)
        logger.info("Evaluation completed successfully!")
        logger.info(f"  - Report saved to: {REPORT_PATH}")
        logger.info(f"  - Plots saved to: {PLOTS_DIR}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
