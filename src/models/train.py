"""
Model Training Module
=====================
Train a machine learning model for phishing email detection.

Usage:
    python src/models/train.py
"""

import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define paths
FEATURES_PATH = Path("data/processed/features.joblib")
MODEL_PATH = Path("src/models/phishing_model.joblib")
TEST_DATA_PATH = Path("data/processed/test_data.joblib")


def load_features(path: Path):
    """Load preprocessed features."""
    logger.info(f"Loading features from {path}")
    
    if not path.exists():
        raise FileNotFoundError(
            f"Features not found at {path}. "
            "Run 'python src/features/build_features.py' first."
        )
    
    X, y = joblib.load(path)
    logger.info(f"Loaded features with shape: {X.shape}")
    
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split data into training and test sets."""
    logger.info(f"Splitting data (test_size={test_size})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,  # Maintain class balance
        random_state=random_state
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Logistic Regression model.
    
    Logistic Regression is chosen for:
    - Interpretability (feature importance via coefficients)
    - Fast training and inference
    - Good baseline performance for text classification
    """
    logger.info("Training Logistic Regression model...")
    
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    logger.info(f"Training accuracy: {train_acc:.4f}")
    
    return model


def cross_validate_model(model, X, y, cv: int = 5):
    """Perform cross-validation to assess model stability."""
    logger.info(f"Performing {cv}-fold cross-validation...")
    
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    
    logger.info(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores


def save_artifacts(model, X_test, y_test):
    """Save trained model and test data."""
    # Create directories if needed
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")
    
    # Save test data for evaluation
    joblib.dump((X_test, y_test), TEST_DATA_PATH)
    logger.info(f"Saved test data to {TEST_DATA_PATH}")


def main():
    """Main training pipeline."""
    logger.info("=" * 50)
    logger.info("Starting model training pipeline")
    logger.info("=" * 50)
    
    try:
        # Load features
        X, y = load_features(FEATURES_PATH)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Cross-validation
        cross_validate_model(model, X_train, y_train)
        
        # Quick test evaluation
        test_acc = accuracy_score(y_test, model.predict(X_test))
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Save artifacts
        save_artifacts(model, X_test, y_test)
        
        logger.info("=" * 50)
        logger.info("Model training completed successfully!")
        logger.info("Run 'python src/models/evaluate.py' for detailed evaluation")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()
