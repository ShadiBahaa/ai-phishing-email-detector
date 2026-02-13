"""
Feature Engineering Module
==========================
Text preprocessing and TF-IDF feature extraction.

Usage:
    python src/features/build_features.py
"""

import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define paths
INPUT_PATH = Path("data/processed/emails_clean.csv")
FEATURES_PATH = Path("data/processed/features.joblib")
VECTORIZER_PATH = Path("src/models/tfidf_vectorizer.joblib")


def clean_text(text: str) -> str:
    """
    Clean and normalize email text.
    
    Preprocessing steps:
    1. Convert to lowercase
    2. Replace URLs with 'URL' token
    3. Replace email addresses with 'EMAIL' token
    4. Remove non-alphabetic characters
    5. Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r"http\S+|www\.\S+", "URL", text)
    
    # Replace email addresses with token
    text = re.sub(r"\S+@\S+", "EMAIL", text)
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def load_data(path: Path) -> pd.DataFrame:
    """Load cleaned email data."""
    logger.info(f"Loading data from {path}")
    
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. "
            "Run 'python src/data/make_dataset.py' first."
        )
    
    return pd.read_csv(path)


def preprocess_emails(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to all emails."""
    logger.info("Preprocessing email texts...")
    
    df = df.copy()
    df["clean_text"] = df["email_text"].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df["clean_text"].str.len() > 0]
    
    logger.info(f"Preprocessed {len(df)} emails")
    return df


def extract_features(df: pd.DataFrame, max_features: int = 5000):
    """
    Extract TF-IDF features from cleaned text.
    
    Args:
        df: DataFrame with 'clean_text' column
        max_features: Maximum number of TF-IDF features
        
    Returns:
        X: Sparse matrix of TF-IDF features
        y: Labels array
        vectorizer: Fitted TfidfVectorizer
    """
    logger.info(f"Extracting TF-IDF features (max_features={max_features})...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency
    )
    
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    
    return X, y, vectorizer


def save_artifacts(X, y, vectorizer):
    """Save features and vectorizer to disk."""
    # Create directories if needed
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save features
    joblib.dump((X, y), FEATURES_PATH)
    logger.info(f"Saved features to {FEATURES_PATH}")
    
    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info(f"Saved vectorizer to {VECTORIZER_PATH}")


def main():
    """Main feature engineering pipeline."""
    logger.info("=" * 50)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 50)
    
    try:
        # Load data
        df = load_data(INPUT_PATH)
        
        # Preprocess emails
        df = preprocess_emails(df)
        
        # Extract features
        X, y, vectorizer = extract_features(df)
        
        # Save artifacts
        save_artifacts(X, y, vectorizer)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("Feature engineering completed!")
        logger.info(f"  - Total samples: {X.shape[0]}")
        logger.info(f"  - Total features: {X.shape[1]}")
        logger.info(f"  - Phishing emails: {sum(y == 1)}")
        logger.info(f"  - Legitimate emails: {sum(y == 0)}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
