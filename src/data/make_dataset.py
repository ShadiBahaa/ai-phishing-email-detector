"""
Data Ingestion Module
=====================
Load raw email data and perform basic cleaning.

Usage:
    python src/data/make_dataset.py
"""

import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define paths
RAW_PATH = Path("data/raw/emails.csv")
OUT_PATH = Path("data/processed/emails_clean.csv")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw email data from CSV file."""
    logger.info(f"Loading raw data from {path}")
    
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. "
            "Please download a phishing email dataset and place it in data/raw/emails.csv"
        )
    
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and check required columns."""
    required_columns = ["email_text", "label"]
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Log data statistics
    logger.info(f"Data columns: {list(df.columns)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing invalid entries."""
    initial_count = len(df)
    
    # Remove rows with missing values in critical columns
    df = df.dropna(subset=["email_text", "label"])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["email_text"])
    
    # Ensure label is integer (0 or 1)
    df["label"] = df["label"].astype(int)
    
    # Remove empty email texts
    df = df[df["email_text"].str.strip().str.len() > 0]
    
    final_count = len(df)
    logger.info(f"Cleaned data: {initial_count} -> {final_count} records")
    logger.info(f"Removed {initial_count - final_count} invalid/duplicate entries")
    
    return df


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Save processed data to CSV."""
    # Create output directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path}")


def main():
    """Main data ingestion pipeline."""
    logger.info("=" * 50)
    logger.info("Starting data ingestion pipeline")
    logger.info("=" * 50)
    
    try:
        # Load raw data
        df = load_raw_data(RAW_PATH)
        
        # Validate data
        df = validate_data(df)
        
        # Clean data
        df = clean_data(df)
        
        # Save processed data
        save_processed_data(df, OUT_PATH)
        
        logger.info("Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
