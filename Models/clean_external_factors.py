import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import os

def setup_logging(log_dir: str = "logs") -> None:
    """
    Sets up logging configuration with improved directory structure.
    
    Args:
        log_dir (str): Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'external_factors.log')),
            logging.StreamHandler()
        ]
    )

def create_directories() -> dict:
    """
    Creates necessary directories for data storage and results.
    
    Returns:
        dict: Dictionary containing paths to created directories
    """
    dirs = {
        'data': 'data',
        'data_raw': 'data/raw',
        'data_processed': 'data/processed',
        'results': 'results/external_factors',
        'logs': 'logs'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    return dirs

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def load_external_factor(file_path: str, value_col: str) -> pd.DataFrame:
    """
    Loads an external factor CSV and ensures proper formatting.

    Args:
        file_path (str): Path to the CSV file
        value_col (str): Name of the column containing the factor values

    Returns:
        pd.DataFrame: Cleaned DataFrame with Date and value columns

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {str(e)}")

    # Standardize column names in case of unexpected spaces
    df.columns = df.columns.str.strip()

    # Validate required columns
    required_cols = ["Date", value_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")

    # Ensure the Date column is correctly formatted
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Check for invalid dates
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} invalid dates in {file_path}")

    # Validate value column
    if df[value_col].isna().any():
        logger.warning(f"Found missing values in {value_col} column of {file_path}")

    # Keep only the Date column and the relevant factor column
    df = df[["Date", value_col]]
    return df

def clean_and_merge_external_factors(
    output_file: str,
    employment_file: str = "data/raw/employment rate.csv",
    unemployment_file: str = "data/raw/unemployment rate.csv",
    inflation_file: str = "data/raw/inflation.csv"
) -> None:
    """
    Cleans and merges Employment Rate, Unemployment Rate, and Inflation datasets.

    Args:
        output_file (str): Path where the merged dataset will be saved
        employment_file (str): Path to employment rate data
        unemployment_file (str): Path to unemployment rate data
        inflation_file (str): Path to inflation data

    Raises:
        ValueError: If there are issues with the data or merging process
    """
    try:
        # Create necessary directories
        dirs = create_directories()
        
        # Load each external factor dataset
        logger.info("Loading external factor datasets...")
        employment_df = load_external_factor(employment_file, "EmploymentRate")
        unemployment_df = load_external_factor(unemployment_file, "UnemploymentRate")
        inflation_df = load_external_factor(inflation_file, "InflationRate")

        # Merge datasets on Date
        logger.info("Merging datasets...")
        merged_df = employment_df.merge(unemployment_df, on="Date", how="left")
        merged_df = merged_df.merge(inflation_df, on="Date", how="left")

        # Validate merged dataset
        if merged_df.isna().any().any():
            logger.warning("Merged dataset contains missing values")
            
        # Sort by date
        merged_df.sort_values('Date', inplace=True)
        logger.info("Sorted merged dataset by date")

        # Save merged dataset
        output_path = os.path.join(dirs['data_processed'], output_file)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"✅ Cleaned External Factors saved as: {output_path}")
        
        # Log summary statistics
        logger.info("\nDataset Summary:")
        logger.info(f"Shape: {merged_df.shape}")
        logger.info("\nFirst few rows:")
        logger.info(merged_df.head().to_string())
        logger.info("\nMissing values:")
        logger.info(merged_df.isnull().sum().to_string())

    except Exception as e:
        logger.error(f"Error in clean_and_merge_external_factors: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Create directory structure
        dirs = create_directories()
        
        # Define output path
        output_path = "cleaned_external_factors.csv"
        
        # Process external factors
        clean_and_merge_external_factors(output_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
