import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union
import os

# Configure logging with improved directory structure
def setup_logging(log_dir: str = "logs") -> None:
    """
    Sets up logging configuration with improved directory structure.
    
    Args:
        log_dir (str): Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'clean_retail_index.log')),
            logging.StreamHandler()
        ]
    )

# Create logger instance
logger = logging.getLogger(__name__)

# Create necessary directories
def create_directories() -> None:
    """Creates necessary directories for data storage and results."""
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "logs",
        "results"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def clean_retail_index_filtered(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    measure_type: str = "value-of-retail-sales-at-current-prices",
    start_date: str = "1989-01-01",
    end_date: str = "2025-01-01"
) -> None:
    """
    Clean and filter the UK Retail Index dataset.

    Args:
        input_file (Union[str, Path]): Path to the input CSV file
        output_file (Union[str, Path]): Path to save the cleaned CSV file
        measure_type (str, optional): Type of price measure to keep. Defaults to 
            "value-of-retail-sales-at-current-prices"
        start_date (str, optional): Start date for filtering data. Format: "YYYY-MM-DD". 
            Defaults to "1989-01-01"
        end_date (str, optional): End date for filtering data. Format: "YYYY-MM-DD". 
            Defaults to "2025-01-01"

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If date parameters are invalid or measure_type is not found
    """
    # Create necessary directories
    create_directories()
    
    # Setup logging
    setup_logging()
    
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Validate dates
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    # ---- Step 1: Read raw CSV ----
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Successfully read input file: {input_file}")
        print(f"Successfully read input file: {input_file}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        raise RuntimeError(f"Error reading input file: {e}")

    # ---- Step 2: Filter for 'All retailing including automotive fuel' ----
    mask_class = df["UnofficialStandardIndustrialClassification"] == "All retailing including automotive fuel"
    df = df[mask_class].copy()
    logger.info(f"Filtered for retail classification. Remaining rows: {len(df)}")
    print(f"Filtered for retail classification. Remaining rows: {len(df)}")

    # ---- Step 3: Filter for specified measure type ----
    if measure_type not in df["type-of-prices"].unique():
        print(f"Measure type '{measure_type}' not found in data. Available types: {df['type-of-prices'].unique()}")
        raise ValueError(f"Measure type '{measure_type}' not found in data. Available types: {df['type-of-prices'].unique()}")
    
    mask_measure = df["type-of-prices"] == measure_type
    df = df[mask_measure].copy()
    logger.info(f"Filtered for measure type: {measure_type}")
    print(f"Filtered for measure type: {measure_type}")

    # ---- Step 4: Parse date from 'mmm-yy' ----
    df.rename(columns={"mmm-yy": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%b-%y", errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    print(f"Parsed and cleaned dates. Remaining rows: {len(df)}")
    
    # ---- Step 5: Filter date range ----
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    logger.info(f"Filtered date range: {start_date} to {end_date}")
    print(f"Filtered date range: {start_date} to {end_date}. Remaining rows: {len(df)}")

    # ---- Step 6: Select and rename columns ----
    df = df[["Date", "v4_1"]]
    df.rename(columns={"v4_1": "SalesIndex"}, inplace=True)
    print("Selected and renamed columns: Date, SalesIndex")
    
    # ---- Step 7: Clean and sort data ----
    df.dropna(subset=["SalesIndex"], inplace=True)
    df.sort_values("Date", inplace=True)
    print(f"Dropped NA in SalesIndex and sorted. Final rows: {len(df)}")
    
    # ---- Step 8: Save result ----
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved cleaned data to: {output_file}")
        print(f"Successfully saved cleaned data to: {output_file}")
        logger.info(f"Final dataset shape: {df.shape}")
        print(f"Final dataset shape: {df.shape}")
        logger.info("\nFirst 5 rows:")
        print("First 5 rows:")
        logger.info(df.head().to_string())
        print(df.head().to_string())
        logger.info("\nLast 5 rows:")
        print("Last 5 rows:")
        logger.info(df.tail().to_string())
        print(df.tail().to_string())
    except Exception as e:
        print(f"Error saving output file: {e}")
        raise RuntimeError(f"Error saving output file: {e}")

if __name__ == "__main__":
    import sys
    import io
    import json
    from datetime import datetime
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    sys.stdout = Tee(sys_stdout, stdout_buffer)
    sys.stderr = Tee(sys_stderr, stderr_buffer)
    output_capture = {'prints': [], 'errors': []}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_output_path = os.path.join('results', f'clean_retail_index_run_output_{timestamp}.json')
    # Example usage
    input_file = r"C:\Users\Owner\Desktop\Retail Sales Forecasting Using Predictive Modelling\Datasets\retail-sales-index-time-series-v33.csv"
    output_file = "data/processed/cleaned_retail_index.csv"
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    try:
        clean_retail_index_filtered(
            input_file=input_file,
            output_file=output_file,
            measure_type="value-of-retail-sales-at-current-prices",
            start_date="1989-01-01",
            end_date="2025-01-01"
        )
    except Exception as e:
        logger.error(f"Error processing retail index: {e}")
        output_capture['errors'].append(str(e))
        raise
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr
        stdout_contents = stdout_buffer.getvalue()
        if stdout_contents:
            output_capture['prints'].extend(stdout_contents.strip().split('\n'))
        stderr_contents = stderr_buffer.getvalue()
        if stderr_contents:
            output_capture['errors'].extend(stderr_contents.strip().split('\n'))
        try:
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_capture, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save run output JSON: {str(e)}")

