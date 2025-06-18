import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import os
import logging
from pathlib import Path

# Set up visualization defaults for better quality plots
plt.rcParams['figure.dpi'] = 300  # Higher resolution
plt.rcParams['font.size'] = 12    # Larger base font size
plt.rcParams['font.family'] = 'serif'  # More professional font
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def setup_logging(log_dir: str = "logs") -> None:
    """
    Sets up logging configuration with improved directory structure.
    
    Args:
        log_dir (str): Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'differencing.log'), encoding='utf-8'),
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
        'results': 'results/differencing',
        'plots': 'results/differencing/plots',
        'logs': 'logs'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    return dirs

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Loads the enhanced dataset (with non-stationary columns) and ensures proper datetime formatting.
    Also extracts Month, Year, and Quarter from Date.
    
    Args:
        file_path (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Loaded and processed DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Extract Month, Year, and Quarter from Date
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Quarter"] = df["Date"].dt.quarter
        
        logger.info(f"Successfully loaded data from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def adf_test(series, col_name="Series"):
    """
    Performs the Augmented Dickey-Fuller test.
    """
    result = adfuller(series.dropna())
    print(f"\nADF Test on {col_name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value:       {result[1]:.4f}")
    print(f"  Critical Values: {result[4]}")
    if result[1] <= 0.05:
        print("  ✅ Stationary (Reject null hypothesis).")
    else:
        print("  ⚠️ Non-stationary (Fail to reject null). Consider further differencing or transformation.")

def kpss_test(series, col_name="Series"):
    """
    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.
    """
    try:
        result = kpss(series.dropna(), regression='c', nlags='auto')
        print(f"\nKPSS Test on {col_name}:")
        print(f"  KPSS Statistic: {result[0]:.4f}")
        print(f"  p-value:        {result[1]:.4f}")
        print(f"  Critical Values: {result[3]}")
        if result[1] > 0.05:
            print("  Stationary (Fail to reject null).")
        else:
            print("  Non-stationary (Reject null). Consider further differencing or transformation.")
    except ValueError as e:
        print(f"KPSS test error on {col_name}: {e}")

def save_original_data(df, output_file):
    """
    Saves the original data with specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_file (str): Path to save the output file
    """
    try:
        original_columns = [
            "Date", "SalesIndex", "EmploymentRate", "UnemploymentRate",
            "InflationRate", "Month", "Year", "Quarter"
        ]
        
        # Ensure all required columns exist
        missing_cols = [col for col in original_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        df_original = df[original_columns].copy()
        df_original.to_csv(output_file, index=False)
        logger.info(f"Saved original dataset as {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving original data: {str(e)}")
        raise

def save_differenced_data(df, output_file):
    """
    Saves the differenced data with specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_file (str): Path to save the output file
    """
    try:
        diff_columns = ["Date"]
        target_cols = ["SalesIndex", "EmploymentRate", "UnemploymentRate", "InflationRate"]
        
        for col in target_cols:
            diff_col = f"{col}_Diff1"
            if diff_col not in df.columns:
                raise ValueError(f"Missing differenced column: {diff_col}")
            diff_columns.append(diff_col)
        
        df_differenced = df[diff_columns].copy()
        df_differenced.to_csv(output_file, index=False)
        logger.info(f"[SUCCESS] Saved differenced dataset as {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving differenced data: {str(e)}")
        raise

def plot_differencing_results(df, target_cols, output_dir):
    """
    Creates visualization plots for differenced data, handling arbitrary differencing order columns.
    Args:
        df (pd.DataFrame): DataFrame containing the data
        target_cols (list): List of columns to plot (should be the actual differenced columns)
        output_dir (str): Directory to save plots
    """
    import re
    for col in target_cols:
        # Try to extract the base column and differencing order
        m = re.match(r"(.+)_Diff(\d+)$", col)
        if m:
            base_col = m.group(1)
            diff_order = int(m.group(2))
        else:
            base_col = col
            diff_order = 0
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        ax1, ax2 = axes
        # Plot original series if available
        if base_col in df.columns:
            ax1.plot(df['Date'], df[base_col], label=f'Original {base_col}', linewidth=2)
            ax1.set_title(f'Original {base_col}', fontsize=14, pad=10)
        else:
            ax1.text(0.5, 0.5, f'Original {base_col} not available', ha='center', va='center', fontsize=12)
            ax1.set_title(f'Original {base_col} (not available)', fontsize=14, pad=10)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        # Plot differenced series
        if col in df.columns:
            ax2.plot(df['Date'], df[col], label=f'Differenced ({col})', color='orange', linewidth=2)
            ax2.set_title(f'Differenced {base_col} (order {diff_order})', fontsize=14, pad=10)
        else:
            ax2.text(0.5, 0.5, f'{col} not available', ha='center', va='center', fontsize=12)
            ax2.set_title(f'Differenced {base_col} (order {diff_order}) (not available)', fontsize=14, pad=10)
            logger.warning(f"{col} not found in DataFrame for plotting.")
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_differencing.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved differencing plot for {col}")

def main():
    dirs = create_directories()
    import sys
    import io
    import json
    from datetime import datetime
    # Capture print and error output
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
    json_output_path = os.path.join('results', 'differencing', f'diff_run_output_{timestamp}.json')
    try:
        """
        Main function to process and difference the retail dataset.
        """
        import glob
        import json
        # Input: latest enhanced_retail_dataset_*.csv from results/eda/
        eda_dir = os.path.join('results', 'eda')
        pattern = os.path.join(eda_dir, 'enhanced_retail_dataset_*.csv')
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            raise FileNotFoundError(f"No enhanced_retail_dataset_*.csv found in {eda_dir}")
        input_file = files[0]
        logger.info(f"Using latest EDA-enhanced dataset from EDA output: {input_file}")
        
        # Output paths
        original_output = os.path.join(dirs['data_processed'], "original_data.csv")
        differenced_output = os.path.join(dirs['data_processed'], "differenced_data.csv")
        enhanced_output = os.path.join(dirs['data_processed'], "enhanced_retail_dataset_with_diff.csv")

        # Load the data
        df = load_data(input_file)
        
        # Save original data first
        save_original_data(df, original_output)

        # Read stationarity recommendation
        rec_file = os.path.join('results', 'stationarity', 'stationarity_recommendation.json')
        if not os.path.exists(rec_file):
            raise FileNotFoundError(f"Stationarity recommendation file not found: {rec_file}")
        with open(rec_file, 'r') as f:
            recommendation = json.load(f)
        columns_to_difference = recommendation.get('columns_to_difference', [])

        print("\nColumns to difference as recommended:")
        logger.info(f"Columns to difference: {columns_to_difference}")
        for col in columns_to_difference:
            print(f"  - {col}")
        if not columns_to_difference:
            print("  None. No differencing required. Saving original data as differenced_data.csv.")
            df.to_csv(differenced_output, index=False)
            logger.info(f"No differencing required. Original data saved as {differenced_output}")
            return

        # Enhanced: Iterative differencing with post-checks
        max_diff = 3
        diff_tracker = {col: 0 for col in columns_to_difference}
        df_work = df.copy()
        still_non_stationary = columns_to_difference.copy()
        diff_cols = []
        for diff_round in range(1, max_diff + 1):
            next_non_stationary = []
            print(f"\n--- Differencing round {diff_round} ---")
            for col in still_non_stationary:
                base_col = col if diff_tracker[col] == 0 else f"{col}_Diff{diff_tracker[col]}"
                diff_col = f"{col}_Diff{diff_tracker[col] + 1}"
                df_work[diff_col] = df_work[base_col].diff()
                diff_tracker[col] += 1
                # Drop initial NaN rows for testing
                test_series = df_work[diff_col].dropna()
                # Stationarity test
                adf_result = adfuller(test_series)
                kpss_result = kpss(test_series, regression='c', nlags='auto')
                adf_p = adf_result[1]
                kpss_p = kpss_result[1]
                print(f"{diff_col}: ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}")
                logger.info(f"{diff_col}: ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}")
                if (adf_p > 0.05) or (kpss_p <= 0.05):
                    next_non_stationary.append(col)
                else:
                    print(f"{col} became stationary after {diff_tracker[col]} difference(s).")
                    logger.info(f"{col} became stationary after {diff_tracker[col]} difference(s).")
                    diff_cols.append(diff_col)
            if not next_non_stationary:
                break
            still_non_stationary = next_non_stationary
        # Final cleanup: drop rows with any NaN in final differenced columns
        keep_cols = ['Date'] + diff_cols
        df_final = df_work[keep_cols].dropna()
        # Plot final differenced columns (use the actual differenced columns)
        plot_differencing_results(df_final, diff_cols, dirs['plots'])
        # Save datasets
        df_final.to_csv(differenced_output, index=False)
        logger.info(f"[SUCCESS] Saved differenced dataset as {differenced_output}")
        df_work.to_csv(enhanced_output, index=False)
        logger.info(f"[SUCCESS] Saved complete enhanced dataset with differenced columns as {enhanced_output}")
        print("\nSummary of differencing:")
        for col in diff_tracker:
            print(f"  {col}: {diff_tracker[col]} difference(s) applied.")
        print(f"\nDifferenced data saved as {differenced_output}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        output_capture['errors'].append(str(e))
        raise
    finally:
        # Collect all print outputs
        stdout_contents = stdout_buffer.getvalue()
        if stdout_contents:
            output_capture['prints'].extend(stdout_contents.strip().split('\n'))
        stderr_contents = stderr_buffer.getvalue()
        if stderr_contents:
            output_capture['errors'].extend(stderr_contents.strip().split('\n'))
        # Save JSON output
        try:
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_capture, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save run output JSON: {str(e)}")

if __name__ == "__main__":
    main()
