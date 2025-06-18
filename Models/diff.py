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
            print("  ✅ Stationary (Fail to reject null).")
        else:
            print("  ⚠️ Non-stationary (Reject null). Consider further differencing or transformation.")
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
        logger.info(f"✅ Saved original dataset as {output_file}")
        
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
    Creates visualization plots for original and differenced data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        target_cols (list): List of columns to plot
        output_dir (str): Directory to save plots
    """
    for col in target_cols:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original series
        ax1.plot(df['Date'], df[col], label='Original', linewidth=2)
        ax1.set_title(f'Original {col}', fontsize=14, pad=10)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot differenced series
        diff_col = f"{col}_Diff1"
        ax2.plot(df['Date'], df[diff_col], label='Differenced', color='orange', linewidth=2)
        ax2.set_title(f'Differenced {col}', fontsize=14, pad=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_differencing.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved differencing plot for {col}")

def main():
    """
    Main function to process and difference the retail dataset.
    """
    try:
        # Create directory structure
        dirs = create_directories()
        
        # Input and output file paths
        input_file = os.path.join(dirs['data_raw'], "enhanced_retail_dataset.csv")
        original_output = os.path.join(dirs['data_processed'], "original_data.csv")
        differenced_output = os.path.join(dirs['data_processed'], "differenced_data.csv")
        enhanced_output = os.path.join(dirs['data_processed'], "enhanced_retail_dataset_with_diff.csv")

        # Load the data
        df = load_data(input_file)
        
        # Save original data first
        save_original_data(df, original_output)

        # Target columns for differencing
        target_cols = ["SalesIndex", "EmploymentRate", "UnemploymentRate", "InflationRate"]

        print("\n🔹 Original columns stationarity check:")
        for col in target_cols:
            if col in df.columns:
                adf_test(df[col], col_name=col)
                kpss_test(df[col], col_name=col)

        # Apply first differencing
        print("\n🔹 Applying 1st differencing to each non-stationary column...")
        for col in target_cols:
            if col in df.columns:
                diff_col = col + "_Diff1"
                df[diff_col] = df[col].diff()

        # Drop rows with NaN from differencing
        df_clean = df.dropna()

        # Re-check stationarity on the differenced columns
        print("\n🔹 Stationarity check on differenced columns:")
        for col in target_cols:
            diff_col = col + "_Diff1"
            if diff_col in df_clean.columns:
                adf_test(df_clean[diff_col], col_name=diff_col)
                kpss_test(df_clean[diff_col], col_name=diff_col)

        # Create visualization plots
        plot_differencing_results(df_clean, target_cols, dirs['plots'])

        # Save the datasets
        save_differenced_data(df_clean, differenced_output)
        df_clean.to_csv(enhanced_output, index=False)
        logger.info(f"[SUCCESS] Saved complete enhanced dataset with differenced columns as {enhanced_output}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
