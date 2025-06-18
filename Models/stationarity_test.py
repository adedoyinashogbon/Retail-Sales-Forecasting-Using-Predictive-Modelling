import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Any
from pathlib import Path
import logging
import os

# Set up visualization defaults
plt.rcParams['figure.dpi'] = 300  # Higher resolution
plt.rcParams['font.size'] = 12    # Larger base font size
plt.rcParams['font.family'] = 'serif'  # More professional font
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Set seaborn style with improved aesthetics
sns.set_theme(style="whitegrid", palette="deep")
sns.set_context("paper", font_scale=1.2)

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
            logging.FileHandler(os.path.join(log_dir, 'stationarity_test.log')),
            logging.StreamHandler()
        ]
    )

def create_directories() -> Dict[str, str]:
    """
    Creates necessary directories for results and visualizations.
    
    Returns:
        Dict[str, str]: Dictionary containing paths to created directories
    """
    # Define directory structure
    dirs = {
        'results': 'results/stationarity',
        'plots': 'results/stationarity/plots',
        'logs': 'logs',
        'data': 'data/processed'
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return dirs

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads the enhanced dataset and ensures proper datetime formatting.

    Args:
        file_path (Union[str, Path]): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded and formatted DataFrame

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or missing required columns
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The file is empty")
        
        if "Date" not in df.columns:
            raise ValueError("The file must contain a 'Date' column")
            
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        logger.info(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def adf_test(series: pd.Series) -> Dict[str, Any]:
    """
    Performs the Augmented Dickey-Fuller test.

    Args:
        series (pd.Series): Time series data to test

    Returns:
        Dict[str, Any]: Dictionary containing test results
    """
    result = adfuller(series.dropna())
    test_results = {
        "statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4]
    }
    
    print("\nADF Test Results:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values: {result[4]}")
    
    if result[1] <= 0.05:
        print("✅ The data is stationary (Reject null hypothesis).")
    else:
        print("⚠️ The data is non-stationary (Fail to reject null hypothesis). Consider differencing.")
    
    return test_results

def kpss_test(series: pd.Series) -> Dict[str, Any]:
    """
    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

    Args:
        series (pd.Series): Time series data to test

    Returns:
        Dict[str, Any]: Dictionary containing test results
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    test_results = {
        "statistic": result[0],
        "p_value": result[1],
        "critical_values": result[3]
    }
    
    print("\nKPSS Test Results:")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values: {result[3]}")
    
    if result[1] > 0.05:
        print("✅ The data is stationary (Fail to reject null hypothesis).")
    else:
        print("⚠️ The data is non-stationary (Reject null hypothesis). Consider differencing.")
    
    return test_results

def plot_series(
    df: pd.DataFrame,
    column: str,
    figsize: tuple = (12, 6),
    style: str = 'default',
    save_path: Union[str, Path, None] = None
) -> None:
    """
    Plots the given time series data with improved visualization settings.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data
        column (str): Column name to plot
        figsize (tuple): Figure size (width, height)
        style (str): Matplotlib style to use (default: 'default')
        save_path (Union[str, Path, None]): Path to save the plot (optional)
    """
    plt.style.use(style)
    plt.figure(figsize=figsize)
    
    # Create the plot with improved aesthetics
    sns.lineplot(
        data=df, 
        x=df.index, 
        y=column, 
        label=column,
        linewidth=2
    )
    
    # Customize plot appearance
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel(column, fontsize=12, fontweight='bold')
    plt.title(f"Time Series Plot for {column}", fontsize=14, pad=20)
    
    # Improve grid appearance
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()

def test_series_stationarity(
    df: pd.DataFrame,
    columns: List[str],
    output_dir: Union[str, Path, None] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Tests stationarity for multiple series and generates plots.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data
        columns (List[str]): List of column names to test
        output_dir (Union[str, Path, None]): Directory to save plots (optional)

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing test results for each series
    """
    # Create directories if not provided
    if output_dir is None:
        dirs = create_directories()
        output_dir = dirs['plots']
    
    results = {}
    
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            continue
            
        print(f"\n{'='*50}")
        print(f"Testing {column} for Stationarity...")
        print(f"{'='*50}")
        
        # Generate and save plot with improved visualization
        plot_path = os.path.join(output_dir, f"{column}_stationarity.png")
        plot_series(
            df, 
            column, 
            figsize=(12, 6),
            save_path=plot_path
        )
        
        # Perform tests
        results[column] = {
            "adf": adf_test(df[column]),
            "kpss": kpss_test(df[column])
        }
    
    return results

def main():
    """
    Main function to run stationarity tests on the retail dataset.
    """
    try:
        # Create necessary directories
        dirs = create_directories()
        
        # Define input/output paths
        import glob
        import json
        import re
        eda_dir = os.path.join('results', 'eda')
        pattern = os.path.join(eda_dir, 'enhanced_retail_dataset_*.csv')
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            raise FileNotFoundError(f"No enhanced_retail_dataset_*.csv found in {eda_dir}")
        file_path = files[0]
        logger.info(f"Using latest EDA-enhanced dataset: {file_path}")

        # Load and process data
        df = load_data(file_path)

        # Define columns to test
        columns_to_test = [
            "SalesIndex", 
            "EmploymentRate", 
            "UnemploymentRate", 
            "InflationRate"
        ]

        # Run stationarity tests
        results = test_series_stationarity(
            df,
            columns_to_test,
            output_dir=dirs['plots']
        )

        # Determine which columns require differencing
        columns_to_difference = []
        for col, tests in results.items():
            adf_p = tests['adf']['p_value']
            kpss_p = tests['kpss']['p_value']
            # ADF: p > 0.05 means non-stationary; KPSS: p <= 0.05 means non-stationary
            if (adf_p > 0.05) or (kpss_p <= 0.05):
                columns_to_difference.append(col)

        recommendation = {
            "stationarity_results": results,
            "columns_to_difference": columns_to_difference
        }
        # Save recommendation JSON
        rec_file = os.path.join(dirs['results'], 'stationarity_recommendation.json')
        with open(rec_file, 'w') as f:
            json.dump(recommendation, f, indent=4)
        logger.info("✅ Stationarity tests completed successfully")
        logger.info(f"Results saved to: {rec_file}")
        print("\nColumns that require differencing:")
        if columns_to_difference:
            for col in columns_to_difference:
                print(f"  - {col}")
        else:
            print("  None. All series appear stationary.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

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
    json_output_path = os.path.join('results', 'stationarity', f'stationarity_test_run_output_{timestamp}.json')
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
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