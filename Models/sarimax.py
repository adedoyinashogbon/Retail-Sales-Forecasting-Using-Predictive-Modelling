import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, Any, List
import logging
from datetime import datetime
import json
import os
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import signal
from scipy import stats
from tqdm import tqdm
import time
from statsmodels.stats.diagnostic import het_arch
from sklearn.preprocessing import StandardScaler

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
            logging.FileHandler(os.path.join(log_dir, 'sarimax.log')),
            logging.StreamHandler()
        ]
    )

def create_directories() -> dict:
    """
    Creates necessary directories for results and visualizations.
    
    Returns:
        dict: Dictionary containing paths to created directories
    """
    dirs = {
        'results': 'results/sarimax',
        'plots': 'results/sarimax/plots',
        'models': 'results/sarimax/models',
        'logs': 'logs',
        'data': 'data/processed'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    return dirs

# Initialize logging and create directories
setup_logging()
logger = logging.getLogger(__name__)
dirs = create_directories()

def load_data(file_path: str, exog_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare time series data with exogenous variables.
    
    Args:
        file_path: Path to the data file
        exog_columns: List of exogenous variable column names
        
    Returns:
        Tuple of (target DataFrame, exogenous DataFrame)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate required columns
        if 'Date' not in df.columns:
            raise ValueError("Data must contain a 'Date' column")
        if 'SalesIndex' not in df.columns:
            raise ValueError("Data must contain a 'SalesIndex' column")
            
        # Convert Date column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Ensure data is sorted by date
        df.sort_index(inplace=True)
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(df['SalesIndex']):
            raise ValueError("SalesIndex column must be numeric")
            
        # Split into target and exogenous variables
        target_df = df[['SalesIndex']]
        
        if exog_columns:
            # Validate exogenous columns exist
            missing_cols = [col for col in exog_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing exogenous columns: {missing_cols}")
            
            # Validate exogenous columns are numeric
            non_numeric_cols = [col for col in exog_columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric_cols:
                raise ValueError(f"Non-numeric exogenous columns: {non_numeric_cols}")
            
            exog_df = df[exog_columns]
        else:
            exog_df = pd.DataFrame(index=df.index)
            
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return target_df, exog_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_test_split(target_data: pd.DataFrame, exog_data: pd.DataFrame, target_col: str, train_size: float = 0.8) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Args:
        target_data: DataFrame containing the target variable
        exog_data: DataFrame containing exogenous variables
        target_col: Name of the target column
        train_size: Proportion of data to use for training
        
    Returns:
        Tuple of (y_train, y_test, X_train, X_test)
    """
    try:
        # Calculate split index
        split_idx = int(len(target_data) * train_size)
        
        # Split target variable
        y_train = target_data[target_col].iloc[:split_idx]
        y_test = target_data[target_col].iloc[split_idx:]
        
        # Split exogenous variables
        X_train = exog_data.iloc[:split_idx]
        X_test = exog_data.iloc[split_idx:]
        
        # Validate splits
        if len(y_train) < 24:  # Require at least 2 years of training data for seasonal patterns
            raise ValueError("Insufficient training data for seasonal analysis")
        if len(y_test) < 12:  # Require at least 1 year of test data
            raise ValueError("Insufficient test data")
            
        logger.info(f"Split data into {len(y_train)} training and {len(y_test)} test samples")
        return y_train, y_test, X_train, X_test
        
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise

def prepare_exog_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare exogenous variables by scaling and handling missing values.
    
    Args:
        X_train: Training exogenous variables
        X_test: Test exogenous variables
        
    Returns:
        Tuple of (prepared X_train, prepared X_test)
    """
    try:
        if X_train.empty:
            return X_train, X_test
            
        # Handle missing values first
        X_train_clean = X_train.fillna(method='ffill').fillna(method='bfill')
        X_test_clean = X_test.fillna(method='ffill').fillna(method='bfill')
        
        # Create scaler
        scaler = StandardScaler()
        
        # Scale training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_clean),
            index=X_train.index,
            columns=X_train.columns
        )
        
        # Scale test data using training scaler
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_clean),
            index=X_test.index,
            columns=X_test.columns
        )
        
        return X_train_scaled, X_test_scaled
        
    except Exception as e:
        logger.error(f"Error preparing exogenous data: {str(e)}")
        raise

def inverse_difference(original_series: pd.Series, differenced_series: pd.Series, seasonal_period: int = 12) -> pd.Series:
    """
    Convert differenced series back to original scale, handling both regular and seasonal differencing.
    
    Args:
        original_series: Original time series (needed for the first value)
        differenced_series: Differenced time series to convert back
        seasonal_period: Seasonal period (default: 12 for monthly data)
        
    Returns:
        Series in original scale
    """
    try:
        # Get the last values from the original series needed for inverse differencing
        last_values = original_series.iloc[-seasonal_period:]
        
        # Initialize the result series with the last original values
        result = pd.Series(index=differenced_series.index)
        
        # First value is the last original value plus the first differenced value
        result.iloc[0] = last_values.iloc[-1] + differenced_series.iloc[0]
        
        # Cumulatively sum the differenced values
        for i in range(1, len(differenced_series)):
            result.iloc[i] = result.iloc[i-1] + differenced_series.iloc[i]
            
        # Add the last values from original series to ensure proper alignment
        result = pd.concat([
            pd.Series(last_values.values, index=[differenced_series.index[0] - pd.DateOffset(months=i) 
                                               for i in range(seasonal_period, 0, -1)]),
            result
        ])
            
        return result
        
    except Exception as e:
        logger.error(f"Error in inverse differencing: {str(e)}")
        raise

def evaluate_model(y_true, y_pred, scale: str = "original"):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        scale: Scale of the data ("original" or "differenced")
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        metrics = {}
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE with handling of zero/negative values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_array = np.abs((y_true - y_pred) / y_true) * 100
            mape_array = mape_array[~np.isnan(mape_array) & ~np.isinf(mape_array)]
            metrics['mape'] = mape_array.mean() if len(mape_array) > 0 else float('inf')
        
        # Directional accuracy
        actual_direction = np.sign(y_true.diff().dropna())
        predicted_direction = np.sign(y_pred.diff().dropna())
        metrics['direction_accuracy'] = (actual_direction == predicted_direction).mean()
        
        # Log metrics with scale information
        logger.info(f"\nEvaluation on {scale} scale:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.3f}")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def display_summary(model, metrics_orig, metrics_diff):
    """
    Display a summary of model results in the console.
    
    Args:
        model: Fitted SARIMAX model
        metrics_orig: Dictionary of evaluation metrics for original scale
        metrics_diff: Dictionary of evaluation metrics for differenced scale
    """
    try:
        print("\nSARIMAX Model Summary")
        print("=" * 50)
        
        # Model Parameters
        print("\nModel Parameters:")
        print(f"Order: {model.order}")
        print(f"Seasonal Order: {model.seasonal_order}")
        print(f"AIC: {model.aic():.2f}")
        print(f"BIC: {model.bic():.2f}")
        
        # Performance Metrics - Original Scale
        print("\nPerformance Metrics (Original Scale):")
        print(f"MAE: {metrics_orig['mae']:.2f}")
        print(f"RMSE: {metrics_orig['rmse']:.2f}")
        print(f"R²: {metrics_orig['r2']:.2f}")
        print(f"MAPE: {metrics_orig['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics_orig['direction_accuracy']:.2%}")
        
        # Performance Metrics - Differenced Scale
        print("\nPerformance Metrics (Differenced Scale):")
        print(f"MAE: {metrics_diff['mae']:.2f}")
        print(f"RMSE: {metrics_diff['rmse']:.2f}")
        print(f"R²: {metrics_diff['r2']:.2f}")
        print(f"MAPE: {metrics_diff['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics_diff['direction_accuracy']:.2%}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Error displaying summary: {str(e)}")
        raise

def plot_forecast_comparison(y_test, forecast, conf_int, title, filename, scale="Original"):
    """
    Plots the forecast comparison with confidence intervals and improved visualization.
    
    Parameters:
        y_test: Test data
        forecast: Forecasted values
        conf_int: Confidence intervals DataFrame with 'lower' and 'upper' columns
        title: Plot title
        filename: Output filename
        scale: String indicating the scale ("Original" or "Differenced")
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Test Data', color='green', linewidth=2)
    plt.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='--', linewidth=2)
    plt.fill_between(conf_int.index, conf_int['lower'], conf_int['upper'],
                     color='red', alpha=0.1, label='95% Confidence Interval')
    
    plt.title(f'SARIMAX Forecast vs Actual ({scale} Scale)', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel(f'Sales Index ({scale} Scale)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals_analysis(residuals, title, filename, scale="Original"):
    """
    Creates comprehensive diagnostic plots for residual analysis with improved visualization.
    
    Parameters:
        residuals: Model residuals
        title: Plot title
        filename: Output filename
        scale: String indicating the scale ("Original" or "Differenced")
    """
    fig = plt.figure(figsize=(15, 12))
    
    # Residuals over time
    ax1 = plt.subplot(221)
    ax1.plot(residuals.index, residuals, color='blue', alpha=0.7, linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals Over Time', fontsize=14, pad=15)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residual', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45)
    
    # Residuals vs Fitted
    ax2 = plt.subplot(222)
    ax2.scatter(residuals.index, residuals, alpha=0.5, color='blue')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals vs Time', fontsize=14, pad=15)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45)
    
    # Q-Q Plot
    ax3 = plt.subplot(223)
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Normal Q-Q Plot', fontsize=14, pad=15)
    ax3.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
    
    # Histogram with normal distribution overlay
    ax4 = plt.subplot(224)
    ax4.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_title('Residuals Distribution', fontsize=14, pad=15)
    ax4.set_xlabel('Residual', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    xmin, xmax = ax4.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax4.plot(x, p, 'k', linewidth=2)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistical summary
    stats_text = (
        f"Mean: {np.mean(residuals):.3f}\n"
        f"Std: {np.std(residuals):.3f}\n"
        f"Skewness: {stats.skew(residuals):.3f}"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.suptitle(f'{title} ({scale} Scale)', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_visualizations(y_train, y_test, forecast_diff, forecast_orig, conf_int_diff, conf_int_orig, residuals_diff, residuals_orig):
    """
    Save model visualizations for both differenced and original scales with improved quality.
    
    Args:
        y_train: Training data
        y_test: Test data
        forecast_diff: Forecast on differenced scale
        forecast_orig: Forecast on original scale
        conf_int_diff: Confidence intervals for differenced forecast
        conf_int_orig: Confidence intervals for original forecast
        residuals_diff: Residuals on differenced scale
        residuals_orig: Residuals on original scale
    """
    try:
        # Plot differenced scale
        plt.figure(figsize=(12, 6))
        plt.plot(y_train.diff().dropna().index, y_train.diff().dropna(), 
                 label='Training Data', color='blue', linewidth=2)
        plt.plot(y_test.diff().dropna().index, y_test.diff().dropna(), 
                 label='Test Data', color='green', linewidth=2)
        plt.plot(forecast_diff.index, forecast_diff, 
                 label='Forecast', color='red', linestyle='--', linewidth=2)
        
        if conf_int_diff is not None:
            plt.fill_between(forecast_diff.index, conf_int_diff.iloc[:, 0], conf_int_diff.iloc[:, 1],
                           color='red', alpha=0.1, label='95% Confidence Interval')
        
        plt.title('SARIMAX Forecast (Differenced Scale)', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Differenced Value', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(dirs['plots'], 'sarimax_forecast_diff.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot original scale
        plt.figure(figsize=(12, 6))
        plt.plot(y_train.index, y_train, label='Training Data', color='blue', linewidth=2)
        plt.plot(y_test.index, y_test, label='Test Data', color='green', linewidth=2)
        plt.plot(forecast_orig.index, forecast_orig, 
                 label='Forecast', color='red', linestyle='--', linewidth=2)
        
        if conf_int_orig is not None:
            plt.fill_between(forecast_orig.index, conf_int_orig.iloc[:, 0], conf_int_orig.iloc[:, 1],
                           color='red', alpha=0.1, label='95% Confidence Interval')
        
        plt.title('SARIMAX Forecast (Original Scale)', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Value', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(dirs['plots'], 'sarimax_forecast_orig.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot residual analysis for both scales
        plot_residuals_analysis(
            residuals_diff,
            'SARIMAX Model Residual Analysis',
            os.path.join(dirs['plots'], 'sarimax_residuals_analysis_diff.png'),
            scale="Differenced"
        )
        
        plot_residuals_analysis(
            residuals_orig,
            'SARIMAX Model Residual Analysis',
            os.path.join(dirs['plots'], 'sarimax_residuals_analysis_orig.png'),
            scale="Original"
        )
        
        logger.info(f"Visualizations saved in {dirs['plots']}")
        
    except Exception as e:
        logger.error(f"Error saving visualizations: {str(e)}")
        raise

def save_results(model, metrics_orig, metrics_diff, residuals_orig, residuals_diff):
    """
    Save model results and metrics to JSON file.
    
    Args:
        model: Fitted SARIMAX model
        metrics_orig: Dictionary of model metrics for original scale
        metrics_diff: Dictionary of model metrics for differenced scale
        residuals_orig: Model residuals on original scale
        residuals_diff: Model residuals on differenced scale
    """
    try:
        # Prepare results dictionary
        results = {
            'metrics': {
                'differenced_scale': {
                    'mae': metrics_diff['mae'],
                    'rmse': metrics_diff['rmse'],
                    'r2': metrics_diff['r2'],
                    'mape': metrics_diff['mape'],
                    'direction_accuracy': metrics_diff['direction_accuracy'],
                    'residuals_mean': float(residuals_diff.mean()),
                    'residuals_std': float(residuals_diff.std()),
                    'residuals_skew': float(stats.skew(residuals_diff.dropna())),
                    'residuals_kurtosis': float(stats.kurtosis(residuals_diff.dropna()))
                },
                'original_scale': {
                    'mae': metrics_orig['mae'],
                    'rmse': metrics_orig['rmse'],
                    'r2': metrics_orig['r2'],
                    'mape': metrics_orig['mape'],
                    'direction_accuracy': metrics_orig['direction_accuracy'],
                    'residuals_mean': float(residuals_orig.mean()),
                    'residuals_std': float(residuals_orig.std()),
                    'residuals_skew': float(stats.skew(residuals_orig.dropna())),
                    'residuals_kurtosis': float(stats.kurtosis(residuals_orig.dropna()))
                }
            },
            'model_order': model.order,
            'model_seasonal_order': model.seasonal_order,
            'model_aic': model.aic(),
            'model_bic': model.bic()
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(dirs['results'], f"sarimax_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def inverse_difference(original_data: pd.Series, differenced_forecast: pd.Series, seasonal_period: int = 12) -> pd.Series:
    """
    Inverse differences the forecasted values to get back to the original scale.
    
    Parameters:
        original_data: Original time series data
        differenced_forecast: Forecasted values on differenced scale
        seasonal_period: Seasonal period (default 12 for monthly data)
    
    Returns:
        Forecast on original scale
    """
    last_value = original_data.iloc[-1]
    forecast_orig = pd.Series(index=differenced_forecast.index)
    
    for i in range(len(differenced_forecast)):
        if i == 0:
            forecast_orig.iloc[i] = last_value + differenced_forecast.iloc[i]
        else:
            forecast_orig.iloc[i] = forecast_orig.iloc[i-1] + differenced_forecast.iloc[i]
    
    return forecast_orig

def calculate_direction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate directional accuracy between actual and predicted values.
    
    Parameters:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Direction accuracy as a float between 0 and 1
    """
    # Calculate directions (1 for up, -1 for down, 0 for no change)
    actual_direction = np.sign(y_true.diff())
    pred_direction = np.sign(y_pred.diff())
    
    # Remove any NaN values that result from differencing
    mask = ~(actual_direction.isna() | pred_direction.isna())
    actual_direction = actual_direction[mask]
    pred_direction = pred_direction[mask]
    
    # Calculate accuracy (matching directions / total)
    direction_accuracy = (actual_direction == pred_direction).mean()
    
    return float(direction_accuracy)

def main():
    """Main execution function."""
    try:
        # Create save directory
        save_dir = "sarimax_results"
        os.makedirs(save_dir, exist_ok=True)
        
        # Define exogenous variables based on available columns
        exog_columns = ['EmploymentRate', 'UnemploymentRate', 'InflationRate']
        
        # Load data
        target_data, exog_data = load_data("original_data.csv", exog_columns)
        
        # Handle any NaN values in target data
        target_data = target_data.fillna(method='ffill').fillna(method='bfill')
        
        # Split data
        y_train, y_test, X_train, X_test = train_test_split(
            target_data,
            exog_data,
            target_col="SalesIndex",
            train_size=0.8
        )
        
        # Calculate differenced data
        y_train_diff = y_train.diff().dropna()
        y_test_diff = y_test.diff().dropna()
        
        # Store original test data without first observation (to match differenced length)
        y_test_trimmed = y_test.iloc[1:]
        
        # Prepare exogenous variables and adjust for differencing
        X_train_scaled, X_test_scaled = prepare_exog_data(X_train, X_test)
        
        # Adjust exogenous variables to match differenced data length
        X_train_scaled_diff = X_train_scaled.iloc[1:]  # Skip first row to match differenced data
        X_test_scaled_diff = X_test_scaled.iloc[1:]    # Skip first row to match differenced data
        
        # Verify no NaN values before model fitting
        if y_train_diff.isna().any():
            y_train_diff = y_train_diff.ffill().bfill()
        if X_train_scaled_diff.isna().any().any():
            X_train_scaled_diff = X_train_scaled_diff.ffill().bfill()
        
        # Fit SARIMAX model using auto_arima on differenced data
        logger.info("Fitting SARIMAX model using auto_arima on differenced data...")
        model = auto_arima(
            y_train_diff,
            exog=X_train_scaled_diff,
            start_p=1, start_q=1, start_d=1,
            max_p=5, max_q=5, max_d=2,
            m=12,  # Monthly seasonal period
            seasonal=True,
            start_P=1, start_Q=1, start_D=1,
            max_P=2, max_Q=2, max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=True,
            information_criterion='aic',
            random_state=42
        )
        
        # Generate forecast on differenced scale
        forecast_result_diff = model.predict(
            n_periods=len(X_test_scaled_diff),
            exog=X_test_scaled_diff,
            return_conf_int=True
        )
        
        # Create Series with proper index for differenced forecast
        forecast_diff = pd.Series(
            forecast_result_diff[0],
            index=y_test_diff.index,
            name='Forecast'
        ).ffill().bfill()
        
        # Create DataFrame with proper index for confidence intervals
        conf_int_diff = pd.DataFrame(
            forecast_result_diff[1],
            index=y_test_diff.index,
            columns=['lower', 'upper']
        ).ffill().bfill()
        
        # Calculate residuals on differenced scale
        residuals_diff = y_test_diff - forecast_diff
        
        # Inverse difference the forecast and confidence intervals
        forecast_orig = inverse_difference(y_train, forecast_diff)
        conf_int_orig = pd.DataFrame({
            'lower': inverse_difference(y_train, conf_int_diff['lower']),
            'upper': inverse_difference(y_train, conf_int_diff['upper'])
        })
        
        # Ensure forecast and confidence intervals are aligned with test data
        forecast_orig = forecast_orig.loc[y_test_trimmed.index]
        conf_int_orig = conf_int_orig.loc[y_test_trimmed.index]
        
        # Calculate residuals on original scale
        residuals_orig = y_test_trimmed - forecast_orig
        
        # Save visualizations for both scales
        save_visualizations(
            y_train, y_test_trimmed,
            forecast_diff, forecast_orig,
            conf_int_diff, conf_int_orig,
            residuals_diff, residuals_orig
        )
        
        # Calculate metrics for both scales
        metrics_diff = evaluate_model(y_test_diff, forecast_diff, "differenced")
        metrics_orig = evaluate_model(y_test_trimmed, forecast_orig, "original")
        
        # Save results
        save_results(model, metrics_orig, metrics_diff, residuals_orig, residuals_diff)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 