import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime
import json
import os
import argparse
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
            logging.FileHandler(os.path.join(log_dir, 'sarima.log')),
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
        'results': 'results/sarima',
        'plots': 'results/sarima/plots',
        'models': 'results/sarima/models',
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

class SARIMAModel:
    """
    A class to handle SARIMA model fitting and evaluation.
    """
    def __init__(self, train_size=0.8, seasonal_period=12, confidence_level=0.95, freq='ME', n_splits=5):
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be positive")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if n_splits <= 0:
            raise ValueError("n_splits must be positive")
            
        self.train_size = train_size
        self.seasonal_period = seasonal_period
        self.confidence_level = confidence_level
        self.freq = freq
        self.n_splits = n_splits
        self.model = None
        self.results = None
    
    def split_data(self, y, train_size=None):
        """
        Splits the data into training and test sets.
        """
        if train_size is None:
            train_size = self.train_size
            
        split_idx = int(len(y) * train_size)
        y_train = pd.Series(y[:split_idx])
        y_test = pd.Series(y[split_idx:])
        return y_train, y_test
    
    def tune_sarima(self, y):
        """
        Tunes SARIMA parameters using auto_arima.
        """
        try:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValueError("Input data must be a pandas Series or numpy array")
            
            if len(y) < 2:
                raise ValueError("Input data must have at least 2 observations")
            
            # Use auto_arima with stepwise search for efficiency
            model = auto_arima(
                y,
                seasonal=True,
                m=self.seasonal_period,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,  # Use stepwise search for efficiency
                max_p=3, max_q=3,
                max_d=2,
                max_P=2, max_Q=2,
                max_D=2,
                information_criterion='aic',
                random_state=42,
                n_fits=10
            )
            
            self.model = model
            return model, model.order
                
        except Exception as e:
            logger.error(f"Error tuning SARIMA model: {str(e)}")
            raise
    
    def fit_final_sarimax(self, y_train):
        """
        Fits the final SARIMA model using the best parameters.
        """
        order = self.model.order
        seasonal_order = self.model.seasonal_order

        # Try different optimization methods in case of convergence issues
        methods = ['lbfgs', 'nm', 'powell']
        best_result = None
        best_aic = np.inf
        
        # Standardize data to help with numerical stability
        y_std = (y_train - y_train.mean()) / y_train.std()
            
        # Try different initialization strategies
        for method in methods:
            try:
                # Initialize model with simpler options first
                sarima_model = SARIMAX(
                    y_std,  # Use standardized data
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    freq=self.freq
                )
                
                # Fit with current method
                result = sarima_model.fit(
                    disp=False,
                    method=method,
                    maxiter=100,  # Reduced maxiter for stability
                    cov_type='robust'
                )
                
                # Check convergence and model validity
                if not result.mle_retvals.get('converged', False):
                    logger.warning(f"Model with {method} method failed to converge")
                    continue
                    
                current_aic = result.aic
                
                if not np.isfinite(current_aic):
                    logger.warning(f"Model with {method} method produced invalid AIC")
                    continue
                
                # Update best result if this one is better
                if current_aic < best_aic:
                    best_result = result
                    best_aic = current_aic
                    logger.info(f"New best fit found using {method} method")
                
            except Exception as e:
                logger.warning(f"Fitting with {method} failed: {str(e)}")
                continue
        
        if best_result is None:
            # Try one last time with a very simple model
            try:
                logger.info("Attempting to fit a simple SARIMA(1,1,1)(1,1,1,12) model as fallback")
                simple_model = SARIMAX(
                    y_std,
                    order=(1,1,1),
                    seasonal_order=(1,1,1,12),
                ).fit(disp=False)
                best_result = simple_model
                logger.warning("Using fallback SARIMA(1,1,1)(1,1,1,12) model")
            except Exception as e:
                logger.error(f"Even fallback model failed: {str(e)}")
                raise ValueError("Could not find a converging model with any method")
            
        self.results = best_result
        
        # Log final model statistics
        logger.info("\nFinal Model Summary:")
        logger.info(f"AIC: {best_result.aic:.3f}")
        logger.info(f"BIC: {best_result.bic:.3f}")
        logger.info(f"Log-likelihood: {best_result.llf:.3f}")
    
    @staticmethod
    def evaluate_forecast(actual, predicted, scale="original"):
        """
        Evaluates forecast accuracy using multiple metrics.
        """
        metrics = {}
        
        # Ensure both series have the same index
        common_index = actual.index.intersection(predicted.index)
        actual = actual[common_index]
        predicted = predicted[common_index]
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(actual, predicted)
        metrics['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAPE with handling of zero/negative values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_array = np.abs((actual - predicted) / actual) * 100
            mape_array = mape_array[~np.isnan(mape_array) & ~np.isinf(mape_array)]
            metrics['mape'] = mape_array.mean() if len(mape_array) > 0 else float('inf')
        
        # R-squared
        metrics['r2'] = r2_score(actual, predicted)
        
        # Directional accuracy
        actual_direction = np.sign(actual.diff().dropna())
        predicted_direction = np.sign(predicted.diff().dropna())
        
        # Ensure both direction series have the same index
        common_direction_index = actual_direction.index.intersection(predicted_direction.index)
        actual_direction = actual_direction[common_direction_index]
        predicted_direction = predicted_direction[common_direction_index]
        
        metrics['direction_accuracy'] = (actual_direction == predicted_direction).mean()
        
        logger.info(f"\nEvaluation on {scale} scale:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.3f}")
            
        return metrics
    
    @staticmethod
    def inverse_difference_forecast(last_value, forecast_diff):
        """
        Converts differenced forecasts back to original scale.
        """
        forecast_original_values = []
        running_value = last_value
        for diff_val in forecast_diff:
            running_value += diff_val
            forecast_original_values.append(running_value)
        return pd.Series(forecast_original_values, index=forecast_diff.index)

def plot_residuals_analysis(residuals, title, filename):
    """
    Creates comprehensive diagnostic plots for residual analysis with improved visualization.
    
    Parameters:
        residuals: Model residuals
        title: Plot title
        filename: Output filename
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
    
    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_residuals(model_fit):
    """
    Analyzes the residuals of the fitted model.
    """
    residuals = model_fit.resid
    
    # Create comprehensive residual analysis plot
    plot_residuals_analysis(residuals, 
                          'SARIMA Model Residual Analysis',
                          os.path.join(dirs['plots'], 'sarima_residuals_analysis.png'))
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=10)
    print("\nLjung-Box Test Results:")
    print(lb_test)
    
    # Check for normality
    _, p_value = stats.normaltest(residuals)
    print(f"\nNormality Test p-value: {p_value:.4f}")
    
    return residuals

def load_data(file_path: str, target_col: str) -> pd.DataFrame:
    """
    Load and prepare time series data.
    
    Args:
        file_path: Path to the data file
        target_col: Name of the target column for forecasting
        
    Returns:
        DataFrame with prepared data
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
        if target_col not in df.columns:
            raise ValueError(f"Data must contain the target column '{target_col}'")
            
        # Convert Date column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Ensure data is sorted by date
        df.sort_index(inplace=True)
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column '{target_col}' must be numeric")
            
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_test_split(data, target_col, train_size=0.8):
    """
    Split data into training and test sets.
    
    Args:
        data: DataFrame containing the data
        target_col: Name of the target column
        train_size: Proportion of data to use for training
        
    Returns:
        Tuple of (y_train, y_test)
    """
    try:
        # Calculate split index
        split_idx = int(len(data) * train_size)
        
        # Split target variable
        y_train = data[target_col].iloc[:split_idx]
        y_test = data[target_col].iloc[split_idx:]
        
        # Validate splits
        if len(y_train) < 24:  # Require at least 2 years of training data for seasonal patterns
            raise ValueError("Insufficient training data for seasonal analysis")
        if len(y_test) < 12:  # Require at least 1 year of test data
            raise ValueError("Insufficient test data")
            
        logger.info(f"Split data into {len(y_train)} training and {len(y_test)} test samples")
        return y_train, y_test
        
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
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

def display_summary(model, metrics_diff, metrics_orig):
    """
    Display a summary of model results in the console.
    
    Args:
        model: Fitted SARIMA model
        metrics_diff: Dictionary of evaluation metrics for differenced scale
        metrics_orig: Dictionary of evaluation metrics for original scale
    """
    try:
        print("\nSARIMA Model Summary")
        print("=" * 50)
        
        # Model Parameters
        print("\nModel Parameters:")
        print(f"Order: {model.order}")
        print(f"Seasonal Order: {model.seasonal_order}")
        print(f"AIC: {model.aic():.2f}")
        print(f"BIC: {model.bic():.2f}")
        
        # Performance Metrics - Differenced Scale
        print("\nPerformance Metrics (Differenced Scale):")
        print(f"MAE: {metrics_diff['mae']:.2f}")
        print(f"RMSE: {metrics_diff['rmse']:.2f}")
        print(f"R²: {metrics_diff['r2']:.2f}")
        print(f"MAPE: {metrics_diff['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics_diff['direction_accuracy']:.2%}")
        
        # Performance Metrics - Original Scale
        print("\nPerformance Metrics (Original Scale):")
        print(f"MAE: {metrics_orig['mae']:.2f}")
        print(f"RMSE: {metrics_orig['rmse']:.2f}")
        print(f"R²: {metrics_orig['r2']:.2f}")
        print(f"MAPE: {metrics_orig['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics_orig['direction_accuracy']:.2%}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Error displaying summary: {str(e)}")
        raise

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
        
        plt.title('SARIMA Forecast (Differenced Scale)', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Differenced Value', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(dirs['plots'], 'sarima_forecast_diff.png'), dpi=300, bbox_inches='tight')
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
        
        plt.title('SARIMA Forecast (Original Scale)', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Value', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(dirs['plots'], 'sarima_forecast_orig.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot residual analysis for both scales
        plot_residuals_analysis(
            residuals_diff,
            'SARIMA Model Residual Analysis (Differenced Scale)',
            os.path.join(dirs['plots'], 'sarima_residuals_analysis_diff.png')
        )
        
        plot_residuals_analysis(
            residuals_orig,
            'SARIMA Model Residual Analysis (Original Scale)',
            os.path.join(dirs['plots'], 'sarima_residuals_analysis_orig.png')
        )
        
        logger.info(f"Visualizations saved in {dirs['plots']}")
        
    except Exception as e:
        logger.error(f"Error saving visualizations: {str(e)}")
        raise

def save_results(model, metrics_diff, metrics_orig, residuals_diff, residuals_orig):
    """
    Save model results and metrics to JSON file.
    
    Args:
        model: Fitted SARIMA model
        metrics_diff: Dictionary of model metrics for differenced scale
        metrics_orig: Dictionary of model metrics for original scale
        residuals_diff: Model residuals on differenced scale
        residuals_orig: Model residuals on original scale
    """
    try:
        # Prepare results dictionary
        results = {
            "model_info": {
                "order": model.order,
                "seasonal_order": model.seasonal_order,
                "aic": float(model.aic()),
                "bic": float(model.bic())
            },
            "metrics": {
                "differenced_scale": metrics_diff,
                "original_scale": metrics_orig
            },
            "residuals": {
                "differenced_scale": {
                    "mean": float(residuals_diff.mean()),
                    "std": float(residuals_diff.std()),
                    "skewness": float(residuals_diff.skew()),
                    "kurtosis": float(residuals_diff.kurtosis())
                },
                "original_scale": {
                    "mean": float(residuals_orig.mean()),
                    "std": float(residuals_orig.std()),
                    "skewness": float(residuals_orig.skew()),
                    "kurtosis": float(residuals_orig.kurtosis())
                }
            }
        }
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(dirs['results'], f"sarima_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def inverse_difference(y_train, forecast, seasonal_period=12):
    """
    Inverse difference the forecast.
    
    Args:
        y_train (pd.Series): Training data.
        forecast (pd.Series): Forecast on differenced scale.
        seasonal_period (int): Seasonal period (default: 12).
    
    Returns:
        pd.Series: Forecast on original scale.
    """
    try:
        # Inverse difference the forecast
        forecast_orig = y_train.iloc[-seasonal_period:] + forecast
        return forecast_orig
    except Exception as e:
        logger.error(f"Error inverting difference: {str(e)}")
        raise

def plot_residuals_analysis(residuals, title, file_path):
    """
    Plot residual analysis.
    
    Args:
        residuals (pd.Series): Residuals.
        title (str): Plot title.
        file_path (str): File path to save plot.
    """
    try:
        # Plot residuals
        plt.figure(figsize=(12, 6))
        plt.plot(residuals.index, residuals, label='Residuals', color='blue', linewidth=2)
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Residual', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting residuals: {str(e)}")
        raise

def main():
    """
    Main execution function for SARIMA model.
    Allows choosing between raw and pre-differenced input data.
    """
    try:
        # Argument parsing for flexible data input
        parser = argparse.ArgumentParser(description="SARIMA Forecasting Model")
        parser.add_argument('--data_csv', type=str, default=None,
                            help="Path to input CSV (differenced or raw). If not set, will auto-detect.")
        parser.add_argument('--already_differenced', action='store_true',
                            help="Set if the input file is already differenced (skip internal differencing).")
        parser.add_argument('--target_col', type=str, default="SalesIndex",
                            help="Target column for forecasting (default: SalesIndex)")
        parser.add_argument('--train_size', type=float, default=0.8,
                            help="Proportion of data for training (default: 0.8)")
        args, _ = parser.parse_known_args()

        # Create save directory
        save_dir = "sarima_results"
        os.makedirs(save_dir, exist_ok=True)

        # Auto-detect input file if not specified
        default_diff = "differenced_data.csv"
        default_raw = "original_data.csv"
        if args.data_csv is not None:
            input_file = args.data_csv
        elif os.path.exists(default_diff):
            input_file = default_diff
        else:
            input_file = default_raw

        # Heuristic: If using differenced_data.csv, set already_differenced True unless overridden
        already_differenced = args.already_differenced or (input_file == default_diff)
        logger.info(f"Using input file: {input_file}")
        logger.info(f"Input is already differenced: {already_differenced}")

        # Load data
        data = load_data(input_file, args.target_col)

        # Split data
        y_train, y_test = train_test_split(
            data,
            target_col=args.target_col,
            train_size=args.train_size
        )

        if already_differenced:
            # Use as-is
            y_train_diff = y_train.copy()
            y_test_diff = y_test.copy()
            logger.info("Skipping internal differencing (input is already differenced)")
        else:
            # Perform internal differencing
            y_train_diff = y_train.diff().dropna()
            y_test_diff = y_test.diff().dropna()
            logger.info("Performed internal differencing on input data")

        # Fit SARIMA model using auto_arima on differenced data
        logger.info("Fitting SARIMA model using auto_arima on differenced data...")
        model = auto_arima(
            y_train_diff,
            start_p=1,
            start_q=1,
            start_d=1,
            max_p=5,
            max_q=5,
            max_d=2,
            m=12,  # Monthly seasonality
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=True,
            information_criterion='aic',
            random_state=42
        )

        # Generate forecast on differenced scale
        forecast_result_diff = model.predict(n_periods=len(y_test_diff), return_conf_int=True)
        forecast_diff = forecast_result_diff[0]
        conf_int_diff = pd.DataFrame(forecast_result_diff[1], columns=['lower', 'upper'])

        # Ensure forecast dates match test data dates
        forecast_diff.index = y_test_diff.index
        conf_int_diff.index = y_test_diff.index

        # Calculate residuals on differenced scale
        residuals_diff = y_test_diff - forecast_diff

        # Inverse difference the forecast if necessary
        if already_differenced:
            forecast_orig = forecast_diff.copy()
            conf_int_orig = conf_int_diff.copy()
            residuals_orig = residuals_diff.copy()
            logger.info("No inverse differencing performed (input is already differenced)")
        else:
            forecast_orig = inverse_difference(y_train, forecast_diff)
            conf_int_orig = pd.DataFrame({
                'lower': inverse_difference(y_train, conf_int_diff['lower']),
                'upper': inverse_difference(y_train, conf_int_diff['upper'])
            })
            # Ensure forecast and confidence intervals are aligned with test data
            forecast_orig = forecast_orig.loc[y_test.index]
            conf_int_orig = conf_int_orig.loc[y_test.index]
            residuals_orig = y_test - forecast_orig

        # Save visualizations for both scales
        save_visualizations(
            y_train, y_test,
            forecast_diff, forecast_orig,
            conf_int_diff, conf_int_orig,
            residuals_diff, residuals_orig
        )

        # Calculate and save metrics for both scales
        metrics_diff = evaluate_model(y_test_diff, forecast_diff, "differenced")
        metrics_orig = evaluate_model(y_test, forecast_orig, "original")

        # Save results
        save_results(model, metrics_diff, metrics_orig, residuals_diff, residuals_orig)

        # Display summary in console
        display_summary(model, metrics_diff, metrics_orig)

        logger.info("SARIMA modeling completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
