#!/usr/bin/env python
"""
lstm_forecasting_cv.py

LSTM forecasting model for retail sales using CSV data with Time Series Cross-Validation.
It includes:
- Reproducibility with random seeds.
- Separate scaling for features and target.
- Command-line parameters for key hyperparameters.
- EarlyStopping and ModelCheckpoint callbacks.
- Logging for traceability.
- Option for performing cross-validation using TimeSeriesSplit.
- Enhanced residual analysis with statistical tests and visualizations.
- Saving results to JSON file for further analysis.

This version builds upon the previous template and adds cross-validation functionality.
"""

import argparse
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
            logging.FileHandler(os.path.join(log_dir, 'lstm.log'), encoding='utf-8'),
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
        'results': 'results/lstm',
        'plots': 'results/lstm/plots',
        'models': 'results/lstm/models',
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

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file, parses the 'Date' column as datetime, and sets it as the index.
    """
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    return df

def train_test_split_lstm(df: pd.DataFrame, train_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets based on the provided ratio.
    """
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    return train_df, test_df

def create_sequences(features: np.ndarray, target: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences for LSTM training.
    :param features: 2D numpy array of features (samples, num_features)
    :param target: 2D numpy array of target values (samples, 1)
    :param seq_length: Length of the lookback window
    :return: Tuple of X (shape: (samples, seq_length, num_features)) and y (shape: (samples, 1))
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length, :])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

def evaluate_forecast(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict:
    """
    Evaluates the forecast using MAE, RMSE, MAPE, R², and Direction Accuracy.
    Returns a dictionary with the metrics.
    """
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_arr = np.abs((true_vals - pred_vals) / true_vals) * 100
        mape_arr = mape_arr[~np.isnan(mape_arr) & ~np.isinf(mape_arr)]
        mape = mape_arr.mean() if len(mape_arr) > 0 else float('inf')
    r2 = r2_score(true_vals, pred_vals)
    
    # Calculate Direction Accuracy
    true_direction = np.diff(true_vals.flatten())
    pred_direction = np.diff(pred_vals.flatten())
    direction_accuracy = np.mean((true_direction * pred_direction) > 0) * 100
    
    logging.info(f"  MAE   = {mae:.3f}")
    logging.info(f"  RMSE  = {rmse:.3f}")
    logging.info(f"  MAPE  = {mape:.3f}%")
    logging.info(f"  R²    = {r2:.3f}")
    logging.info(f"  Direction Accuracy = {direction_accuracy:.2f}%")
    
    return {
        "MAE": mae, 
        "RMSE": rmse, 
        "MAPE": mape, 
        "R2": r2,
        "Direction_Accuracy": direction_accuracy
    }

def prepare_tableau_data(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.DatetimeIndex, 
                         residuals: np.ndarray = None, confidence_intervals: dict = None) -> pd.DataFrame:
    """
    Prepares data for Tableau visualization.
    
    Parameters:
        y_true: Actual values
        y_pred: Predicted values
        dates: Datetime index
        residuals: Optional residuals array
        confidence_intervals: Optional dictionary with 'lower' and 'upper' bounds
        
    Returns:
        DataFrame ready for Tableau
    """
    # Ensure all arrays are flattened and have the same length
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Create base DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Add residuals if provided
    if residuals is not None:
        df['Residuals'] = residuals.flatten()
    
    # Add confidence intervals if provided
    if confidence_intervals is not None:
        df['Lower_Bound'] = confidence_intervals['lower'].flatten()
        df['Upper_Bound'] = confidence_intervals['upper'].flatten()
    
    # Calculate percentage error
    df['Percentage_Error'] = ((df['Actual'] - df['Predicted']) / df['Actual'] * 100)
    
    # Add rolling statistics
    df['Rolling_MA_Actual'] = df['Actual'].rolling(window=7).mean()
    df['Rolling_MA_Predicted'] = df['Predicted'].rolling(window=7).mean()
    
    return df

def plot_forecast(true_vals: np.ndarray, pred_vals: np.ndarray, dates: pd.DatetimeIndex = None,
                  title: str = "LSTM Forecast vs Actual", save_path: str = None) -> None:
    """
    Plots actual vs. forecasted values with enhanced visualization.
    
    Parameters:
        true_vals: Actual values
        pred_vals: Predicted values
        dates: Optional datetime index for x-axis
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Plot actual and predicted values
    if dates is not None:
        plt.plot(dates, true_vals, label="Actual", color="green", linewidth=2)
        plt.plot(dates, pred_vals, label="Forecast", color="red", linestyle="--", linewidth=2)
        plt.xlabel("Date", fontsize=12)
    else:
        plt.plot(true_vals, label="Actual", color="green", linewidth=2)
        plt.plot(pred_vals, label="Forecast", color="red", linestyle="--", linewidth=2)
        plt.xlabel("Time Steps", fontsize=12)
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("Sales Index", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics as text
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    mape = np.mean(np.abs((true_vals - pred_vals) / true_vals)) * 100
    r2 = r2_score(true_vals, pred_vals)
    
    metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Forecast plot saved to: {save_path}")
    else:
        plt.show()

def plot_training_history(history, title: str = "Model Training History", save_path: str = None) -> None:
    """
    Plots training and validation loss history with enhanced visualization.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    metrics_text = f"Final Training Loss: {final_train_loss:.4f}\nFinal Validation Loss: {final_val_loss:.4f}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_residual_diagnostics(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                            dates: pd.DatetimeIndex = None, save_path: str = None) -> None:
    """
    Creates comprehensive diagnostic plots for residual analysis with enhanced visualization.
    
    Parameters:
        residuals: Array of model residuals
        y_true: True values
        y_pred: Predicted values
        dates: DatetimeIndex for x-axis of residuals over time plot
        save_path: Optional path to save the plot
    """
    residuals = residuals.flatten()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    fig = plt.figure(figsize=(15, 12))
    
    # Residuals over time
    ax1 = plt.subplot(221)
    if dates is not None:
        ax1.plot(dates, residuals, color='blue', alpha=0.7)
        ax1.set_xlabel('Date', fontsize=12)
        plt.xticks(rotation=45)
    else:
        ax1.plot(residuals, color='blue', alpha=0.7)
        ax1.set_xlabel('Time', fontsize=12)
    ax1.set_title('Residuals Over Time', fontsize=14, pad=15)
    ax1.set_ylabel('Residual', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs Fitted
    ax2 = plt.subplot(222)
    ax2.scatter(y_pred, residuals, alpha=0.5, color='blue')
    ax2.set_title('Residuals vs Fitted Values', fontsize=14, pad=15)
    ax2.set_xlabel('Fitted Values', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Q-Q Plot
    ax3 = plt.subplot(223)
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Normal Q-Q Plot', fontsize=14, pad=15)
    
    # Histogram with normal distribution overlay
    ax4 = plt.subplot(224)
    ax4.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_title('Residuals Distribution', fontsize=14, pad=15)
    ax4.set_xlabel('Residual', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    xmin, xmax = ax4.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax4.plot(x, p, 'k', linewidth=2)
    ax4.grid(True, alpha=0.3)
    
    # Add statistical summary
    stats_text = f"Mean: {np.mean(residuals):.3f}\nStd: {np.std(residuals):.3f}\nSkewness: {stats.skew(residuals):.3f}"
    fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_visualizations(results_dir: str = "lstm_results") -> str:
    """
    Creates a directory for saving visualizations if it doesn't exist.
    
    Parameters:
        results_dir: Directory name for saving visualizations
    Returns:
        Path to the visualizations directory
    """
    viz_dir = os.path.join(results_dir, "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir

def build_model(seq_length: int, num_features: int, lstm_units: int, dropout_rate: float) -> Sequential:
    """
    Builds the stacked LSTM forecasting model.
    This version always uses two LSTM layers.
    :param seq_length: Length of the input sequence.
    :param num_features: Number of input features.
    :param lstm_units: Number of LSTM units in each layer.
    :param dropout_rate: Dropout rate (set to 0.0 if not desired).
    :return: Compiled Keras Sequential model.
    """
    model = Sequential()
    # First LSTM layer with return_sequences=True for stacking
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, num_features)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    # Second LSTM layer
    model.add(LSTM(lstm_units, return_sequences=False))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    # Compile with optimized settings
    model.compile(
        optimizer="adam", 
        loss="mse",
        jit_compile=True,  # Enable XLA compilation
        run_eagerly=False   # Disable eager execution
    )
    
    return model

def safe_predict(model: Sequential, X: np.ndarray) -> np.ndarray:
    """
    Safely makes predictions using the model, handling tensor conversions.
    
    Parameters:
        model: Trained Keras model
        X: Input features array
        
    Returns:
        Numpy array of predictions
    """
    predictions = model.predict(X, verbose=0)
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
    return predictions

def analyze_residuals(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Performs comprehensive residual analysis including statistical tests and visualizations.
    
    Parameters:
        residuals: Array of model residuals
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing residual analysis results
    """
    try:
        # Ensure residuals are 1D array
        residuals = residuals.flatten()
        
        # Basic statistics
        results = {
            "basic_stats": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals))
            }
        }
        
        # Normality test
        _, p_value = stats.normaltest(residuals)
        results["normality_test"] = {
            "p_value": float(p_value)
        }
        
        # Autocorrelation tests
        lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
        results["ljung_box_test"] = {
            str(lag): float(lb_test.loc[lag, 'lb_pvalue']) 
            for lag in [10, 20, 30]
        }
        
        # Log results
        logging.info("\nResidual Analysis:")
        logging.info(f"Mean residual: {results['basic_stats']['mean']:.3f}")
        logging.info(f"Std residual: {results['basic_stats']['std']:.3f}")
        logging.info(f"Skewness: {results['basic_stats']['skewness']:.3f}")
        logging.info(f"Kurtosis: {results['basic_stats']['kurtosis']:.3f}")
        logging.info(f"Normality test p-value: {results['normality_test']['p_value']:.3f}")
        logging.info("\nLjung-Box test p-values:")
        for lag, p_value in results['ljung_box_test'].items():
            logging.info(f"Lag {lag}: {p_value:.3f}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in residual analysis: {str(e)}")
        raise

def calculate_feature_importance(model: Sequential, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
    Calculates feature importance using permutation method.
    
    Parameters:
        model: Trained Keras model
        X: Input features array
        y: Target values
        feature_names: List of feature names
        
    Returns:
        Dictionary containing feature importance scores and analysis
    """
    try:
        # Get baseline performance
        baseline_pred = safe_predict(model, X)
        baseline_mse = mean_squared_error(y, baseline_pred)
        
        # Calculate importance for each feature
        importance_scores = {}
        for i in range(X.shape[2]):
            # Create a copy of the input data
            X_permuted = X.copy()
            # Permute the i-th feature
            X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
            
            # Get predictions with permuted feature
            permuted_pred = safe_predict(model, X_permuted)
            permuted_mse = mean_squared_error(y, permuted_pred)
            
            # Calculate importance score (increase in MSE)
            importance = permuted_mse - baseline_mse
            importance_scores[feature_names[i]] = float(importance)
        
        # Sort features by importance
        sorted_importance = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        # Log results
        logging.info("\nFeature Importance Analysis:")
        logging.info("Features sorted by importance (higher score = more important):")
        for feature, score in sorted_importance.items():
            logging.info(f"{feature}: {score:.6f}")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        features = list(sorted_importance.keys())
        scores = list(sorted_importance.values())
        
        plt.bar(range(len(features)), scores)
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.title('Feature Importance Scores', fontsize=14, pad=15)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score (MSE Increase)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_visualizations(), "feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Feature importance plot saved to: {save_path}")
        
        return {
            "importance_scores": sorted_importance,
            "baseline_mse": float(baseline_mse),
            "visualization_path": save_path
        }
        
    except Exception as e:
        logging.error(f"Error in feature importance calculation: {str(e)}")
        raise

def save_results(results: dict, results_dir: str = None, filename: str = None) -> None:
    """
    Saves the results dictionary to a JSON file with improved organization.
    
    Args:
        results: Dictionary containing model results
        results_dir: Directory to save results (uses default if None)
        filename: Optional filename (if None, generates timestamped filename)
    """
    # Use default results directory if none provided
    if results_dir is None:
        results_dir = "lstm_results"
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lstm_results_{timestamp}.json"
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert all numpy types in the results dictionary
    results_converted = json.loads(
        json.dumps(results, default=convert_numpy)
    )
    
    results_path = os.path.join(results_dir, filename)
    with open(results_path, 'w') as f:
        json.dump(results_converted, f, indent=4)
    
    logging.info(f"Results saved to {results_path}")

def cross_validate_model(X: np.ndarray, y: np.ndarray, args, dirs: dict) -> None:
    """
    Performs time series cross-validation using TimeSeriesSplit.
    Builds and trains a new model for each fold, evaluates performance, and prints average metrics.
    """
    tscv = TimeSeriesSplit(n_splits=args.cv_splits)
    fold_metrics = []
    fold_residuals = []
    fold_dates = []
    fold_results = []
    fold = 1
    
    for train_index, val_index in tscv.split(X):
        logging.info(f"\n--- Starting fold {fold} ---")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Store dates for this fold
        fold_dates.extend(args.dates[val_index])

        model = build_model(seq_length=args.seq_length, num_features=X.shape[2],
                            lstm_units=args.lstm_units, dropout_rate=args.dropout_rate)
        logging.info(f"Model summary for fold {fold}:")
        model.summary(print_fn=logging.info)

        # Set up callbacks for each fold
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint_filepath = os.path.join(dirs['models'], f"best_model_fold{fold}.keras")
        model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, verbose=1)

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stop, model_checkpoint],
            verbose=1
        )

        # Predict and evaluate
        y_val_pred_scaled = safe_predict(model, X_val_fold)
        y_val_pred = args.scaler_target.inverse_transform(y_val_pred_scaled)
        y_val_true = args.scaler_target.inverse_transform(y_val_fold)
        
        logging.info(f"\nFold {fold} performance:")
        metrics = evaluate_forecast(y_val_true, y_val_pred)
        fold_metrics.append(metrics)
        
        # Store residuals for analysis
        fold_residuals.append(y_val_true - y_val_pred)
        
        # Store fold results
        fold_results.append({
            "fold": fold,
            "metrics": metrics,
            "training_history": {
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            }
        })
        
        # Plot training history for this fold
        fold_history_path = os.path.join(dirs['plots'], f"fold_{fold}_training_history.png")
        plot_training_history(history,
                            title=f"Model Training History - Fold {fold}",
                            save_path=fold_history_path)
        logging.info(f"Training history plot saved to: {fold_history_path}")
        
        fold += 1

    # Calculate and print average metrics
    avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
    
    logging.info("\nCross-Validation Results:")
    logging.info("Average Performance:")
    logging.info(f"  MAE   = {avg_metrics['MAE']:.3f} (±{std_metrics['MAE']:.3f})")
    logging.info(f"  RMSE  = {avg_metrics['RMSE']:.3f} (±{std_metrics['RMSE']:.3f})")
    logging.info(f"  MAPE  = {avg_metrics['MAPE']:.3f}% (±{std_metrics['MAPE']:.3f})")
    logging.info(f"  R²    = {avg_metrics['R2']:.3f} (±{std_metrics['R2']:.3f})")
    logging.info(f"  Direction Accuracy = {avg_metrics['Direction_Accuracy']:.2f}% (±{std_metrics['Direction_Accuracy']:.2f})")
    
    # Combine residuals from all folds for overall analysis
    all_residuals = np.concatenate(fold_residuals)
    all_dates = pd.DatetimeIndex(fold_dates)
    logging.info("\nPerforming residual analysis on all folds...")
    residual_analysis = analyze_residuals(all_residuals, 
                                        np.concatenate([y_val_true for _ in range(args.cv_splits)]),
                                        np.concatenate([y_val_pred for _ in range(args.cv_splits)]))
    
    # Plot residual diagnostics with dates
    plot_residual_diagnostics(all_residuals,
                            np.concatenate([y_val_true for _ in range(args.cv_splits)]),
                            np.concatenate([y_val_pred for _ in range(args.cv_splits)]),
                            dates=all_dates,
                            save_path=os.path.join(dirs['plots'], "residual_diagnostics.png"))
    
    # Save cross-validation results
    results = {
        "model_parameters": {
            "seq_length": args.seq_length,
            "lstm_units": args.lstm_units,
            "dropout_rate": args.dropout_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "cv_splits": args.cv_splits
        },
        "cross_validation": {
            "fold_results": fold_results,
            "average_metrics": avg_metrics,
            "std_metrics": std_metrics
        },
        "residual_analysis": residual_analysis,
        "visualization_paths": {
            "residuals": os.path.join(dirs['plots'], "residual_diagnostics.png"),
            "fold_histories": [os.path.join(dirs['plots'], f"fold_{i}_training_history.png") 
                             for i in range(1, args.cv_splits + 1)]
        }
    }
    
    # Save results to JSON
    save_results(results)

def main(args):
    """
    Main function to orchestrate the LSTM modeling process.
    """
    try:
        # Create results directory
        results_dir = dirs['results']
        logger.info(f"Results will be saved to: {results_dir}")
        
        # Load the data
        logger.info("Loading data...")
        original_df = load_data(os.path.join(dirs['data'], args.data_csv))
        
        # Store dates for later use
        args.dates = original_df.index
        
        # Get feature names (excluding the target column)
        feature_names = original_df.columns[:-1].tolist()
        
        # Log model parameters and configuration
        logger.info("\n=== LSTM Model Configuration ===")
        logger.info(f"Data file: {args.data_csv}")
        logger.info(f"Training size: {args.train_size}")
        logger.info(f"Sequence length: {args.seq_length}")
        logger.info(f"Number of epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"LSTM units: {args.lstm_units}")
        logger.info(f"Dropout rate: {args.dropout_rate}")
        logger.info(f"Cross-validation enabled: {args.cv}")
        if args.cv:
            logger.info(f"Number of CV splits: {args.cv_splits}")
        logger.info("================================\n")
        
        # Create directory for visualizations
        viz_dir = dirs['plots']
        logger.info(f"Visualizations will be saved to: {viz_dir}")
        
        if args.cv:
            # If cross-validation is enabled, prepare the entire dataset
            logger.info("\n=== Starting Cross-Validation Process ===")
            logger.info(f"Total dataset shape: {original_df.shape}")
            logger.info(f"Number of features: {len(feature_names)}")
            logger.info(f"Feature names: {feature_names}")
            
            # Prepare features and target for the entire dataset
            features = original_df.iloc[:, :-1].values
            target = original_df.iloc[:, -1].values.reshape(-1, 1)
            
            # Scale the entire dataset
            scaler_features = MinMaxScaler()
            scaler_target = MinMaxScaler()
            features_scaled = scaler_features.fit_transform(features)
            target_scaled = scaler_target.fit_transform(target)
            args.scaler_target = scaler_target
            
            # Create sequences for the entire dataset
            X, y = create_sequences(features_scaled, target_scaled, seq_length=args.seq_length)
            logger.info(f"Sequenced data shape for CV: X: {X.shape}, y: {y.shape}")
            
            # Perform cross-validation
            cross_validate_model(X, y, args, dirs)
            
        else:
            # Regular train-test split mode
            logger.info("Splitting data into training and testing sets...")
            train_df, test_df = train_test_split_lstm(original_df, train_size=args.train_size)

            # Separate features and target
            train_features = train_df.iloc[:, :-1].values
            train_target = train_df.iloc[:, -1].values.reshape(-1, 1)
            test_features = test_df.iloc[:, :-1].values
            test_target = test_df.iloc[:, -1].values.reshape(-1, 1)

            # Scale features and target separately
            logger.info("Scaling features and target...")
            scaler_features = MinMaxScaler()
            scaler_target = MinMaxScaler()
            train_features_scaled = scaler_features.fit_transform(train_features)
            test_features_scaled = scaler_features.transform(test_features)
            train_target_scaled = scaler_target.fit_transform(train_target)
            test_target_scaled = scaler_target.transform(test_target)

            args.scaler_target = scaler_target

            # Create sequences for LSTM training
            logger.info("Creating sequences...")
            X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, seq_length=args.seq_length)
            X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, seq_length=args.seq_length)
            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Build the model
            logger.info("Building LSTM model...")
            model = build_model(seq_length=args.seq_length, num_features=X_train.shape[2],
                              lstm_units=args.lstm_units, dropout_rate=args.dropout_rate)
            model.summary(print_fn=logging.info)

            # Set up callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            checkpoint_filepath = os.path.join(results_dir, "best_model.keras")
            model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, verbose=1)

            # Train the model
            logger.info("\n=== Starting Model Training ===")
            history = model.fit(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop, model_checkpoint],
                verbose=1
            )
            
            # Log training history summary
            logger.info("\n=== Training History Summary ===")
            logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
            logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
            logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

            # Generate predictions
            logger.info("\n=== Generating Predictions ===")
            train_pred_scaled = safe_predict(model, X_train)
            train_pred = scaler_target.inverse_transform(train_pred_scaled)
            train_true = scaler_target.inverse_transform(y_train)
            
            test_pred_scaled = safe_predict(model, X_test)
            test_pred = scaler_target.inverse_transform(test_pred_scaled)
            test_true = scaler_target.inverse_transform(y_test)
            
            # Combine predictions and actual values
            all_true = np.vstack([train_true, test_true])
            all_pred = np.vstack([train_pred, test_pred])
            
            # Calculate residuals
            all_residuals = all_true - all_pred
            
            # Calculate confidence intervals (95%)
            std_residuals = np.std(all_residuals)
            confidence_intervals = {
                'lower': all_pred - 1.96 * std_residuals,
                'upper': all_pred + 1.96 * std_residuals
            }

            # Prepare Tableau-ready data
            logger.info("Preparing Tableau-ready data...")
            train_dates = train_df.index[args.seq_length:]
            test_dates = test_df.index[args.seq_length:]
            all_dates = pd.DatetimeIndex(train_dates.tolist() + test_dates.tolist())
            
            tableau_data = prepare_tableau_data(
                all_true, all_pred, 
                dates=all_dates,
                residuals=all_residuals,
                confidence_intervals=confidence_intervals
            )
            
            # Save Tableau data
            tableau_data_path = os.path.join(results_dir, "tableau_forecast_data.csv")
            tableau_data.to_csv(tableau_data_path, index=False)
            logger.info(f"Tableau-ready data saved to {tableau_data_path}")

            # Generate and save visualizations
            plot_forecast(test_true, test_pred, 
                        dates=test_dates,
                        title="LSTM Forecast vs Actual Values (Test Set)",
                        save_path=os.path.join(viz_dir, "forecast_vs_actual.png"))

            plot_training_history(history,
                                title="Model Training and Validation Loss",
                                save_path=os.path.join(viz_dir, "training_history.png"))
            
            plot_residual_diagnostics(all_residuals, all_true, all_pred,
                                    dates=test_dates,
                                    save_path=os.path.join(viz_dir, "residual_diagnostics.png"))

            # Calculate feature importance
            logger.info("Calculating feature importance...")
            feature_importance = calculate_feature_importance(model, X_test, y_test, feature_names)

            # Save results
            results = {
                "model_parameters": {
                    "seq_length": args.seq_length,
                    "lstm_units": args.lstm_units,
                    "dropout_rate": args.dropout_rate,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "train_size": args.train_size
                },
                "metrics": {
                    "test_set": evaluate_forecast(test_true, test_pred),
                    "full_dataset": evaluate_forecast(all_true, all_pred)
                },
                "training_history": {
                    "loss": history.history['loss'],
                    "val_loss": history.history['val_loss']
                },
                "feature_importance": feature_importance,
                "visualization_paths": {
                    "forecast": os.path.join(viz_dir, "forecast_vs_actual.png"),
                    "residuals": os.path.join(viz_dir, "residual_diagnostics.png"),
                    "training": os.path.join(viz_dir, "training_history.png"),
                    "feature_importance": feature_importance["visualization_path"],
                    "tableau_data": tableau_data_path
                }
            }
            save_results(results)
            
            logger.info("\n=== Model Training and Evaluation Complete ===")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Forecasting Model with Cross-Validation for Retail Sales")
    parser.add_argument("--data_csv", type=str, default="original_data.csv",
                         help="Path to the CSV file containing the retail sales data")
    parser.add_argument("--train_size", type=float, default=0.8,
                         help="Proportion of data to use for training (ignored if --cv is used)")
    parser.add_argument("--seq_length", type=int, default=12,
                         help="Sequence length (lookback window) for LSTM. Default is 12.")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Number of epochs for training the model")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Batch size for model training")
    parser.add_argument("--lstm_units", type=int, default=64,
                         help="Number of units in each LSTM layer")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                         help="Dropout rate (default 0.0)")
    parser.add_argument("--cv", action="store_true",
                         help="If set, perform cross-validation instead of a single train-test split")
    parser.add_argument("--cv_splits", type=int, default=3,
                         help="Number of folds for time series cross-validation (default 3)")
    
    args = parser.parse_args()
    main(args)
