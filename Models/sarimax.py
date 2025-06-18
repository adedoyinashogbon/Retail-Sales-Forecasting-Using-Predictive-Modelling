import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import argparse
from datetime import datetime
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Directory setup
RESULTS_DIR = 'results/sarimax'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
LOGS_DIR = 'logs'
LOG_FILE = os.path.join(LOGS_DIR, 'sarimax.log')
for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()])
logger = logging.getLogger(__name__)

def load_data(file_path, target_col, exog_cols=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    if 'Date' not in df.columns:
        raise ValueError("Data must contain a 'Date' column")
    if target_col not in df.columns:
        raise ValueError(f"Data must contain the target column '{target_col}'")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    y = df[target_col].astype(float)
    X = pd.DataFrame()
    if exog_cols:
        missing = [col for col in exog_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing exogenous columns: {missing}. Continuing without them.")
        X = df[[col for col in exog_cols if col in df.columns]].astype(float)
    return y, X

def train_test_split(y, X, train_size=0.8):
    split_idx = int(len(y) * train_size)
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    X_train = X.iloc[:split_idx] if not X.empty else pd.DataFrame(index=y_train.index)
    X_test = X.iloc[split_idx:] if not X.empty else pd.DataFrame(index=y_test.index)
    return y_train, y_test, X_train, X_test

def inverse_difference(original_series, differenced_series, seasonal_period=12):
    result = pd.Series(index=differenced_series.index, dtype=float)
    result.iloc[0] = original_series.iloc[-seasonal_period] + differenced_series.iloc[0]
    for i in range(1, len(differenced_series)):
        result.iloc[i] = result.iloc[i-1] + differenced_series.iloc[i]
    return result

def evaluate_model(y_true, y_pred, scale="original"):
    mask = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if not (y_true == 0).any() else np.nan,
        'direction_accuracy': np.mean(np.sign(y_true.diff().dropna()) == np.sign(y_pred.diff().dropna()))
    }
    logger.info(f"Evaluation ({scale}): {metrics}")
    return metrics

def save_results(model, metrics_orig, metrics_diff):
    try:
        results = {
            'metrics': {
                'original_scale': metrics_orig,
                'differenced_scale': metrics_diff
            },
            'model_order': getattr(model, 'order', None),
            'model_seasonal_order': getattr(model, 'seasonal_order', None),
            'model_aic': getattr(model, 'aic', None),
            'model_bic': getattr(model, 'bic', None)
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"sarimax_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {results_file}")
        # Print results to terminal
        print("\nSARIMAX Results Summary")
        print("="*50)
        print(f"Model Order: {results['model_order']}")
        print(f"Seasonal Order: {results['model_seasonal_order']}")
        print(f"AIC: {results['model_aic']}")
        print(f"BIC: {results['model_bic']}")
        print("\nMetrics (Original Scale):")
        for k, v in metrics_orig.items():
            print(f"  {k.upper()}: {v:.4f}")
        print("\nMetrics (Differenced Scale):")
        for k, v in metrics_diff.items():
            print(f"  {k.upper()}: {v:.4f}")
        print("="*50)
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="SARIMAX Time Series Forecasting with Exogenous Variables")
    parser.add_argument('--data_csv', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--target_col', type=str, required=True, help="Target column name for forecasting")
    parser.add_argument('--exog_cols', type=str, default=None, help="Comma-separated list of exogenous variable column names (optional)")
    parser.add_argument('--train_size', type=float, default=0.8, help="Proportion of data for training (default: 0.8)")
    parser.add_argument('--already_differenced', action='store_true', help="Flag if data is already differenced")
    parser.add_argument('--fixed_model_order', type=str, default=None, help="Use specified model order as 'p,d,q' (e.g., '5,0,1')")
    parser.add_argument('--fixed_seasonal_order', type=str, default=None, help="Use specified seasonal order as 'P,D,Q,s' (e.g., '1,0,1,12')")
    args = parser.parse_args()

    exog_cols = [c.strip() for c in args.exog_cols.split(',')] if args.exog_cols else None
    y, X = load_data(args.data_csv, args.target_col, exog_cols)
    y_train, y_test, X_train, X_test = train_test_split(y, X, args.train_size)

    if args.already_differenced:
        y_train_diff, y_test_diff = y_train.copy(), y_test.copy()
        X_train_diff, X_test_diff = X_train.copy(), X_test.copy()
        logger.info("Skipping internal differencing (input is already differenced)")
    else:
        y_train_diff, y_test_diff = y_train.diff().dropna(), y_test.diff().dropna()
        X_train_diff = X_train.loc[y_train_diff.index] if not X_train.empty else pd.DataFrame(index=y_train_diff.index)
        X_test_diff = X_test.loc[y_test_diff.index] if not X_test.empty else pd.DataFrame(index=y_test_diff.index)
        logger.info("Performed internal differencing on input data")

    # MODE 1: Use fixed model order if provided
    if args.fixed_model_order and args.fixed_seasonal_order:
        order = tuple(map(int, args.fixed_model_order.split(',')))
        seasonal_order = tuple(map(int, args.fixed_seasonal_order.split(',')))
        logger.info(f"Fitting SARIMAX with fixed order {order} and seasonal order {seasonal_order}")
        model = SARIMAX(y_train_diff, exog=X_train_diff if not X_train_diff.empty else None,
                        order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
    # MODE 2: Use auto_arima to find best order
    else:
        logger.info("Fitting SARIMAX using auto_arima for order selection...")
        model_fit = auto_arima(
            y_train_diff,
            exogenous=X_train_diff if not X_train_diff.empty else None,
            start_p=1, start_q=1, start_d=1,
            max_p=5, max_q=5, max_d=2,
            m=12, seasonal=True, stepwise=True,
            suppress_warnings=True, error_action='ignore', trace=True,
            information_criterion='aic', random_state=42
        )
    n_periods = len(y_test_diff)
    forecast_result = model_fit.predict(n_periods=n_periods, exogenous=X_test_diff if not X_test_diff.empty else None, return_conf_int=True)
    forecast_diff = pd.Series(forecast_result[0], index=y_test_diff.index)
    conf_int_diff = pd.DataFrame(forecast_result[1], index=y_test_diff.index, columns=['lower', 'upper'])
    residuals_diff = y_test_diff - forecast_diff
    if args.already_differenced:
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
        forecast_orig = forecast_orig.loc[y_test.index]
        conf_int_orig = conf_int_orig.loc[y_test.index]
        residuals_orig = y_test - forecast_orig
    metrics_diff = evaluate_model(y_test_diff, forecast_diff, "differenced")
    metrics_orig = evaluate_model(y_test, forecast_orig, "original")
    save_results(model_fit, metrics_orig, metrics_diff)
    logger.info("SARIMAX modeling completed successfully")

if __name__ == "__main__":
    main()
