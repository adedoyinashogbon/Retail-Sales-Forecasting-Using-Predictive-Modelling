import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
from dataclasses import dataclass
import warnings
import io
import json
from datetime import datetime
import os

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
            logging.FileHandler(os.path.join(log_dir, 'eda.log')),
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
        'results': 'results/eda',
        'plots': 'results/eda/plots',
        'data': 'data/processed',
        'logs': 'logs'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    return dirs

# Initialize logging and create directories
setup_logging()
logger = logging.getLogger(__name__)
dirs = create_directories()

@dataclass
class VisualizationConfig:
    """Configuration settings for visualizations."""
    figsize: tuple = (14, 6)
    color: str = "blue"
    palette: str = "Blues"
    dpi: int = 300  # Increased DPI for better quality
    style: str = "seaborn-v0_8"
    fontsize: int = 12
    title_fontsize: int = 14
    label_fontsize: int = 12

class DataLoader:
    """Handles data loading and preprocessing operations."""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Loads the final merged dataset and ensures proper datetime formatting.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and preprocessed dataframe
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or missing required columns
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("The loaded dataset is empty")
                
            if "Date" not in df.columns:
                raise ValueError("Dataset must contain a 'Date' column")
                
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            logger.info(f"Successfully loaded data from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

class DataExplorer:
    """Handles data exploration and visualization operations."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        try:
            plt.style.use(self.config.style)
        except OSError as e:
            logger.warning(f"Could not set style '{self.config.style}': {str(e)}")
            logger.info("Using default style instead")
            plt.style.use('default')
    
    def explore_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs comprehensive data exploration including summary statistics and visualizations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, Any]: Dictionary containing exploration results
        """
        try:
            results = {}
            results["basic_info"] = self._get_basic_info(df)
            results["missing_values"] = df.isnull().sum().to_dict()
            results["summary_stats"] = df.describe().to_dict()
            
            # Generate and save visualizations
            self._plot_sales_trends(df)
            self._plot_monthly_seasonality(df)
            self._plot_correlation_heatmap(df)
            
            # Save results to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(dirs['results'], f"eda_results_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"EDA results saved to {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error during data exploration: {str(e)}")
            raise
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gets basic information about the dataset."""
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        return {
            "info": info_str,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    
    def _plot_sales_trends(self, df: pd.DataFrame) -> None:
        """Plots retail sales trends over time with improved visualization."""
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        plt.plot(df.index, df["SalesIndex"], color=self.config.color, 
                 label="Retail Sales Index", linewidth=2)
        plt.xlabel("Year", fontsize=self.config.label_fontsize, fontweight='bold')
        plt.ylabel("Sales Index", fontsize=self.config.label_fontsize, fontweight='bold')
        plt.title("Retail Sales Trends Over Time", 
                 fontsize=self.config.title_fontsize, pad=20)
        plt.legend(fontsize=self.config.fontsize, frameon=True, 
                  facecolor='white', edgecolor='none')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(dirs['plots'], f"sales_trends_{timestamp}.png"), 
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_seasonality(self, df: pd.DataFrame) -> None:
        """Plots monthly seasonality patterns with improved visualization."""
        plot_df = df.copy()
        plot_df["Month"] = plot_df.index.month
        monthly_avg_sales = plot_df.groupby("Month")["SalesIndex"].mean().reset_index()
        
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        sns.barplot(
            data=monthly_avg_sales,
            x="Month",
            y="SalesIndex",
            palette=self.config.palette
        )
        
        plt.xlabel("Month", fontsize=self.config.label_fontsize, fontweight='bold')
        plt.ylabel("Average Sales Index", fontsize=self.config.label_fontsize, fontweight='bold')
        plt.title("Average Monthly Retail Sales (Seasonality)", 
                 fontsize=self.config.title_fontsize, pad=20)
        plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                  rotation=45)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(dirs['plots'], f"monthly_seasonality_{timestamp}.png"), 
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Plots correlation heatmap for relevant features with improved visualization."""
        factor_cols = [col for col in ["SalesIndex", "EmploymentRate", "UnemploymentRate", "InflationRate"] 
                      if col in df.columns]
        
        if len(factor_cols) > 1:
            plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            correlation_matrix = df[factor_cols].corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                square=True,
                annot_kws={"size": self.config.fontsize}
            )
            plt.title("Correlation Between Retail Sales and Economic Factors", 
                     fontsize=self.config.title_fontsize, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(dirs['plots'], f"correlation_heatmap_{timestamp}.png"), 
                       dpi=self.config.dpi, bbox_inches='tight')
            plt.close()

class FeatureEngineer:
    """Handles feature engineering operations."""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features including lag features, rolling averages, and rate of change calculations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with new features
        """
        try:
            df = df.copy()
            
            # Time-based features
            df["Year"] = df.index.year
            df["Quarter"] = df.index.quarter
            
            # Lag Features
            lag_features = ["SalesIndex", "EmploymentRate", "UnemploymentRate", "InflationRate"]
            for feature in lag_features:
                if feature in df.columns:
                    df[f"{feature}_Lag1"] = df[feature].shift(1)
            
            # Rolling Averages
            for feature in lag_features:
                if feature in df.columns:
                    df[f"{feature}_MA3"] = df[feature].rolling(window=3).mean()
            
            # Rate of Change Features
            for feature in ["InflationRate", "EmploymentRate"]:
                if feature in df.columns:
                    df[f"{feature}_Change"] = df[feature].pct_change() * 100
            
            # Drop rows with NaN
            df.dropna(inplace=True)
            logger.info("Feature engineering completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise

def main() -> pd.DataFrame:
    """
    Main function to orchestrate the EDA process.
    
    Returns:
        pd.DataFrame: Enhanced dataset with new features
    """
    try:
        # Configuration
        config = VisualizationConfig()
        file_path = os.path.join(dirs['data'], "final_merged_dataset.csv")
        
        # Initialize components
        loader = DataLoader()
        explorer = DataExplorer(config)
        engineer = FeatureEngineer()
        
        # Load and process data
        df = loader.load_data(file_path)
        results = explorer.explore_data(df)
        df_enhanced = engineer.create_features(df)
        
        # Save enhanced dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(dirs['results'], f"enhanced_retail_dataset_{timestamp}.csv")
        df_enhanced.to_csv(output_path)
        logger.info(f"[SUCCESS] Enhanced dataset saved as {output_path}")
        
        return df_enhanced
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    df_enhanced = main()
