# Retail Sales Forecasting Using Predictive Modelling

This project implements and compares multiple time series forecasting models (ARIMA, SARIMA, SARIMAX, LSTM) to predict retail sales. It includes robust evaluation, high-quality visualizations, and a comprehensive academic report.

---

## 📁 Project Structure

- `Datasets/` — Raw and processed data files (**not tracked by git**)
- `Models/` — Python scripts for each model, utilities, and results/logs
- `Results/` — Generated figures, plots, and outputs (**not tracked by git**)
- `Tek Code/` — LaTeX report, figures, presentation, and bibliography
- `Research/` — Literature review, notes, and supporting research

---

## 🚀 Key Features

- Automated training, evaluation, and result saving for ARIMA, SARIMA, SARIMAX, and LSTM models
- Consistent metrics: MAE, RMSE, R², MAPE, Direction Accuracy
- High-quality visualizations and diagnostic plots
- Modular, well-logged Python codebase
- Academic report in LaTeX (`Tek Code/Final_report.tex`)

---

## ⚙️ Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place required datasets in `Datasets/`** (see report for details)

---

## 🧑‍🔬 Preprocessing Pipeline

The project includes a robust, modular preprocessing pipeline to prepare data for modeling:

1. **Data Cleaning:**
   - `Models/clean_retail_index.py` and `Models/clean_external_factors.py`
   - Cleans raw retail and external datasets, handling missing values and formatting.
2. **EDA & Feature Engineering:**
   - `Models/eda.py`
   - Merges, explores, and enhances features; saves the enhanced dataset in `results/eda/` with a timestamp.
3. **Stationarity Testing:**
   - `Models/stationarity_test.py`
   - Loads the latest EDA output, runs ADF and KPSS tests, and recommends columns for differencing. Outputs detailed results and recommendations to `results/stationarity/`.
4. **Iterative Differencing:**
   - `Models/diff.py`
   - Reads stationarity recommendations, applies differencing iteratively (up to order 3), and re-tests for stationarity after each step. Only stops when all series are stationary or max order reached. Plots and saves all results robustly.

**To run the full preprocessing sequence:**

```bash
python Models/clean_external_factors.py
python Models/clean_retail_index.py
python Models/eda.py
python Models/stationarity_test.py
python Models/diff.py
```

- All outputs are saved with timestamps and in dedicated results folders for traceability.
- Intermediate and final datasets are in `results/eda/`, `results/stationarity/`, and `data/processed/`.
- Plots and logs are saved for each step.

---

## 🧑‍💻 Running Models

Run individual model scripts from the `Models/` directory:

```bash
python Models/arima.py
python Models/lstm.py
python Models/sarima.py
python Models/sarimax.py
```

Results and plots will be saved in model-specific results folders.

### SARIMAX: Dropping Low-Correlation Exogenous Variables

The SARIMAX script supports an optional command-line argument for exogenous variable selection:

- `--drop_failed_correlators`: If you include this flag, exogenous variables with an absolute correlation less than 0.05 with the target (on the training set) will be dropped automatically before model fitting. By default, all exogenous variables are kept and their correlations are logged for your review.

**Usage Examples:**

- **Keep all exogenous variables (default):**
  ```bash
  python Models/sarimax.py --data_csv <input.csv> --target_col <target> --exog_cols <col1,col2,...>
  ```
- **Drop low-correlation exogenous variables:**
  ```bash
  python Models/sarimax.py --data_csv <input.csv> --target_col <target> --exog_cols <col1,col2,...> --drop_failed_correlators
  ```

- The correlation threshold is 0.05 (modifiable in code).
- Correlations are computed on the training set and logged for transparency.

---

## 📊 Building the Report

Navigate to `Tek Code/` and build the LaTeX report:

```bash
cd "Tek Code"
pdflatex Final_report.tex
bibtex Final_report
pdflatex Final_report.tex
pdflatex Final_report.tex
```

---

## 📝 Notes

- Large files, datasets, and generated outputs are excluded from version control via `.gitignore`.
- For questions or contributions, see the report or contact the author.

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under [CC0 1.0 Universal](LICENSE).

---

## 📚 References

See `Tek Code/references.bib` and the final report for full academic references.

---
