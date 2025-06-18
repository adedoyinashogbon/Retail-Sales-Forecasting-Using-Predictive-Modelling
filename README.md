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

## 🧑‍💻 Running Models

Run individual model scripts from the `Models/` directory:

```bash
python Models/arima.py
python Models/lstm.py
python Models/sarima.py
python Models/sarimax.py
```

Results and plots will be saved in model-specific results folders.

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
