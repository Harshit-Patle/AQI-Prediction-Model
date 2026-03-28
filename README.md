# 🌫️ AQI Prediction Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-success)

A machine learning project that predicts the **Air Quality Index (AQI) category** of Indian cities using only PM2.5 and NO2 sensor readings. Trained on 5 years of CPCB monitoring data (2015–2019) across 19 cities, the model classifies daily air quality into 6 categories — from *Good* to *Severe* — and ships with a standalone inference script that runs on any new CSV file.

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [Dataset Setup](#-dataset-setup)
- [Approach](#-approach)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Running Inference](#-running-inference)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Demo

```bash
$ python inference.py --input data/test.csv --output output/predictions.csv

[*] Loading input data from: data/test.csv
[*] Applying feature engineering...
[*] Loading model from: models/best_model.pkl
[*] Generating predictions...
[*] Formatting submission file...
[SUCCESS] 2993 predictions saved to output/predictions.csv
```

Sample output:

| City | StationId | Date | AQI_Bucket |
|---|---|---|---|
| Delhi | DL031 | 2020-01-01 | Very Poor |
| Mumbai | MH005 | 2020-02-15 | Satisfactory |
| Bengaluru | KA009 | 2020-06-10 | Good |
| Kolkata | WB013 | 2020-01-14 | Severe |

---

## 🧩 Problem Statement

India has some of the world's most polluted cities, yet real-time AQI monitoring is expensive and sparse. Full AQI calculation requires 12+ pollutants — most stations only measure a subset. This project explores whether **just two sensors (PM2.5 and NO2)** are enough to reliably classify daily air quality.

**Target — 6 AQI categories (India CPCB standard):**

| Category | AQI Range | Health Implication |
|---|---|---|
| Good | 0–50 | Minimal impact |
| Satisfactory | 51–100 | Minor discomfort to sensitive individuals |
| Moderate | 101–200 | Discomfort to people with lung/heart conditions |
| Poor | 201–300 | Breathing discomfort on prolonged exposure |
| Very Poor | 301–400 | Respiratory illness on prolonged exposure |
| Severe | 401+ | Serious impact even on healthy people |

---

## 📁 Project Structure

```
aqi-prediction/
│
├── inference.py            # Standalone CLI prediction script
├── requirements.txt        # Pinned dependencies
├── .gitignore
├── README.md
│
├── data/
│   ├── train.csv           # Training data (2015–2019) — download separately
│   └── test.csv            # Test data (Jan–Jul 2020) — download separately
│
├── models/
│   └── best_model.pkl      # Trained LightGBM model
│
├── notebooks/
│   ├── exploration.ipynb   # EDA — distributions, city trends, seasonal patterns
│   └── modeling.ipynb      # Feature engineering, model training & evaluation
│
└── output/
    └── predictions.csv     # Generated after running inference.py
```

---

## 📦 Dataset Setup

The dataset contains daily air quality readings from one CPCB monitoring station per city, spanning 2015–2020 across 19 Indian cities.

**Download the data from Kaggle:**

```bash
# Option 1 — Kaggle CLI
pip install kaggle
kaggle datasets download -d rohanrao/air-quality-data-in-india
unzip air-quality-data-in-india.zip -d data/
```

> Alternatively, download manually from [Kaggle: Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and place `train.csv` and `test.csv` inside the `data/` folder.

**Required columns in input CSV:**

| Column | Type | Description |
|---|---|---|
| `City` | Categorical | City name |
| `StationId` | Categorical | CPCB station ID |
| `Date` | Date | YYYY-MM-DD |
| `PM2.5` | Float | Fine particulate matter (µg/m³) |
| `NO2` | Float | Nitrogen dioxide (µg/m³) |
| `season` | Categorical | Winter / Summer / Monsoon / Post-Monsoon |

---

## 🔬 Approach

### Feature Engineering

Raw sensor data alone isn't enough — domain knowledge was used to engineer 6 additional features:

| Feature | Description | Reasoning |
|---|---|---|
| `year`, `month`, `day`, `day_of_week` | Date decomposition | Captures seasonal and annual pollution cycles |
| `is_weekend` | Binary flag | Traffic-derived NO2 drops on weekends |
| `is_north` | Binary flag | North Indian cities show drastically worse winter AQI due to stubble burning and temperature inversions |
| `pollution_ratio` | PM2.5 / (NO2 + 0.001) | Ratio of particulate to gaseous pollution cleanly separates AQI categories that raw values cannot |
| `season_encoded` | Ordinal encoding | Winter=1, Summer=2, Monsoon=3, Post-Monsoon=4 |

Missing PM2.5 and NO2 values are imputed with global training medians (`45.0` and `20.0`).

### Model Selection

Three classifiers were benchmarked using **5-fold Stratified Cross-Validation**. **Macro F1** was chosen as the metric because the dataset is heavily imbalanced — *Poor* accounts for ~46% of samples while *Very Poor* is only ~2.7%. Accuracy would be misleading; Macro F1 weights all 6 classes equally.

| Model | CV Macro F1 |
|---|---|
| Logistic Regression | 0.5042 |
| Random Forest | 0.6362 |
| **LightGBM ✅** | **0.6399** |

**LightGBM** was selected as the final model. It handles class imbalance natively via `class_weight='balanced'`, captures non-linear pollutant boundaries efficiently, and trains in seconds on 17K rows.

### Interesting Finding — COVID-19 Lockdown Signal

The test set spans Jan–Jul 2020. India's national lockdown began **March 24, 2020**, causing an unprecedented drop in pollution. Since this pattern is absent from the training data, it acts as a natural out-of-distribution test. The model picked it up through the interaction of the `month` feature and dropping PM2.5/NO2 values — cities like Amaravati shift almost entirely from *Poor* to *Good* from mid-April onwards, without any explicit lockdown feature.

---

## 📊 Results

| Metric | Value |
|---|---|
| Final Model | LightGBM (`LGBMClassifier`) |
| CV Macro F1 | **0.6399** |
| Training rows | 17,402 |
| Test rows predicted | 2,993 |
| Cities covered | 19 |
| Training period | 2015–2019 |
| Test period | Jan–Jul 2020 |

---

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/Harshit-Patle/AQI-Prediction-Model.git
cd AQI-Prediction-Model
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset** (see [Dataset Setup](#-dataset-setup))

**5. Run the notebooks in order**

```
notebooks/exploration.ipynb  →  notebooks/modeling.ipynb
```

---

## ⚙️ Running Inference

> **Note:** Before running, update `MODEL_PATH` in `inference.py` to `'models/best_model.pkl'`.

```bash
python inference.py --input data/test.csv --output output/predictions.csv
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | Path to input CSV file |
| `--output` | No | `predictions.csv` | Path for output CSV |

The output CSV will contain: `City`, `StationId`, `Date`, `AQI_Bucket`.

---

## 🔮 Future Improvements

- [ ] Add more pollutants (SO2, O3, CO) when available to improve prediction accuracy
- [ ] Train a time-series aware model (e.g. LSTM or temporal CV splits) to better handle seasonal drift
- [ ] Build a Streamlit web app for interactive city-wise AQI prediction
- [ ] Add SHAP explainability plots to visualise which features drive each prediction
- [ ] Experiment with city-specific models for North vs South India
- [ ] Add automated retraining pipeline when new CPCB data is released

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a branch** for your feature
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes
   ```bash
   git commit -m "Add: your feature description"
   ```
4. **Push** and open a **Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

Please make sure your code follows the existing style and that notebooks have all cells run with outputs saved before submitting a PR.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ☕ and too much curiosity about Indian air quality data</p>
